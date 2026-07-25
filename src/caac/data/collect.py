"""Compute-tree collection (phase P3).

Generates the offline training substrate: for each problem, run reasoning to a
decision point, expand *every* action with a few rollouts each, and recurse to a
depth limit. Each node records the features the controller would have seen, the
measured cost of each action, and the empirical success rate of finishing there.

This is the most expensive step in the project and is run once per model. What
it produces is reusable: estimators, the oracle bound, and any future allocation
policy can be developed against the saved tree without re-running rollouts.

Labels come from final answers only, so no human step-level annotation is needed.
Runs against any backend, so it can be exercised on CPU with the mock before
being pointed at a real model.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from caac.cost.accounting import CostMeter, CostSource
from caac.data.compute_tree import ComputeTree, TreeNode
from caac.eval.answer_match import answers_match
from caac.signals.features import FeatureExtractor
from caac.types import Action, CostVector, ReasoningState, Trajectory
from caac.utils.logging import get_logger

__all__ = ["CollectionConfig", "collect_tree"]

log = get_logger(__name__)


@dataclass
class CollectionConfig:
    """Knobs controlling tree size, and therefore GPU cost.

    Total rollouts scale roughly as
        n_problems * max_depth * |actions| * n_rollouts
    so these numbers are the budget dial. Start small, check the tree is
    well-formed, then scale up.
    """

    segment_tokens: int = 64
    max_depth: int = 3
    n_rollouts: int = 4
    max_rollout_segments: int = 6
    branch_temperature: float = 0.9
    verify_enabled: bool = True


def collect_tree(backend, problems, *, verifier=None, config=None) -> ComputeTree:
    """Build a ComputeTree over ``problems``.

    ``problems`` is a list of {"id", "question", "answer"} dicts. The answer is
    used only to label outcomes and is never shown to the model.
    """
    cfg = config or CollectionConfig()
    tree = ComputeTree(
        meta={
            "segment_tokens": cfg.segment_tokens,
            "n_rollouts": cfg.n_rollouts,
            "max_depth": cfg.max_depth,
        }
    )
    extractor = FeatureExtractor(max_steps=cfg.max_depth)

    for i, prob in enumerate(problems):
        log.info("collecting %d/%d id=%s", i + 1, len(problems), prob["id"])
        meter = CostMeter()
        extractor.reset()
        first = backend.generate(
            prompt=prob["question"], prefix="", max_tokens=cfg.segment_tokens
        )
        root_id = _expand(
            tree, backend, verifier, prob, first, 0, cfg, extractor, meter, is_root=True
        )
        tree.roots[prob["id"]] = root_id

    return tree


def _expand(tree, backend, verifier, prob, traj, depth, cfg, extractor, meter, is_root=False):
    """Create a node for ``traj`` and recursively expand each action from it."""
    node_id = uuid.uuid4().hex[:12]

    state = ReasoningState(query_id=prob["id"], prompt=prob["question"], step=depth)
    state.trajectories.append(traj)
    features, _ = extractor.extract(state)

    rollout_correct = _rollout_success(backend, prob, traj, cfg, meter)
    terminal = depth >= cfg.max_depth or traj.finished

    node = TreeNode(
        node_id=node_id,
        query_id=prob["id"],
        depth=depth,
        features=[float(x) for x in features],
        rollout_correct=rollout_correct,
        n_rollouts=cfg.n_rollouts,
        is_terminal=terminal,
    )

    if not terminal:
        for action in Action.spending():
            if action is Action.VERIFY and not (cfg.verify_enabled and verifier):
                continue
            children, cost = _apply_action(backend, verifier, prob, traj, action, cfg, meter)
            if not children:
                continue
            node.children[action.value] = [
                _expand(tree, backend, verifier, prob, c, depth + 1, cfg, extractor, meter)
                for c in children
            ]
            node.action_cost[action.value] = {
                "tokens": cost.tokens,
                "forward_passes": cost.forward_passes,
                "kv_bytes": cost.kv_bytes,
                "latency_s": cost.latency_s,
            }

    tree.add(node, is_root=is_root)
    return node_id


def _apply_action(backend, verifier, prob, traj, action, cfg, meter):
    """Produce the child trajectories reached by taking ``action``.

    Several children per action are kept so the value label averages over the
    stochasticity of that action rather than a single lucky rollout.
    """
    children: list[Trajectory] = []
    cost = CostVector.zero()
    n = 1 if action is Action.VERIFY else max(1, cfg.n_rollouts // 2)

    for _ in range(n):
        if action is Action.CONTINUE:
            with meter.track(CostSource.REASONING) as t:
                seg = backend.generate(
                    prompt=prob["question"], prefix=traj.text, max_tokens=cfg.segment_tokens
                )
                t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
            children.append(_clone_extend(traj, seg))
            cost = cost + CostVector(tokens=seg.n_tokens, forward_passes=seg.n_tokens)

        elif action is Action.BRANCH:
            with meter.track(CostSource.BRANCH) as t:
                seg = backend.generate(
                    prompt=prob["question"], prefix=traj.text,
                    max_tokens=cfg.segment_tokens, temperature=cfg.branch_temperature,
                )
                t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
            children.append(_clone_extend(traj, seg))
            cost = cost + CostVector(tokens=seg.n_tokens, forward_passes=seg.n_tokens)

        elif action is Action.VERIFY:
            with meter.track(CostSource.VERIFIER) as t:
                res = verifier.verify(prob["question"], traj.text)
                t.add(tokens=res.n_tokens, forward_passes=res.n_forward_passes)
            # VERIFY does not change the trace, only the belief.
            children.append(_clone(traj))
            cost = cost + CostVector(
                tokens=res.n_tokens, forward_passes=res.n_forward_passes
            )

    return children, cost


def _rollout_success(backend, prob, traj, cfg, meter) -> float:
    """Empirical P(correct) of finishing greedily from this trajectory."""
    hits = 0
    for _ in range(cfg.n_rollouts):
        cur = _clone(traj)
        for _ in range(cfg.max_rollout_segments):
            if cur.finished:
                break
            with meter.track(CostSource.REASONING) as t:
                seg = backend.generate(
                    prompt=prob["question"], prefix=cur.text, max_tokens=cfg.segment_tokens
                )
                t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
            cur.extend(seg)
        if answers_match(cur.answer, prob["answer"]):
            hits += 1
    return hits / max(1, cfg.n_rollouts)


def _clone(traj: Trajectory) -> Trajectory:
    return Trajectory(
        text=traj.text,
        token_logprobs=list(traj.token_logprobs),
        token_entropies=list(traj.token_entropies),
        answer=traj.answer,
        finished=traj.finished,
    )


def _clone_extend(traj: Trajectory, seg: Trajectory) -> Trajectory:
    child = _clone(traj)
    child.extend(seg)
    return child
