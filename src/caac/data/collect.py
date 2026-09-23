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

COST-ACCOUNTING FIX (see reports/ or PROGRESS.md for the full writeup): two
sources of compute used to vanish from the ledger entirely --
  1. The very first segment generated for each problem (used to seed the root
     node) was generated *before* any node existed to attribute its cost to,
     and its generation call was not wrapped in a CostMeter.track(...) block.
  2. When the oracle (compute_tree.oracle_value/_oracle_cost) chose to STOP at
     a node, it charged that choice zero cost -- even though the node's
     ``rollout_correct`` label is defined as "P(correct) of *finishing*
     greedily from here", which is not a free operation.
Both are now tracked explicitly (``root_gen_cost`` on the root node,
``finish_cost`` on every node) and consumed by compute_tree.py's oracle
functions, so the oracle's reported cost is finally apples-to-apples with
every baseline's fully-inclusive token count. This changes zero backend calls
-- it is bookkeeping only, not a re-run of the pipeline for more compute.
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

        # Fix (1/2): this segment used to be generated with no cost tracking
        # at all. It is real, unavoidable compute -- every baseline in this
        # codebase pays for it too -- so it is now tracked and stashed on the
        # root node as `root_gen_cost`, which oracle_frontier() adds back in.
        with meter.track(CostSource.REASONING) as t:
            first = backend.generate(
                prompt=prob["question"], prefix="", max_tokens=cfg.segment_tokens
            )
            t.add(tokens=first.n_tokens, forward_passes=first.n_tokens)
        root_gen_cost = CostVector(tokens=first.n_tokens, forward_passes=first.n_tokens)

        root_id = _expand(
            tree, backend, verifier, prob, first, 0, cfg, extractor, meter,
            is_root=True, root_gen_cost=root_gen_cost,
        )
        tree.roots[prob["id"]] = root_id

    return tree


def _expand(tree, backend, verifier, prob, traj, depth, cfg, extractor, meter,
            is_root=False, root_gen_cost=None):
    """Create a node for ``traj`` and recursively expand each action from it."""
    node_id = uuid.uuid4().hex[:12]

    state = ReasoningState(query_id=prob["id"], prompt=prob["question"], step=depth)
    state.trajectories.append(traj)
    features, _ = extractor.extract(state)

    rollout_correct, finish_cost = _rollout_success(backend, prob, traj, cfg, meter)
    terminal = depth >= cfg.max_depth or traj.finished

    node = TreeNode(
        node_id=node_id,
        query_id=prob["id"],
        depth=depth,
        features=[float(x) for x in features],
        rollout_correct=rollout_correct,
        n_rollouts=cfg.n_rollouts,
        is_terminal=terminal,
        finish_cost={
            "tokens": finish_cost.tokens,
            "forward_passes": finish_cost.forward_passes,
            "kv_bytes": finish_cost.kv_bytes,
            "latency_s": finish_cost.latency_s,
        },
        root_gen_cost=(
            {
                "tokens": root_gen_cost.tokens,
                "forward_passes": root_gen_cost.forward_passes,
                "kv_bytes": root_gen_cost.kv_bytes,
                "latency_s": root_gen_cost.latency_s,
            }
            if is_root and root_gen_cost is not None
            else {}
        ),
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


def _rollout_success(backend, prob, traj, cfg, meter):
    """Empirical P(correct) of finishing greedily from this trajectory.

    Fix (2/2): also returns ``finish_cost`` -- the real cost of ONE greedy
    rollout to completion from this node. This used to be computed (folded
    into ``meter``, for the run's overall total) but never attached to the
    node itself, so the oracle had no way to charge for it when choosing
    STOP. With a fixed seed + greedy decoding all n_rollouts are identical
    (see the determinism finding in PROGRESS.md), so the cost of rollout #0
    is exactly the cost of any of them -- no extra compute is spent to get
    this number, it is simply no longer thrown away.
    """
    hits = 0
    finish_cost = CostVector.zero()
    for r in range(cfg.n_rollouts):
        cur = _clone(traj)
        rollout_cost = CostVector.zero()
        for _ in range(cfg.max_rollout_segments):
            if cur.finished:
                break
            with meter.track(CostSource.REASONING) as t:
                seg = backend.generate(
                    prompt=prob["question"], prefix=cur.text, max_tokens=cfg.segment_tokens
                )
                t.add(tokens=seg.n_tokens, forward_passes=seg.n_tokens)
            cur.extend(seg)
            rollout_cost = rollout_cost + CostVector(
                tokens=seg.n_tokens, forward_passes=seg.n_tokens
            )
        if r == 0:
            finish_cost = rollout_cost
        if answers_match(cur.answer, prob["answer"]):
            hits += 1
    return hits / max(1, cfg.n_rollouts), finish_cost


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
