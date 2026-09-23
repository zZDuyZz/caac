"""Compute tree: the offline training substrate.

Records what *would have happened* under every action at every decision point,
for a sample of queries. Expensive to collect once, cheap to reuse -- the
artifact the project is organised around: estimators, the oracle bound, and any
future policy develop against it without re-running a single rollout.

Also makes the E0 oracle bound computable exactly, by backward induction.

COST-ACCOUNTING FIX: ``TreeNode`` now carries two extra cost fields --
``finish_cost`` (cost of actually completing a greedy rollout from this node,
used to price the STOP action honestly) and ``root_gen_cost`` (the root's own
first-segment cost, which used to be generated before any node existed to
attribute it to). Without these, the oracle's STOP action was priced at zero,
while its accuracy (``rollout_correct``) implicitly assumed a finished
rollout -- an apples-to-oranges comparison against baselines that pay for
every token they generate. See collect.py's module docstring and
PROGRESS.md for the full writeup. Old tree.json files collected before this
fix do NOT have valid finish_cost/root_gen_cost and must be re-collected.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from caac.types import Action, CostVector, CostWeights

__all__ = ["TreeNode", "ComputeTree", "oracle_value", "oracle_frontier"]


@dataclass
class TreeNode:
    """One decision point plus the outcome of each action taken from it.

    ``rollout_correct`` is the empirical P(correct) of finishing greedily from
    this node -- the label for the correctness estimator. ``children`` maps an
    action to the nodes reached by taking it.

    ``finish_cost`` is the real cost of completing ONE greedy rollout to a
    final answer from this node -- this is what must be charged when the
    oracle picks STOP, since ``rollout_correct`` is only meaningful once that
    completion has actually happened; it is not a free lookup.

    ``root_gen_cost`` is set only on root nodes: the cost of the very first
    segment generated for this query, before any node existed yet to
    attribute it to. It has to be added back in separately (in
    ``oracle_frontier``) since every path through the tree pays it exactly
    once, regardless of which actions are chosen afterward.
    """

    node_id: str
    query_id: str
    depth: int
    features: list
    rollout_correct: float
    n_rollouts: int
    children: dict = field(default_factory=dict)       # action.value -> [node_id]
    action_cost: dict = field(default_factory=dict)    # action.value -> cost dict
    is_terminal: bool = False
    finish_cost: dict = field(default_factory=dict)    # cost dict: 1 greedy rollout to completion
    root_gen_cost: dict = field(default_factory=dict)  # cost dict: root-only, first segment

    def child_ids(self, action: Action) -> list:
        return self.children.get(action.value, [])

    def cost_of(self, action: Action) -> CostVector:
        raw = self.action_cost.get(action.value)
        return CostVector(**raw) if raw else CostVector.zero()

    def finish_cost_vec(self) -> CostVector:
        return CostVector(**self.finish_cost) if self.finish_cost else CostVector.zero()

    def root_gen_cost_vec(self) -> CostVector:
        return CostVector(**self.root_gen_cost) if self.root_gen_cost else CostVector.zero()


@dataclass
class ComputeTree:
    """A collection of trees, one per query, stored flat by node id."""

    nodes: dict = field(default_factory=dict)
    roots: dict = field(default_factory=dict)  # query_id -> node_id
    meta: dict = field(default_factory=dict)

    def add(self, node: TreeNode, is_root: bool = False) -> None:
        self.nodes[node.node_id] = node
        if is_root:
            self.roots[node.query_id] = node.node_id

    def get(self, node_id: str) -> TreeNode:
        return self.nodes[node_id]

    def __len__(self) -> int:
        return len(self.nodes)

    # -- training matrices -------------------------------------------------

    def correctness_dataset(self):
        """(features, labels) for the correctness estimator."""
        feats = [n.features for n in self.nodes.values()]
        labels = [n.rollout_correct for n in self.nodes.values()]
        return np.array(feats, dtype=np.float64), np.array(labels, dtype=np.float64)

    def gain_dataset(self):
        """(features, actions, value_labels) for the gain estimator.

        The value label is mean rollout success under the children minus the
        success of stopping here -- exactly the gross gain the estimator must
        predict.
        """
        feats, actions, values = [], [], []
        for node in self.nodes.values():
            for action in Action.spending():
                cids = node.child_ids(action)
                if not cids:
                    continue
                child_val = float(np.mean([self.nodes[c].rollout_correct for c in cids]))
                feats.append(node.features)
                actions.append(action.value)
                values.append(child_val - node.rollout_correct)
        return (
            np.array(feats, dtype=np.float64),
            np.array(actions),
            np.array(values, dtype=np.float64),
        )

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "roots": self.roots,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path) -> "ComputeTree":
        payload = json.loads(Path(path).read_text())
        tree = cls(meta=payload.get("meta", {}), roots=payload.get("roots", {}))
        for node_id, raw in payload["nodes"].items():
            tree.nodes[node_id] = TreeNode(**raw)
        return tree


# --------------------------------------------------------------------------
# Oracle bound (experiment E0)
# --------------------------------------------------------------------------


def oracle_value(tree: ComputeTree, node_id: str, lam: float, weights: CostWeights):
    """Exact optimal value at a node, by backward induction.

    The ceiling any learned policy can reach given the same action set and tree.
    Computing it *before* training is the point: if the gap to the strongest
    baseline is small, the direction has little headroom and should be rescoped.

    Returns (value, best_action).
    """
    node = tree.get(node_id)
    # STOP's payoff must be discounted by the cost of actually finishing --
    # rollout_correct assumes a completed rollout, which is not free (fix,
    # see TreeNode.finish_cost docstring above).
    best_value = node.rollout_correct - lam * node.finish_cost_vec().scalar(weights)
    best_action = Action.STOP
    if node.is_terminal:
        return best_value, best_action

    for action in Action.spending():
        cids = node.child_ids(action)
        if not cids:
            continue
        exp = float(np.mean([oracle_value(tree, c, lam, weights)[0] for c in cids]))
        value = exp - lam * node.cost_of(action).scalar(weights)
        if value > best_value:
            best_value, best_action = value, action
    return best_value, best_action


def _oracle_cost(tree: ComputeTree, node_id: str, lam: float, weights: CostWeights) -> float:
    node = tree.get(node_id)
    _, action = oracle_value(tree, node_id, lam, weights)
    if action is Action.STOP:
        # Real cost of the chosen action: finishing greedily from here, not
        # zero (fix, see TreeNode.finish_cost docstring above).
        return node.finish_cost_vec().scalar(weights)
    cids = node.child_ids(action)
    if not cids:
        return node.finish_cost_vec().scalar(weights)
    downstream = float(np.mean([_oracle_cost(tree, c, lam, weights) for c in cids]))
    return node.cost_of(action).scalar(weights) + downstream


def oracle_frontier(tree: ComputeTree, lambdas: list, weights: CostWeights) -> list:
    """Sweep lambda to trace the oracle accuracy-cost frontier.

    The same sweep applies to VOC and every baseline, so the curves are
    directly comparable on one plot.

    Reported cost = the root's own first-segment cost (``root_gen_cost``,
    paid regardless of which action is ever chosen) + the cost of whatever
    the optimal policy does from there onward (``_oracle_cost``, which now
    correctly prices STOP as "cost to finish", not zero). This is the fix
    that makes the number comparable to a baseline's fully-inclusive token
    count -- see collect.py's module docstring for the full rationale.
    """
    rows = []
    for lam in lambdas:
        accs, costs = [], []
        for root_id in tree.roots.values():
            root = tree.get(root_id)
            value, _ = oracle_value(tree, root_id, lam, weights)
            action_cost = _oracle_cost(tree, root_id, lam, weights)
            accuracy = value + lam * action_cost  # undo penalty -> accuracy
            total_cost = root.root_gen_cost_vec().scalar(weights) + action_cost
            accs.append(accuracy)
            costs.append(total_cost)
        rows.append(
            {
                "lambda": lam,
                "accuracy": float(np.mean(accs)),
                "cost": float(np.mean(costs)),
                "n_queries": len(accs),
            }
        )
    return rows
