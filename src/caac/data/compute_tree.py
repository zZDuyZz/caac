"""Compute tree: the offline training substrate.

Records what *would have happened* under every action at every decision point,
for a sample of queries. Expensive to collect once, cheap to reuse -- the
artifact the project is organised around: estimators, the oracle bound, and any
future policy develop against it without re-running a single rollout.

Also makes the E0 oracle bound computable exactly, by backward induction.
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

    def child_ids(self, action: Action) -> list:
        return self.children.get(action.value, [])

    def cost_of(self, action: Action) -> CostVector:
        raw = self.action_cost.get(action.value)
        return CostVector(**raw) if raw else CostVector.zero()


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
    best_value, best_action = node.rollout_correct, Action.STOP
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
        return 0.0
    cids = node.child_ids(action)
    if not cids:
        return 0.0
    downstream = float(np.mean([_oracle_cost(tree, c, lam, weights) for c in cids]))
    return node.cost_of(action).scalar(weights) + downstream


def oracle_frontier(tree: ComputeTree, lambdas: list, weights: CostWeights) -> list:
    """Sweep lambda to trace the oracle accuracy-cost frontier.

    The same sweep applies to VOC and every baseline, so the curves are
    directly comparable on one plot.
    """
    rows = []
    for lam in lambdas:
        accs, costs = [], []
        for root_id in tree.roots.values():
            value, _ = oracle_value(tree, root_id, lam, weights)
            cost = _oracle_cost(tree, root_id, lam, weights)
            accs.append(value + lam * cost)  # undo penalty -> accuracy
            costs.append(cost)
        rows.append(
            {
                "lambda": lam,
                "accuracy": float(np.mean(accs)),
                "cost": float(np.mean(costs)),
                "n_queries": len(accs),
            }
        )
    return rows
