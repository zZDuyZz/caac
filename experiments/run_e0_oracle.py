"""E0 -- oracle upper bound.

Runs on CPU using a synthetic compute tree so the oracle machinery is
exercisable before any real tree is collected. With a real tree (phase P3),
point --tree at the saved JSON and the same code produces the oracle frontier.

The oracle gap (oracle frontier vs best baseline) is a go/no-go gate: if it is
small, the direction has little headroom.
"""

from __future__ import annotations

import argparse

from caac.core.budget import lambda_grid
from caac.data.compute_tree import ComputeTree, TreeNode, oracle_frontier
from caac.types import Action, CostWeights


def _synthetic_tree(n_queries: int = 20, seed: int = 0) -> ComputeTree:
    """A small tree where CONTINUE helps on some queries, not others."""
    import numpy as np

    rng = np.random.default_rng(seed)
    tree = ComputeTree()
    for i in range(n_queries):
        base = float(rng.uniform(0.3, 0.7))
        gain = float(rng.uniform(0.0, 0.4))
        leaf = TreeNode(
            node_id=f"c{i}", query_id=f"q{i}", depth=1, features=[base],
            rollout_correct=min(1.0, base + gain), n_rollouts=8, is_terminal=True,
        )
        root = TreeNode(
            node_id=f"r{i}", query_id=f"q{i}", depth=0, features=[base],
            rollout_correct=base, n_rollouts=8,
            children={Action.CONTINUE.value: [f"c{i}"]},
            action_cost={Action.CONTINUE.value: {"tokens": 64.0, "forward_passes": 64.0}},
        )
        tree.add(leaf)
        tree.add(root, is_root=True)
    return tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default=None, help="path to a saved ComputeTree JSON")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    tree = ComputeTree.load(args.tree) if args.tree else _synthetic_tree(args.n)
    weights = CostWeights(tokens=1.0)
    frontier = oracle_frontier(tree, lambda_grid(1e-3, 10.0, 10), weights)

    print(f"{'lambda':>10} {'accuracy':>10} {'cost':>10}")
    for row in frontier:
        print(f"{row['lambda']:>10.4f} {row['accuracy']:>10.4f} {row['cost']:>10.2f}")


if __name__ == "__main__":
    main()
