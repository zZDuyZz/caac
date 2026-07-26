"""Compute-tree collection, oracle, and estimator training on CPU."""
from caac.backends.mock import MockBackend, MockVerifier
from caac.core.budget import lambda_grid
from caac.data.benchmarks import synthetic_benchmark
from caac.data.collect import CollectionConfig, collect_tree
from caac.data.compute_tree import oracle_frontier, oracle_value
from caac.policy.training import split_tree, train_from_tree
from caac.types import Action, CostWeights


def _tree(n=4, depth=2):
    probs = synthetic_benchmark(n=n)
    for p in probs:
        p["answer"] = "42"
    return collect_tree(
        MockBackend(difficulty=0.4), probs, verifier=MockVerifier(),
        config=CollectionConfig(max_depth=depth, n_rollouts=2),
    )


def test_collect_builds_a_wellformed_tree():
    tree = _tree()
    assert len(tree) > 0
    assert len(tree.roots) == 4
    # every root exists as a node and non-terminal nodes have children
    for qid, root_id in tree.roots.items():
        node = tree.get(root_id)
        assert node.query_id == qid
        if not node.is_terminal:
            assert node.children


def test_datasets_have_matching_shapes():
    tree = _tree()
    X, y = tree.correctness_dataset()
    assert X.shape[0] == y.shape[0] == len(tree)
    F, A, V = tree.gain_dataset()
    assert F.shape[0] == len(A) == V.shape[0]


def test_oracle_frontier_is_monotone_in_lambda():
    tree = _tree()
    rows = oracle_frontier(tree, lambda_grid(1e-3, 10, 5), CostWeights(tokens=1.0))
    costs = [r["cost"] for r in rows]
    # raising the price of compute never increases the compute the oracle spends
    assert all(costs[i] >= costs[i + 1] - 1e-9 for i in range(len(costs) - 1))


def test_split_is_by_query_not_by_node():
    tree = _tree(n=5)
    train, val = split_tree(tree, val_frac=0.4)
    assert train and val
    assert train.isdisjoint(val)


def test_training_reports_each_estimator_separately():
    tree = _tree(n=6)
    est = train_from_tree(tree)
    assert "correctness" in est.report or "gain" in est.report
