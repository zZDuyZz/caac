"""Oracle bound via backward induction on a synthetic compute tree."""
from caac.data.compute_tree import ComputeTree, TreeNode, oracle_value
from caac.types import Action, CostWeights


def _tiny_tree():
    """Root can STOP (0.5) or CONTINUE to a child that is correct (1.0)."""
    tree = ComputeTree()
    child = TreeNode(node_id="c", query_id="q", depth=1, features=[0.0],
                     rollout_correct=1.0, n_rollouts=4, is_terminal=True)
    root = TreeNode(
        node_id="r", query_id="q", depth=0, features=[0.0],
        rollout_correct=0.5, n_rollouts=4,
        children={Action.CONTINUE.value: ["c"]},
        action_cost={Action.CONTINUE.value: {"tokens": 1.0}},
    )
    tree.add(child)
    tree.add(root, is_root=True)
    return tree


def test_oracle_prefers_continue_when_cheap():
    tree = _tiny_tree()
    w = CostWeights(tokens=1.0)
    value, action = oracle_value(tree, "r", lam=0.01, weights=w)
    assert action is Action.CONTINUE
    assert value > 0.5  # better than stopping


def test_oracle_prefers_stop_when_expensive():
    tree = _tiny_tree()
    w = CostWeights(tokens=1.0)
    value, action = oracle_value(tree, "r", lam=100.0, weights=w)
    assert action is Action.STOP
    assert value == 0.5
