"""VOC rule: when/where logic, feasibility, lambda behaviour."""
from caac.core.voc import VOCPolicy
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.types import Action, CostWeights, ReasoningState, Trajectory


def _state(belief=0.5, budget=1e9, branches=1):
    s = ReasoningState(query_id="q", prompt="p", budget_remaining=budget, n_branches=branches)
    s.trajectories.append(Trajectory(text="x", token_logprobs=[-0.1], token_entropies=[0.1]))
    s.belief = belief
    return s


def test_stop_when_no_action_clears_cost():
    # Huge lambda makes every action's cost dominate its gain -> STOP.
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0), lam=1e6)
    d = pol.decide(_state(belief=0.5))
    assert d.action is Action.STOP


def test_spends_when_gain_beats_cost():
    # Tiny lambda -> some spending action wins.
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0), lam=1e-6)
    d = pol.decide(_state(belief=0.3))
    assert d.action in Action.spending()


def test_branch_blocked_at_cap():
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0),
                    lam=1e-6, max_branches=2)
    d = pol.decide(_state(belief=0.1, branches=2))
    assert d.voc[Action.BRANCH] == float("-inf")


def test_insufficient_budget_blocks_action():
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0), lam=1e-6)
    d = pol.decide(_state(belief=0.3, budget=1.0))  # cannot afford a 64-token segment
    assert d.action is Action.STOP


def test_verify_gain_is_inverted_u():
    """H2 shape check: VERIFY gain peaks in the middle, low at extremes."""
    g = HeuristicGain()
    lo = g.predict(_state(belief=0.02), Action.VERIFY)
    mid = g.predict(_state(belief=0.6), Action.VERIFY)
    hi = g.predict(_state(belief=0.99), Action.VERIFY)
    assert mid > lo and mid > hi
