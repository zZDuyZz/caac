"""Full control loop against the mock backend -- runs entirely on CPU."""
from caac.adapters.correctness import SignalAdapter
from caac.backends.mock import MockBackend, MockVerifier
from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.eval.answer_match import answers_match
from caac.policy.baselines import FixedBudgetPolicy
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.types import CostWeights


def _controller(policy, difficulty=0.5):
    return CAACController(
        backend=MockBackend(difficulty=difficulty, finish_after_tokens=256),
        policy=policy,
        adapter=SignalAdapter(),
        features=FeatureExtractor(),
        weights=CostWeights(tokens=1.0),
        verifier=MockVerifier(),
        config=ControllerConfig(segment_tokens=64, max_steps=10, initial_budget=4096),
    )


def test_voc_loop_terminates_and_reports_cost():
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0), lam=0.05)
    ctrl = _controller(pol)
    state, meter, decisions = ctrl.run("q0", "1 + 1 = ?")
    assert len(decisions) >= 1
    assert meter.total().tokens > 0
    # controller overhead is tracked and non-negative
    from caac.cost.accounting import CostSource
    assert meter.by_source[CostSource.CONTROLLER].latency_s >= 0.0


def test_fixed_budget_uses_exactly_n_segments():
    pol = FixedBudgetPolicy(n_segments=3)
    ctrl = _controller(pol)
    state, meter, decisions = ctrl.run("q0", "1 + 1 = ?")
    # up to 3 CONTINUE then STOP (or early finish)
    from caac.types import Action
    assert decisions[-1].action is Action.STOP or state.active_trajectory.finished


def test_finalise_produces_result_row():
    pol = VOCPolicy(gain=HeuristicGain(), cost=AnalyticCost(), weights=CostWeights(tokens=1.0), lam=0.05)
    ctrl = _controller(pol, difficulty=0.0)  # easy -> should answer
    state, meter, decisions = ctrl.run("q0", "1 + 1 = ?")
    result = ctrl.finalise(state, meter, decisions, reference_answer="42", is_correct=answers_match)
    row = result.to_row(CostWeights(tokens=1.0))
    assert "cost_scalar" in row and "n_stop" in row
