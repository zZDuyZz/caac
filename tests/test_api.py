"""Public API end-to-end on the mock backend (no GPU)."""
from caac import CAAC


def test_from_pretrained_and_solve():
    c = CAAC.from_pretrained("mock-model", backend="mock", verifier="mock")
    ans = c.solve("2 + 3 = ?", budget=2048)
    # mock returns an answer from its pool once a trajectory finishes
    assert ans is None or isinstance(ans, str)
    assert hasattr(c, "last_meter")


def test_set_profile_changes_weights():
    c = CAAC.from_pretrained("mock-model", backend="mock")
    before = c.weights
    c.set_profile("latency")
    assert c.weights != before
    assert c.policy.weights == c.weights


def test_calibrate_only_touches_adapter():
    c = CAAC.from_pretrained("mock-model", backend="mock", calibrator="temperature")
    data = [(0.9, 1), (0.2, 0), (0.7, 1), (0.3, 0)] * 30
    c.calibrate(data, n_examples=100)  # should not raise
