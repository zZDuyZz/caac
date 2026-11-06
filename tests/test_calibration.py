"""Calibration metrics (ECE, Brier, risk-coverage) used for the estimator report."""
import numpy as np

from caac.eval.calibration import (
    area_under_risk_coverage, brier_score, calibration_report,
    expected_calibration_error, selective_risk_curve,
)


def test_perfect_calibration_has_zero_ece():
    # probs equal to the empirical accuracy in each bin.
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, labels, n_bins=10) < 1e-9
    assert brier_score(probs, labels) < 1e-9


def test_selective_risk_decreases_with_confidence():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=500)
    labels = (rng.uniform(size=500) < probs).astype(float)  # well-calibrated-ish
    cov, risk = selective_risk_curve(probs, labels)
    # risk at low coverage (most confident) <= risk at full coverage
    assert risk[0] <= risk[-1] + 0.05
    assert 0.0 <= area_under_risk_coverage(probs, labels) <= 1.0


def test_report_has_all_keys():
    probs = np.array([0.2, 0.8, 0.6, 0.4])
    labels = np.array([0, 1, 1, 0])
    r = calibration_report(probs, labels)
    for k in ("ece", "brier", "aurc", "base_rate"):
        assert k in r
