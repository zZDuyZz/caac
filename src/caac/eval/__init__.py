"""Evaluation: answer matching, calibration, Pareto."""
from caac.eval.answer_match import answers_match, extract_answer, normalize_answer
from caac.eval.calibration import (
    area_under_risk_coverage, brier_score, calibration_report,
    expected_calibration_error, selective_risk_curve,
)
from caac.eval.pareto import (
    OperatingPoint, acc_cost_auc, accuracy_at_cost, compare_frontiers,
    cost_at_accuracy, pareto_frontier,
)

__all__ = [
    "extract_answer", "normalize_answer", "answers_match",
    "expected_calibration_error", "brier_score", "area_under_risk_coverage",
    "selective_risk_curve", "calibration_report",
    "OperatingPoint", "pareto_frontier", "cost_at_accuracy", "accuracy_at_cost",
    "acc_cost_auc", "compare_frontiers",
]
