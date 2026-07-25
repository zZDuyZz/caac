"""Estimator training from a compute tree (phase P4).

Fits the three quantities the VOC rule needs, each separately so a failing
component can be diagnosed:

    p_theta  -- calibrated correctness estimator   (adapter, per-model)
    Delta_psi -- gain model per action             (policy, shared)
    c_phi    -- cost model                         (policy, shared; analytic by default)

Training is offline supervised regression on the tree, not online RL. The signal
needed is a structured expected value, which fitted value estimation on the tree
provides directly and more stably, and separating the estimators keeps the
diagnosis possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caac.adapters.calibration import build_calibrator
from caac.adapters.correctness import FittedAdapter
from caac.data.compute_tree import ComputeTree
from caac.policy.gain import FittedGain
from caac.utils.logging import get_logger

__all__ = ["TrainedEstimators", "train_from_tree", "split_tree"]

log = get_logger(__name__)


@dataclass
class TrainedEstimators:
    """Container returned by ``train_from_tree``."""

    adapter: FittedAdapter
    gain: FittedGain
    report: dict


def split_tree(tree: ComputeTree, val_frac: float = 0.2, seed: int = 0):
    """Split nodes by *query*, not by node.

    Splitting by node would leak: nodes from the same query share a prefix, so a
    node in train and its child in validation is effectively the same example.
    """
    rng = np.random.default_rng(seed)
    qids = sorted(tree.roots)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_ids = set(qids[:n_val])
    train_ids = set(qids[n_val:])
    return train_ids, val_ids


def _subset(tree: ComputeTree, query_ids: set):
    """Rows of the training matrices restricted to the given queries."""
    feats, labels = [], []
    gfeats, gactions, gvalues = [], [], []
    for node in tree.nodes.values():
        if node.query_id not in query_ids:
            continue
        feats.append(node.features)
        labels.append(node.rollout_correct)
        for action_value, child_ids in node.children.items():
            if not child_ids:
                continue
            child_val = float(
                np.mean([tree.nodes[c].rollout_correct for c in child_ids])
            )
            gfeats.append(node.features)
            gactions.append(action_value)
            gvalues.append(child_val - node.rollout_correct)
    return (
        np.array(feats, dtype=np.float64),
        np.array(labels, dtype=np.float64),
        np.array(gfeats, dtype=np.float64),
        np.array(gactions),
        np.array(gvalues, dtype=np.float64),
    )


def train_from_tree(
    tree: ComputeTree,
    *,
    calibrator: str = "temperature",
    val_frac: float = 0.2,
    seed: int = 0,
    label_threshold: float = 0.5,
) -> TrainedEstimators:
    """Fit the adapter and gain model, and report validation quality.

    ``rollout_correct`` is a rate in [0, 1]; the correctness estimator needs
    binary targets, so it is thresholded. The rate itself is kept for the gain
    labels, where the continuous value is the quantity of interest.
    """
    train_ids, val_ids = split_tree(tree, val_frac, seed)
    Xtr, ytr, Ftr, Atr, Vtr = _subset(tree, train_ids)
    Xva, yva, Fva, Ava, Vva = _subset(tree, val_ids)

    log.info("train nodes=%d  val nodes=%d", len(Xtr), len(Xva))

    adapter = FittedAdapter(calibrator=build_calibrator(calibrator))
    adapter.calibrate(Xtr, (ytr >= label_threshold).astype(float))

    gain = FittedGain()
    if len(Ftr):
        gain.fit(Ftr, Atr, Vtr)

    report = _evaluate(adapter, gain, Xva, yva, Fva, Ava, Vva, label_threshold)
    return TrainedEstimators(adapter=adapter, gain=gain, report=report)


def _evaluate(adapter, gain, Xva, yva, Fva, Ava, Vva, thr) -> dict:
    """Validation quality of each estimator, reported separately.

    Separate numbers are the point: if the policy underperforms, this says
    whether the belief, the gain model, or neither is the bottleneck.
    """
    from caac.eval.calibration import calibration_report

    out: dict = {}

    if len(Xva):
        raw = adapter._model.predict_proba(Xva)[:, 1]
        probs = adapter.calibrator.transform(raw)
        labels = (yva >= thr).astype(float)
        out["correctness"] = calibration_report(probs, labels)

    if len(Fva):
        errs = {}
        for action in set(Ava):
            mask = Ava == action
            if mask.sum() == 0:
                continue
            from caac.types import Action, ReasoningState, Trajectory

            preds = []
            for row in Fva[mask]:
                s = ReasoningState(query_id="v", prompt="")
                s.trajectories.append(Trajectory())
                s.features = row
                preds.append(gain.predict(s, Action(action)))
            preds = np.array(preds)
            errs[str(action)] = {
                "mae": float(np.mean(np.abs(preds - Vva[mask]))),
                "n": int(mask.sum()),
            }
        out["gain"] = errs

    return out
