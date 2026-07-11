"""Uncertainty signals and the tiered feature vector.

Feature tiering is a HARD constraint that protects the multi-model claim.
Features are classified by invariance across models:

  * tier-0 (universal):      belief, step fraction, budget fraction, n_branches
  * tier-1 (self-normalising): entropy slope, answer stability
  * tier-2 (model-specific): raw token confidence, raw entropy scale

The policy layer may read only tier-0 and tier-1; tier-2 goes to the adapter.
``policy_features`` enforces this: it raises if asked for tier-2, so a
model-specific feature can never silently leak into the shared policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caac.types import ReasoningState

__all__ = [
    "mean_token_confidence",
    "min_token_confidence",
    "sequence_entropy",
    "tail_entropy_slope",
    "answer_stability",
    "FeatureExtractor",
    "TIER0",
    "TIER1",
    "TIER2",
    "POLICY_FEATURES",
]


# --------------------------------------------------------------------------
# Individual signals
# --------------------------------------------------------------------------


def mean_token_confidence(logprobs, window=None) -> float:
    """Average token probability over the (optionally windowed) trace."""
    if not logprobs:
        return 0.5
    arr = np.exp(np.asarray(logprobs[-window:] if window else logprobs))
    return float(np.clip(arr.mean(), 0.0, 1.0))


def min_token_confidence(logprobs, window=None) -> float:
    """Weakest link -- a single very low-confidence token often marks where a
    trajectory went wrong, information the mean washes out."""
    if not logprobs:
        return 0.5
    arr = np.exp(np.asarray(logprobs[-window:] if window else logprobs))
    return float(np.clip(arr.min(), 0.0, 1.0))


def sequence_entropy(entropies, window=None) -> float:
    """Mean per-token predictive entropy, normalised to roughly [0, 1]."""
    if not entropies:
        return 0.5
    arr = np.asarray(entropies[-window:] if window else entropies)
    return float(np.clip(arr.mean() / 5.0, 0.0, 1.0))


def tail_entropy_slope(entropies, window=64) -> float:
    """Whether uncertainty is rising or falling over the recent tail.

    Sign carries the signal: falling suggests convergence (favouring STOP),
    rising suggests the trajectory is coming apart (favouring BRANCH).
    """
    if len(entropies) < 4:
        return 0.0
    arr = np.asarray(entropies[-window:])
    x = np.arange(arr.size, dtype=np.float64)
    slope = float(np.polyfit(x, arr, 1)[0])
    return float(np.clip(slope * 100.0, -1.0, 1.0))


def answer_stability(history) -> float:
    """Fraction of recent segments agreeing on the same trial answer."""
    recent = [a for a in history[-5:] if a is not None]
    if len(recent) < 2:
        return 0.0
    top = max(set(recent), key=recent.count)
    return recent.count(top) / len(recent)


# --------------------------------------------------------------------------
# Tiering
# --------------------------------------------------------------------------

TIER0 = ("belief", "step_fraction", "budget_fraction", "n_branches", "n_verify")
TIER1 = ("tail_entropy_slope", "answer_stability")
TIER2 = ("mean_token_confidence", "min_token_confidence", "sequence_entropy")

# What the shared policy is allowed to read.
POLICY_FEATURES = TIER0 + TIER1

ALL_FEATURES = TIER0 + TIER1 + TIER2


@dataclass
class FeatureExtractor:
    """Builds the feature dict from a reasoning state.

    Everything here is O(window) arithmetic over statistics that already exist,
    so extraction adds no forward pass -- what keeps the controller's share of
    total cost near the design target.
    """

    window: int = 64
    max_steps: int = 32
    _answers: list = field(default_factory=list)

    def extract(self, state: ReasoningState):
        traj = state.active_trajectory
        self._answers.append(traj.answer)

        feats = {
            # tier-2 (model-specific)
            "mean_token_confidence": mean_token_confidence(traj.token_logprobs, self.window),
            "min_token_confidence": min_token_confidence(traj.token_logprobs, self.window),
            "sequence_entropy": sequence_entropy(traj.token_entropies, self.window),
            # tier-1 (self-normalising)
            "tail_entropy_slope": tail_entropy_slope(traj.token_entropies, self.window),
            "answer_stability": answer_stability(self._answers),
            # tier-0 (universal)
            "belief": state.belief,
            "step_fraction": state.step / max(1, self.max_steps),
            "budget_fraction": self._budget_fraction(state),
            "n_branches": state.n_branches / 4.0,
            "n_verify": state.n_verify_used / 4.0,
            "verifier_score": float(state.meta.get("verifier_score", 0.5)),
        }

        merged = dict(state.meta)
        merged.update(feats)
        cps = merged.get("belief_checkpoints", [])
        cps = cps + [{"step": state.step, "belief": state.belief, "prefix": traj.text}]
        merged["belief_checkpoints"] = cps

        vec = np.array([feats[k] for k in ALL_FEATURES], dtype=np.float64)
        return vec, merged

    def policy_features(self, feats: dict) -> np.ndarray:
        """Return ONLY the tier-0/1 features the shared policy may use.

        Raises if a tier-2 key is requested -- the hard constraint that keeps
        the policy transferable across models.
        """
        vec = []
        for k in POLICY_FEATURES:
            if k in TIER2:
                raise ValueError(f"tier-2 feature '{k}' cannot enter the policy layer")
            vec.append(feats[k])
        return np.array(vec, dtype=np.float64)

    def _budget_fraction(self, state: ReasoningState) -> float:
        if not np.isfinite(state.budget_remaining) or state.initial_budget <= 0:
            return 1.0
        return float(np.clip(state.budget_remaining / state.initial_budget, 0.0, 1.0))

    def reset(self) -> None:
        self._answers.clear()
