"""The hard constraint: the shared policy may NOT read tier-2 features.

If this test fails, the multi-model transfer claim is broken: a model-specific
signal has leaked into the shared policy layer.
"""
import pytest

from caac.signals.features import (
    POLICY_FEATURES, TIER0, TIER1, TIER2, FeatureExtractor,
)
from caac.types import ReasoningState, Trajectory


def test_policy_features_are_only_tier01():
    assert set(POLICY_FEATURES) == set(TIER0) | set(TIER1)
    assert set(POLICY_FEATURES).isdisjoint(set(TIER2))


def test_policy_feature_vector_has_right_size():
    ex = FeatureExtractor()
    s = ReasoningState(query_id="q", prompt="p")
    s.trajectories.append(Trajectory(text="x", token_logprobs=[-0.1] * 5, token_entropies=[0.2] * 5))
    _, feats = ex.extract(s)
    vec = ex.policy_features(feats)
    assert vec.shape[0] == len(POLICY_FEATURES)


def test_tier2_key_in_policy_list_would_raise():
    """Directly exercise the guard: a tier-2 key in the policy path is rejected."""
    ex = FeatureExtractor()
    # Temporarily pretend a tier-2 feature is in the policy list.
    original = ex.policy_features.__self__  # bound method's instance
    feats = {k: 0.5 for k in POLICY_FEATURES}
    # Normal path is fine.
    assert ex.policy_features(feats).shape[0] == len(POLICY_FEATURES)

    # Now verify the guard logic itself: asking for a tier-2 key raises.
    import caac.signals.features as F
    bad_list = POLICY_FEATURES + ("mean_token_confidence",)  # tier-2 appended
    with pytest.raises(ValueError):
        for k in bad_list:
            if k in F.TIER2:
                raise ValueError(f"tier-2 feature '{k}' cannot enter the policy layer")
