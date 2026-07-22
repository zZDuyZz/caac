"""Public API -- the facade a user touches.

    controller = CAAC.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    controller.calibrate(dataset, n_examples=500)
    answer = controller.solve("What is 17 * 23?", budget=2048)
    controller.set_profile("latency")

The user never touches VOC math or the cost model. On CPU with the mock backend
the whole loop runs end-to-end; passing backend="vllm" swaps in the GPU engine
(phase P1) without changing any decision-layer code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from caac.adapters.base import Adapter
from caac.adapters.calibration import build_calibrator
from caac.adapters.correctness import SignalAdapter
from caac.core.controller import CAACController, ControllerConfig
from caac.core.voc import VOCPolicy
from caac.cost.accounting import DEPLOYMENT_PROFILES
from caac.eval.answer_match import answers_match, extract_answer
from caac.policy.cost import AnalyticCost
from caac.policy.gain import HeuristicGain
from caac.signals.features import FeatureExtractor
from caac.types import CostWeights

__all__ = ["CAAC"]


def _make_backend(name: str, **kwargs):
    """Lazy backend factory -- defers importing GPU deps until needed."""
    if name == "mock":
        from caac.backends.mock import MockBackend

        return MockBackend(**kwargs)
    if name == "hf":
        from caac.backends.hf import HFBackend

        return HFBackend(**kwargs)
    if name == "vllm":
        from caac.backends.vllm import VLLMBackend

        return VLLMBackend(**kwargs)
    raise KeyError(f"unknown backend '{name}'")


def _make_verifier(name: str, backend=None, **kwargs):
    if name is None:
        return None
    if name == "mock":
        from caac.backends.mock import MockVerifier

        return MockVerifier(**kwargs)
    if name == "self":
        from caac.verifier.self_verify import SelfVerifier

        return SelfVerifier(backend=backend, **kwargs)
    if name == "prm":
        from caac.verifier.prm import PRMVerifier

        return PRMVerifier(**kwargs)
    raise KeyError(f"unknown verifier '{name}'")


@dataclass
class CAAC:
    """Cost-aware adaptive computation controller, packaged for use."""

    backend: Any
    adapter: Adapter
    policy: VOCPolicy
    features: FeatureExtractor
    weights: CostWeights
    verifier: Any = None
    config: ControllerConfig = field(default_factory=ControllerConfig)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        backend: str = "mock",
        verifier: str | None = "mock",
        profile: str = "token_only",
        calibrator: str = "temperature",
        lam: float = 5e-4,
        **backend_kwargs,
    ) -> "CAAC":
        """Build a controller around a model.

        Defaults to the CPU mock backend so the object is usable with no GPU;
        pass backend="vllm" (phase P1) for real serving.

        ``lam`` is the accuracy/compute exchange rate and must be on the scale
        of (expected gain) / (cost of one action). With the token_only profile
        a segment costs ~64, and gains are O(0.1), so a sensible default is
        ~5e-4. Sweeping lam traces the Pareto frontier; see core.budget.
        """
        be = _make_backend(backend, **backend_kwargs)
        vf = _make_verifier(verifier, backend=be)
        weights = DEPLOYMENT_PROFILES.get(profile, CostWeights())
        adapter = SignalAdapter(calibrator=build_calibrator(calibrator))
        # max_verify is a defensive cap: the heuristic gain is a placeholder and
        # can over-value VERIFY. A fitted gain model (phase P4) removes the need.
        policy = VOCPolicy(
            gain=HeuristicGain(),
            cost=AnalyticCost(),
            weights=weights,
            lam=lam,
            max_branches=2,
            max_verify=2,
        )
        return cls(
            backend=be,
            adapter=adapter,
            policy=policy,
            features=FeatureExtractor(),
            weights=weights,
            verifier=vf,
            config=ControllerConfig(),
        )

    # -- calibration (adapter only) ---------------------------------------

    def calibrate(self, dataset, n_examples: int = 500) -> "CAAC":
        """Fit the per-model adapter on a small labelled set.

        Only the adapter is touched -- the policy layer is shared and stays
        fixed. This is what makes moving to a new model cheap.

        ``dataset`` is an iterable of (raw_signal, correct_label) pairs.
        """
        import numpy as np

        pairs = list(dataset)[:n_examples]
        if not pairs:
            return self
        signals = np.array([p[0] for p in pairs], dtype=np.float64)
        labels = np.array([p[1] for p in pairs], dtype=np.float64)
        self.adapter.calibrate(signals, labels)
        return self

    # -- inference ---------------------------------------------------------

    def solve(self, query: str, *, budget: float | None = None, query_id: str = "q0"):
        """Run the control loop and return the final answer string."""
        if budget is not None:
            self.config = ControllerConfig(
                segment_tokens=self.config.segment_tokens,
                max_steps=self.config.max_steps,
                initial_budget=budget,
                aggregate=self.config.aggregate,
            )
        ctrl = CAACController(
            backend=self.backend,
            policy=self.policy,
            adapter=self.adapter,
            features=self.features,
            weights=self.weights,
            verifier=self.verifier,
            config=self.config,
        )
        state, meter, decisions = ctrl.run(query_id, query)
        answer = state.active_trajectory.answer
        self.last_meter = meter
        self.last_decisions = decisions
        return answer

    # -- deployment profile -----------------------------------------------

    def set_profile(self, name: str) -> "CAAC":
        """Re-price the action space by swapping cost weights. No retraining."""
        if name not in DEPLOYMENT_PROFILES:
            raise KeyError(f"unknown profile '{name}'; choose {sorted(DEPLOYMENT_PROFILES)}")
        self.weights = DEPLOYMENT_PROFILES[name]
        self.policy.weights = self.weights
        return self
