"""vLLM backend: segment-wise generation with logprob + entropy extraction.

This replaces the mock backend for real runs (phase P1+). The controller talks
to it through exactly one method, ``generate(prompt, prefix, max_tokens)``, and
depends on two things being present on every returned Trajectory:

  * ``token_logprobs``  -- log P(chosen token)      -> token confidence signals
  * ``token_entropies`` -- entropy of the next-token distribution -> uncertainty

Both are produced for free during decoding; the only requirement is to ask vLLM
for the top-k logprobs so the entropy can be estimated from them.

Design points that matter:
  * A single vLLM engine is created once and reused. Re-creating it per call
    would dominate cost and defeat the whole framework.
  * Generation is *segmented*: each call continues ``prefix`` for at most
    ``max_tokens`` tokens. Decision points only exist because generation can be
    interrupted at segment boundaries.
  * ``finished`` is set when the model emits EOS or a stop string, so the
    controller knows a trajectory has completed on its own.
  * A trial answer is extracted from the text so far, matching the
    trial-answer mechanism the controller relies on (belief before the
    trajectory finishes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from caac.eval.answer_match import extract_answer
from caac.types import Trajectory

__all__ = ["VLLMBackend"]


@dataclass
class VLLMBackend:
    """Segment-wise generation via a persistent vLLM engine.

    Parameters
    ----------
    model_name : HF model id, e.g. "Qwen/Qwen2.5-1.5B-Instruct".
    dtype : "bfloat16" | "float16" | "auto".
    top_logprobs : how many top logprobs to request per token. Entropy is
        estimated from this truncated distribution; 20 is a good default
        (cheap, and the tail contributes little to entropy).
    temperature : sampling temperature. 0.0 = greedy (use for the main
        trajectory); a higher value is passed explicitly when BRANCH wants
        diversity.
    gpu_memory_utilization : fraction of VRAM vLLM may use.
    max_model_len : context window cap; None uses the model default.
    stop : optional list of stop strings (in addition to EOS).
    seed : sampling seed for reproducibility.
    """

    model_name: str
    dtype: str = "bfloat16"
    top_logprobs: int = 20
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    stop: list[str] | None = None
    seed: int = 0

    _llm: Any = field(default=None, repr=False, init=False)
    _tok: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        # Import here so `import caac` stays light and CPU-only.
        from vllm import LLM
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.seed,
        )

    # -- main entry point --------------------------------------------------

    def generate(
        self,
        prompt: str,
        prefix: str,
        max_tokens: int,
        *,
        temperature: float | None = None,
    ) -> Trajectory:
        """Continue ``prefix`` for at most ``max_tokens`` tokens.

        The full text fed to the model is (chat-formatted prompt) + prefix, so
        the model sees the reasoning so far and extends it. Only the *new*
        tokens are returned in the Trajectory; the controller concatenates.
        """
        from vllm import SamplingParams

        full_prompt = self._build_prompt(prompt, prefix)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.top_logprobs,
            stop=self.stop,
            seed=self.seed,
        )

        outputs = self._llm.generate([full_prompt], params, use_tqdm=False)
        out = outputs[0].outputs[0]

        logprobs, entropies = self._extract_signals(out)
        text = out.text
        finished = out.finish_reason in ("stop", "length") and self._is_complete(out)

        return Trajectory(
            text=text,
            token_logprobs=logprobs,
            token_entropies=entropies,
            answer=extract_answer(prefix + text),
            finished=finished,
        )

    # -- prompt formatting -------------------------------------------------

    def _build_prompt(self, prompt: str, prefix: str) -> str:
        """Apply the model's chat template, then append the reasoning prefix.

        Using the chat template matters: instruct models behave very
        differently without it. The prefix is appended *after* the template so
        the model continues the reasoning rather than restarting it.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            base = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without a chat template.
            base = prompt + "\n"
        return base + prefix

    # -- signal extraction -------------------------------------------------

    def _extract_signals(self, out: Any) -> tuple[list[float], list[float]]:
        """Pull per-token logprob (of the chosen token) and predictive entropy.

        vLLM returns, per generated position, a dict {token_id: Logprob} for
        the top-k tokens. The chosen token's logprob is the confidence signal;
        the entropy of the (renormalised) top-k distribution approximates the
        predictive entropy.
        """
        chosen_logprobs: list[float] = []
        entropies: list[float] = []

        token_ids = out.token_ids
        per_pos = out.logprobs or []

        for i, lp_dict in enumerate(per_pos):
            if not lp_dict:
                chosen_logprobs.append(math.log(1e-6))
                entropies.append(0.0)
                continue

            # Logprob of the token actually chosen at this position.
            chosen_id = token_ids[i] if i < len(token_ids) else None
            chosen = lp_dict.get(chosen_id)
            chosen_lp = float(chosen.logprob) if chosen is not None else math.log(1e-6)
            chosen_logprobs.append(chosen_lp)

            # Entropy over the truncated top-k distribution.
            lps = [float(v.logprob) for v in lp_dict.values()]
            entropies.append(self._entropy_from_logprobs(lps))

        return chosen_logprobs, entropies

    @staticmethod
    def _entropy_from_logprobs(logprobs: list[float]) -> float:
        """H = -sum p log p over the renormalised top-k probabilities."""
        if not logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in logprobs]
        z = sum(probs) or 1.0
        probs = [p / z for p in probs]
        return float(-sum(p * math.log(p + 1e-12) for p in probs))

    def _is_complete(self, out: Any) -> bool:
        """Heuristic: a stop-reason of 'stop' (EOS/stop-string) means the model
        chose to end. 'length' means it hit max_tokens mid-thought, which is a
        segment boundary, not completion."""
        return out.finish_reason == "stop"
