"""Process reward model verifier (phase P1+).

Scores a partial reasoning trace with an external PRM (e.g. Qwen2.5-Math-PRM).
The forward passes it costs are returned in the result so the controller charges
them into the same budget as reasoning tokens -- the condition for a fair
comparison against methods that lean on an external PRM.

Two backends for the PRM are supported:
  * a vLLM engine (fast, preferred for throughput), or
  * a plain HF model (simpler, fine for the calibration study).
The interface the controller sees is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from caac.verifier.base import VerifierResult

__all__ = ["PRMVerifier"]


@dataclass
class PRMVerifier:
    """External PRM scorer.

    Parameters
    ----------
    model_name : PRM model id, e.g. "Qwen/Qwen2.5-Math-PRM-7B".
    dtype : torch dtype string.
    step_separator : the token/string the PRM uses to mark step boundaries;
        the score is read at the final step. Qwen math PRMs use "\n\n".
    device : "cuda" | "cpu".
    """

    model_name: str
    dtype: str = "bfloat16"
    step_separator: str = "\n\n"
    device: str = "cuda"

    _model: Any = field(default=None, repr=False, init=False)
    _tok: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()

    def verify(self, prompt: str, trace: str) -> VerifierResult:
        """Return P(trace is on a correct path) plus the cost of computing it.

        The PRM emits a reward per reasoning step; the controller only needs a
        single scalar, so the reward at the final step is used. Token count is
        the length of (prompt + trace), which is what the PRM forward pass sees.
        """
        import torch

        text = prompt + "\n" + trace
        inputs = self._tok(text, return_tensors="pt", truncation=True).to(self.device)
        n_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            outputs = self._model(**inputs)

        score = self._read_step_reward(outputs, inputs)

        return VerifierResult(
            score=float(score),
            n_tokens=float(n_tokens),
            n_forward_passes=float(n_tokens),  # one forward over the sequence
        )

    def _read_step_reward(self, outputs: Any, inputs: Any) -> float:
        """Extract the final-step reward as a probability in [0, 1].

        PRM output formats differ. This handles the common Qwen-math case where
        the model exposes per-token reward logits; adapt the indexing to the
        specific PRM if its head differs (documented in the model card).
        """
        import torch

        # Common case: a reward tensor of shape (batch, seq, 2) or (batch, seq).
        logits = getattr(outputs, "logits", None)
        if logits is None:
            # Some PRMs return a raw tensor as the first element.
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        logits = logits.float()
        if logits.dim() == 3 and logits.shape[-1] == 2:
            probs = torch.softmax(logits, dim=-1)[..., 1]  # P(correct) per token
        elif logits.dim() == 3:
            probs = torch.sigmoid(logits[..., 0])
        else:
            probs = torch.sigmoid(logits)

        # Reward at the last non-pad position.
        last = int(inputs["attention_mask"][0].sum().item()) - 1
        last = max(0, min(last, probs.shape[1] - 1))
        return float(probs[0, last].item())
