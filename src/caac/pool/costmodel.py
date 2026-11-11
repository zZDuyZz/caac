"""What a controller may know about the price of a step BEFORE it pays for it.

Deciding which step to try first, and whether a step is worth its price, needs
the step's cost. The pool records what every completion actually cost, but a
deployed controller has not generated the completion yet: reading the recorded
length of a sample it has not drawn, or of the big model's answer it has not
bought, would let it pick the cheaper repair by looking ahead. Expected cost
uses only what has been paid for already (the greedy turn, the samples drawn so
far) and one constant fitted on TRAIN: how long the big model writes relative to
the small one. Billing still uses the recorded lengths; only decisions use this.
"""

from __future__ import annotations

from dataclasses import dataclass

from caac.pool.ledger import Call
from caac.pool.schema import PoolRecord

__all__ = ["CostModel"]


@dataclass
class CostModel:
    big_ratio: float = 1.0      # big-model decode length / small-model decode length

    def fit(self, small: dict, big: dict | None) -> "CostModel":
        ratios = []
        for q, s in small.items():
            b = (big or {}).get(q)
            if b is not None and s.greedy.decode_tokens > 0 and b.greedy.finish_reason != "length" \
                    and s.greedy.finish_reason != "length":
                ratios.append(b.greedy.decode_tokens / s.greedy.decode_tokens)
        if ratios:
            ratios.sort()
            self.big_ratio = min(2.0, max(0.5, ratios[len(ratios) // 2]))
        return self

    @staticmethod
    def per_sample(small: PoolRecord, drawn: list[int]) -> float:
        """Expected length of the next sample: what the question has shown so far."""
        seen = [small.samples[i].decode_tokens for i in drawn if i < small.n_samples]
        seen.append(small.greedy.decode_tokens)
        return sum(seen) / len(seen)

    def sample_call(self, small: PoolRecord, drawn: list[int], n_new: int, rnd: int = 1) -> Call:
        n = max(0, int(n_new))
        return Call(small.model, small.prompt_tokens, [round(self.per_sample(small, drawn))] * n, rnd)

    def escalate_call(self, small: PoolRecord, big: PoolRecord, drawn: list[int], rnd: int = 1) -> Call:
        return Call(big.model, big.prompt_tokens,
                    [round(self.big_ratio * self.per_sample(small, drawn))], rnd)

    def to_dict(self) -> dict:
        return {"big_ratio": self.big_ratio}
