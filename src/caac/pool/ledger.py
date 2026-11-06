"""Honest cost ledger over a list of model CALLS (PLAN_V2 section 7).

A policy, when it runs on one question, produces a list of Calls. Each Call is
one model invocation that drew one or more completions sharing a prefill. The
ledger turns those Calls into per-profile scalar costs, enforcing:

  * prefill counted ONCE per call, decode per completion;
  * every token that was paid for is counted, including samples later discarded
    and the small model's rollout before an escalation;
  * five cost profiles (PLAN_V2 section 7): raw tokens, FLOPs (~2*params/token,
    prefill + decode), API price, eFLOPs-style memory-aware, and SEQUENTIAL
    ROUNDS (parallel completions in one call = one round -- the latency axis
    offline replay otherwise hides).

Model sizes/prices live in a small registry you edit for your actual models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Call", "ModelSpec", "MODEL_REGISTRY", "cost_profiles", "PROFILE_NAMES"]


@dataclass
class ModelSpec:
    """Size and price of one served model.

    ``params_b_nonembed`` is what the FLOPs proxy should use: the 2N-per-token
    rule counts matrix multiplies, and an embedding lookup is a gather, not a
    matmul. It matters at this scale -- Qwen2.5-1.5B carries 0.23B of its 1.54B
    parameters in tied embeddings, 15% of the total. (The 7B/1.5B RATIO barely
    moves, 4.94 to 4.99, so earlier numbers computed on totals stand; the
    absolute FLOPs drop by ~15%.) Falls back to ``params_b`` when unknown.

    Prices are per provider and change often, so each entry records WHICH
    provider and WHEN -- a dollar figure without those is not reproducible.
    """

    params_b: float                 # billions of parameters, total
    price_in_per_mtok: float = 0.0  # USD per 1M input (prefill) tokens
    price_out_per_mtok: float = 0.0  # USD per 1M output (decode) tokens
    params_b_nonembed: float | None = None   # used for FLOPs when present
    price_source: str = ""          # provider + date the price was read

    @property
    def flops_params_b(self) -> float:
        return self.params_b_nonembed if self.params_b_nonembed else self.params_b


# Prices read 2026-11-05. Both Qwen entries come from the SAME provider
# (Together) so the small/big ratio is internally consistent; mixing providers
# would make the escalation cost an artefact of who we shopped at.
#
# Worth stating in the paper: Together charges 3.0x more for the 7B than the
# 1.5B, while the FLOPs proxy says 5.0x. The two cost axes disagree by 1.7x on
# the price of an escalation, which is precisely the axis-dependence of §10 --
# and it is not a modelling choice of ours, it is what providers charge.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        params_b=1.54, params_b_nonembed=1.31,
        price_in_per_mtok=0.10, price_out_per_mtok=0.10,
        price_source="Together, read 2026-11-05 (pricepertoken, page dated 2026-07-21)"),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        params_b=7.61, params_b_nonembed=6.53,
        price_in_per_mtok=0.30, price_out_per_mtok=0.30,
        price_source="Together (7B-Instruct-Turbo), read 2026-11-05"),
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec(
        params_b=1.24, params_b_nonembed=0.97,
        price_in_per_mtok=0.027, price_out_per_mtok=0.201,
        price_source="costperprompt aggregate, read 2026-11-05"),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(
        params_b=8.03, params_b_nonembed=6.98,
        price_in_per_mtok=0.02, price_out_per_mtok=0.05,
        price_source="DeepInfra (pricepertoken, page dated 2026-07-03), read 2026-11-05"),
}


@dataclass
class Call:
    """One model invocation during a policy's execution on one question.

    prefill_tokens: prompt length (counted ONCE for this call).
    decode_tokens:  list of generated-token counts, one per completion drawn.
    round_index:    which sequential round this call belongs to. Completions in
                    the same call are parallel (same round). An escalation or a
                    "sample more after greedy" is a NEW round.
    """

    model: str
    prefill_tokens: int
    decode_tokens: list[int]
    round_index: int = 0
    note: str = ""


PROFILE_NAMES = ("tokens", "flops", "api_usd", "eflops", "rounds")


def _flops_per_token(params_b: float) -> float:
    # ~2 * params per token (Kaplan); constant factor cancels in ratios.
    return 2.0 * params_b * 1e9


def cost_profiles(calls: list[Call], mem_coeff: float = 0.3) -> dict[str, float]:
    """Return the scalar cost of a policy execution under each profile.

    mem_coeff scales a crude KV-memory term for the eflops profile (decode is
    memory-bound; this makes small models look less cheap, per Kinetics). It is
    a planning stand-in -- state the exact form you use in the paper.
    """
    out = {p: 0.0 for p in PROFILE_NAMES}
    rounds_seen = set()

    for call in calls:
        spec = MODEL_REGISTRY.get(call.model)
        if spec is None:
            raise KeyError(f"model {call.model!r} not in MODEL_REGISTRY")
        n_comp = len(call.decode_tokens)
        total_decode = sum(call.decode_tokens)

        # tokens: prefill once + all decode
        out["tokens"] += call.prefill_tokens + total_decode

        # flops: prefill tokens + decode tokens, each * 2*params (non-embedding)
        fpt = _flops_per_token(spec.flops_params_b)
        out["flops"] += (call.prefill_tokens + total_decode) * fpt

        # api $: input priced once, output per token
        out["api_usd"] += (
            call.prefill_tokens / 1e6 * spec.price_in_per_mtok
            + total_decode / 1e6 * spec.price_out_per_mtok
        )

        # eflops: flops + memory term growing with decode length per completion
        mem = mem_coeff * fpt * sum(d for d in call.decode_tokens)
        out["eflops"] += (call.prefill_tokens + total_decode) * fpt + mem

        rounds_seen.add(call.round_index)

    # rounds: number of distinct sequential rounds (latency proxy)
    out["rounds"] = float(len(rounds_seen))
    return out
