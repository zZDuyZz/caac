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
    params_b: float                 # billions of parameters (for FLOPs)
    price_in_per_mtok: float = 0.0  # USD per 1M input (prefill) tokens
    price_out_per_mtok: float = 0.0  # USD per 1M output (decode) tokens


# EDIT these to your real models/providers before reporting dollar costs.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(params_b=1.54, price_in_per_mtok=0.0, price_out_per_mtok=0.0),
    "Qwen/Qwen2.5-7B-Instruct":  ModelSpec(params_b=7.62, price_in_per_mtok=0.0, price_out_per_mtok=0.0),
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec(params_b=1.24),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(params_b=8.03),
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

        # flops: prefill tokens + decode tokens, each * 2*params
        fpt = _flops_per_token(spec.params_b)
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
