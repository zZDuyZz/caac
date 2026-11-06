"""Tests for the cost registry and the parameter-count checker.

Every FLOPs and dollar figure in the paper flows through MODEL_REGISTRY, so
these lock the properties that would otherwise be wrong in a way no experiment
would reveal: that FLOPs use non-embedding parameters, that each price names
its provider and date, and that the arithmetic in verify_params.py reproduces
the published counts.
"""

from __future__ import annotations

import json

import pytest

from caac.pool.ledger import MODEL_REGISTRY, Call, cost_profiles
from verify_params import count_params

QWEN_1_5B = {           # Qwen/Qwen2.5-1.5B-Instruct config.json
    "model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28,
    "intermediate_size": 8960, "num_attention_heads": 12,
    "num_key_value_heads": 2, "vocab_size": 151936, "tie_word_embeddings": True,
}
QWEN_7B = {             # Qwen/Qwen2.5-7B-Instruct config.json
    "model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28,
    "intermediate_size": 18944, "num_attention_heads": 28,
    "num_key_value_heads": 4, "vocab_size": 152064, "tie_word_embeddings": False,
}


def test_counts_reproduce_the_published_qwen_sizes():
    """Qwen2.5 reports 1.54B total / 1.31B non-embedding for the 1.5B and
    7.61B / 6.53B for the 7B. If this drifts, the FLOPs axis is wrong."""
    a = count_params(QWEN_1_5B)
    assert a["total"] / 1e9 == pytest.approx(1.54, abs=0.02)
    assert a["non_embed"] / 1e9 == pytest.approx(1.31, abs=0.02)
    b = count_params(QWEN_7B)
    assert b["total"] / 1e9 == pytest.approx(7.61, abs=0.03)
    assert b["non_embed"] / 1e9 == pytest.approx(6.53, abs=0.03)


def test_tied_embeddings_are_counted_once():
    tied = count_params(QWEN_1_5B)
    assert tied["lm_head"] == 0
    untied = count_params({**QWEN_1_5B, "tie_word_embeddings": False})
    assert untied["total"] - tied["total"] == tied["embed"]
    assert untied["non_embed"] == tied["non_embed"]


def test_embeddings_are_a_large_share_at_this_scale():
    """The reason FLOPs must not use total parameters: 15% of the 1.5B model
    is embedding lookup, which performs no matmul."""
    c = count_params(QWEN_1_5B)
    assert c["embed"] / c["total"] > 0.13


def test_registry_entries_used_in_experiments_are_complete():
    for name in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
        spec = MODEL_REGISTRY[name]
        assert spec.params_b > 0
        assert spec.params_b_nonembed and spec.params_b_nonembed < spec.params_b
        assert spec.price_in_per_mtok > 0 and spec.price_out_per_mtok > 0
        assert spec.price_source, f"{name} has a price with no provider/date"


def test_flops_use_non_embedding_parameters():
    spec = MODEL_REGISTRY["Qwen/Qwen2.5-1.5B-Instruct"]
    assert spec.flops_params_b == spec.params_b_nonembed
    got = cost_profiles([Call("Qwen/Qwen2.5-1.5B-Instruct", 100, [100])])["flops"]
    assert got == pytest.approx(200 * 2 * spec.params_b_nonembed * 1e9)


def test_flops_fall_back_to_total_when_non_embedding_is_unknown():
    from caac.pool.ledger import ModelSpec
    assert ModelSpec(params_b=3.0).flops_params_b == 3.0


def test_the_two_cost_axes_disagree_about_escalation():
    """Together charges 3.0x more for the 7B; the FLOPs proxy says 5.0x. The
    gap is the axis-dependence the paper reports, so pin it down."""
    s = MODEL_REGISTRY["Qwen/Qwen2.5-1.5B-Instruct"]
    b = MODEL_REGISTRY["Qwen/Qwen2.5-7B-Instruct"]
    flops_ratio = b.flops_params_b / s.flops_params_b
    price_ratio = b.price_out_per_mtok / s.price_out_per_mtok
    assert flops_ratio == pytest.approx(4.99, abs=0.05)
    assert price_ratio == pytest.approx(3.0, abs=0.01)
    assert flops_ratio / price_ratio > 1.5


def test_api_price_is_charged_per_direction():
    spec = MODEL_REGISTRY["Qwen/Qwen2.5-7B-Instruct"]
    usd = cost_profiles([Call("Qwen/Qwen2.5-7B-Instruct", 1_000_000, [1_000_000])])["api_usd"]
    assert usd == pytest.approx(spec.price_in_per_mtok + spec.price_out_per_mtok)


def test_unknown_model_is_refused_rather_than_guessed():
    with pytest.raises(KeyError):
        cost_profiles([Call("some/unregistered-model", 10, [10])])


def test_count_params_handles_an_explicit_head_dim():
    cfg = {**QWEN_7B, "head_dim": 128}
    assert count_params(cfg)["total"] == count_params(QWEN_7B)["total"]


def test_llama_configs_get_no_qkv_bias():
    llama = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16,
             "intermediate_size": 8192, "num_attention_heads": 32,
             "num_key_value_heads": 8, "vocab_size": 128256,
             "tie_word_embeddings": True}
    qwenish = count_params({**llama, "model_type": "qwen2"})
    assert count_params(llama)["total"] < qwenish["total"]
    # Llama-3.2-1B: 1.24B total
    assert count_params(llama)["total"] / 1e9 == pytest.approx(1.24, abs=0.02)
