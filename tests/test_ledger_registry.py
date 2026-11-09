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


def test_a_model_without_a_price_makes_api_cost_nan_not_free():
    import math
    got = cost_profiles([Call("Qwen/Qwen3.5-2B", 100, [100])])
    assert math.isnan(got["api_usd"]) and got["flops"] > 0
    mixed = cost_profiles([Call("Qwen/Qwen2.5-1.5B-Instruct", 100, [100]),
                           Call("Qwen/Qwen3.5-2B", 100, [100], round_index=1)])
    assert math.isnan(mixed["api_usd"])


def test_refreshed_models_are_flagged_unverified_until_recounted():
    """Entries that come from model cards stay flagged until verify_params.py has
    run on the weights (this is what stops their FLOPs from being reported)."""
    for name in ("google/gemma-4-E2B-it", "google/gemma-4-12B-it"):
        assert MODEL_REGISTRY[name].params_verified is False
    for name in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                 "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B"):
        assert MODEL_REGISTRY[name].params_verified is True


def test_qwen35_counts_match_the_safetensors_recount_of_2026_11_09():
    s, b = MODEL_REGISTRY["Qwen/Qwen3.5-2B"], MODEL_REGISTRY["Qwen/Qwen3.5-9B"]
    assert (s.params_b, s.params_b_nonembed) == (1.882, 1.373)
    assert (b.params_b, b.params_b_nonembed) == (8.954, 6.920)


def _fake_safetensors(path, tensors):
    """Header-only safetensors file (the counter never reads tensor data)."""
    import struct
    hdr, off = {}, 0
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        hdr[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + 2 * n]}
        off += 2 * n
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_safetensors_count_classifies_towers_embeddings_and_mtp(tmp_path):
    from verify_params import count_safetensors
    f = tmp_path / "model.safetensors"
    _fake_safetensors(f, {
        "model.language_model.embed_tokens.weight": (100, 8),        # embed 800
        "model.language_model.embed_tokens_per_layer.weight": (100, 4),  # Gemma PLE 400
        "model.language_model.layers.0.linear_attn.in_proj.weight": (8, 8),  # text 64
        "model.language_model.layers.0.mlp.up_proj.weight": (16, 8),   # text 128
        "model.visual.blocks.0.attn.qkv.weight": (8, 24),             # tower 192
        "model.audio_tower.conv.weight": (2, 5),                      # tower 10
        "mtp.layers.0.mlp.up_proj.weight": (4, 4),                    # mtp 16
        "lm_head.weight": (100, 8),                                   # head 800
    })
    c = count_safetensors([f])
    assert c["embed"] == 1200 and c["text"] == 192 and c["lm_head"] == 800
    assert c["tower"] == 202 and c["mtp"] == 16
    assert c["non_embed"] == 192 and c["total"] == 192 + 1200 + 800


def test_safetensors_count_sums_over_shards(tmp_path):
    from verify_params import count_safetensors
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _fake_safetensors(a, {"model.layers.0.q_proj.weight": (4, 4)})
    _fake_safetensors(b, {"model.layers.1.q_proj.weight": (4, 4)})
    assert count_safetensors([a, b])["non_embed"] == 32
