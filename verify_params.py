"""Check MODEL_REGISTRY parameter counts against the models' real config.json.

The FLOPs axis of every cost number in the paper is ``2 * params * tokens``, so
a wrong parameter count silently rescales the headline. The registry currently
holds counts taken from model cards; this recomputes them from the architecture
and says whether they agree.

    python verify_params.py --config /path/to/Qwen2.5-1.5B-Instruct/config.json
    python verify_params.py --hf Qwen/Qwen2.5-1.5B-Instruct     # needs the hub

Run it on the GPU box in Session 2/3, where the weights are downloaded anyway,
and paste the output into CAAC_MASTER §9. Exit code is non-zero if any model
disagrees with the registry by more than the tolerance, so it can gate a run.

Counts a Llama/Qwen-style decoder exactly: embeddings, per-layer attention
(with Qwen2's q/k/v biases) and gated MLP, the RMSNorms, and an untied LM head
when ``tie_word_embeddings`` is false. "Non-embedding" excludes the token
embedding matrix and the LM head, which is the convention the 2N rule assumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caac.pool.ledger import MODEL_REGISTRY


def count_params(cfg: dict) -> dict:
    """Exact parameter count from a HF decoder config."""
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", heads)
    vocab = cfg["vocab_size"]
    head_dim = cfg.get("head_dim", h // heads)
    tied = cfg.get("tie_word_embeddings", False)
    # Qwen2/Qwen2.5 use biases on q/k/v only; Llama uses none
    qkv_bias = 1 if str(cfg.get("model_type", "")).startswith("qwen2") else 0

    embed = vocab * h
    q = h * heads * head_dim + qkv_bias * heads * head_dim
    k = h * kv * head_dim + qkv_bias * kv * head_dim
    v = k
    o = heads * head_dim * h
    mlp = 3 * h * inter                      # gate, up, down
    norms = 2 * h                            # input + post-attention RMSNorm
    per_layer = q + k + v + o + mlp + norms

    non_embed = layers * per_layer + h       # + final norm
    head = 0 if tied else vocab * h
    total = non_embed + embed + head
    return {"total": total, "non_embed": non_embed, "embed": embed,
            "lm_head": head, "tied": tied, "layers": layers, "hidden": h,
            "per_layer": per_layer}


def load_cfg(path_or_id: str, is_hf: bool) -> tuple[str, dict]:
    if is_hf:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(path_or_id, "config.json")
        return path_or_id, json.loads(Path(p).read_text())
    cfg = json.loads(Path(path_or_id).read_text())
    name = cfg.get("_name_or_path") or Path(path_or_id).parent.name
    return name, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[], help="path to config.json")
    ap.add_argument("--hf", action="append", default=[], help="hub model id")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance")
    a = ap.parse_args()
    if not a.config and not a.hf:
        ap.error("give at least one --config or --hf")

    print(f"\n=== PARAMETER COUNTS vs MODEL_REGISTRY (tolerance {a.tol:.0%}) ===")
    print(f"  {'model':36} {'total B':>9} {'non-emb B':>10} {'registry':>9} "
          f"{'reg non-emb':>12} {'verdict':>9}")
    bad = 0
    for src, is_hf in [(c, False) for c in a.config] + [(m, True) for m in a.hf]:
        try:
            name, cfg = load_cfg(src, is_hf)
        except Exception as e:                      # noqa: BLE001
            print(f"  {src[:36]:36} could not read config: {e}")
            bad += 1
            continue
        c = count_params(cfg)
        tot_b, ne_b = c["total"] / 1e9, c["non_embed"] / 1e9
        spec = MODEL_REGISTRY.get(name)
        if spec is None:
            print(f"  {name[:36]:36} {tot_b:9.3f} {ne_b:10.3f} "
                  f"{'NOT IN REGISTRY':>22}")
            bad += 1
            continue
        reg_ne = spec.params_b_nonembed or spec.params_b
        ok = (abs(tot_b - spec.params_b) / spec.params_b <= a.tol
              and abs(ne_b - reg_ne) / reg_ne <= a.tol)
        bad += 0 if ok else 1
        print(f"  {name[:36]:36} {tot_b:9.3f} {ne_b:10.3f} {spec.params_b:9.3f} "
              f"{reg_ne:12.3f} {'OK' if ok else 'MISMATCH':>9}")
        print(f"  {'':36} {c['layers']} layers x {c['per_layer'] / 1e6:.1f}M, "
              f"embed {c['embed'] / 1e9:.3f}B, "
              f"{'tied' if c['tied'] else 'untied'} head"
              f"   price: {spec.price_source or 'NONE RECORDED'}")
    if bad:
        print(f"\n  {bad} model(s) need attention -- fix MODEL_REGISTRY before "
              f"reporting any FLOPs or dollar number.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
