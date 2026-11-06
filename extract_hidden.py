"""A4: question-only hidden states for the stage-0 probe (CAAC_MASTER §8.1).

One HF forward pass over the SAME chat prompt build_pool uses (prefill only, no
generation, thinking OFF like the pools). Hidden states come from transformers,
not vLLM, so the probe does not depend on the serving engine's version.
Multimodal checkpoints (Qwen3.5, Gemma 4) are loaded through the image-text
class when the causal-LM class refuses them; with text-only input their
``hidden_states`` are the language model's.
For every question we keep, at a few relative depths, (a) the hidden state of
the LAST prompt token and (b) the attention-masked MEAN over prompt tokens.

Output: one ``.npz`` per split
  qids   : (N,) str           -- row order
  last   : (N, L, H) float16  -- last-token state at each kept layer
  mean   : (N, L, H) float16  -- mean-pooled state at each kept layer
  layers : (L,) int           -- hidden_states indices kept (0 = embeddings)
plus ``<out>.manifest.json`` (model, dtype, prompt suffix, layers, n).

Cost per CAAC_MASTER §9: probe = one forward pass of the question (prefill).
Default model = the SMALL model (the router runs before any generation).

Usage (GPU box):
  python extract_hidden.py --model Qwen/Qwen3.5-2B --dataset math500 \
      --out /workspace/hidden/math500_qs.npz
  python extract_hidden.py ... --dataset math_train --qids-file qids/a3_math_train.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_REL_LAYERS = (0.25, 0.5, 0.75, 1.0)


def pick_layers(n_hidden_states: int, rel=DEFAULT_REL_LAYERS) -> list[int]:
    """hidden_states has n_layers+1 entries (0 = embeddings). Map relative
    depths to indices in [1, n_layers]; dedupe, keep order."""
    n_layers = n_hidden_states - 1
    out = []
    for r in rel:
        i = max(1, min(n_layers, round(r * n_layers)))
        if i not in out:
            out.append(i)
    return out


def pool_states(hidden_states, attention_mask, layers):
    """Given HF ``hidden_states`` (tuple of (B,T,H) tensors) and a LEFT-padded
    attention mask (B,T), return (last, mean) as float32 numpy (B,L,H).
    Left padding => the last real token is at position -1 for every row."""
    import torch
    m = attention_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # (B,T,1)
    last, mean = [], []
    for li in layers:
        h = hidden_states[li]
        last.append(h[:, -1, :].float())
        mean.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float())
    return (torch.stack(last, 1).cpu().numpy(), torch.stack(mean, 1).cpu().numpy())


def load_text_model(name: str, dtype, auto_classes=None):
    """Load a checkpoint for a text-only forward pass.

    Tries the causal-LM auto class first, then the image-text-to-text class
    (multimodal checkpoints whose config is not registered as a causal LM).
    ``dtype`` is passed as ``dtype=`` (current transformers) and falls back to
    the older ``torch_dtype=``. ``auto_classes`` is injectable for tests.
    Returns (model, class_name_used).
    """
    if auto_classes is None:
        import transformers
        auto_classes = [getattr(transformers, n) for n in
                        ("AutoModelForCausalLM", "AutoModelForImageTextToText")
                        if hasattr(transformers, n)]
    errors = []
    for cls in auto_classes:
        for kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
            try:
                return cls.from_pretrained(name, **kw), getattr(cls, "__name__", str(cls))
            except TypeError as e:            # old transformers: no dtype= kwarg
                errors.append(f"{cls}: {e}")
                continue
            except (ValueError, KeyError) as e:  # config not mapped to this class
                errors.append(f"{cls}: {e}")
                break
    raise RuntimeError("no auto class could load " + name + ": " + " | ".join(errors))


def run(model, tok, prompts, batch_size, layers=None, device="cpu", log=print):
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    L_all, M_all = [], []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False)  # chat template already has specials
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        if layers is None:
            layers = pick_layers(len(out.hidden_states))
        last, mean = pool_states(out.hidden_states, enc["attention_mask"], layers)
        L_all.append(last.astype(np.float16)); M_all.append(mean.astype(np.float16))
        log(f"hidden {min(s + batch_size, len(prompts))}/{len(prompts)}")
    return np.concatenate(L_all), np.concatenate(M_all), layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qids-file", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from caac.data.benchmarks import load_benchmark
    from caac.data.prompts import build_chat_prompt, SUFFIX

    items = load_benchmark(a.dataset, split=a.split)
    if a.qids_file:
        keep = [l.strip() for l in open(a.qids_file) if l.strip()]
        by_id = {it["id"]: it for it in items}
        missing = [q for q in keep if q not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} qids not in dataset (first: {missing[:3]})")
        items = [by_id[q] for q in keep]
    tok = AutoTokenizer.from_pretrained(a.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cls = load_text_model(a.model, getattr(torch, a.dtype))
    model = model.to(device).eval()
    prompts = [build_chat_prompt(tok, it["question"], thinking=False) for it in items]
    last, mean, layers = run(model, tok, prompts, a.batch_size, device=device,
                             log=lambda m: print(m, flush=True))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qids=np.array([it["id"] for it in items]),
                        last=last, mean=mean, layers=np.array(layers))
    Path(str(out) + ".manifest.json").write_text(json.dumps({
        "model": a.model, "dataset": a.dataset, "split": a.split, "qids_file": a.qids_file,
        "n": len(items), "layers": layers, "dtype_model": a.dtype, "dtype_saved": "float16",
        "prompt_suffix": SUFFIX, "pooling": ["last_token", "masked_mean"],
        "chat_template_kwargs": {"enable_thinking": False}, "model_class": model_cls,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__}, indent=2))
    print(f"done -> {out}  shape={last.shape}", flush=True)


if __name__ == "__main__":
    main()
