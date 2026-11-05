"""Inspect which MATH / ARC datasets are cached and what columns they have.

Run this on the GPU box (or any box with the HF cache) BEFORE trusting the
MATH/ARC loaders in benchmarks.py. It prints, for each candidate source, the
splits with row counts and the column names, plus one sample row, so we can wire
load_math(source=..., config=...) to exactly what's available.

Usage:
  HF_HOME=/workspace/hf_cache python inspect_math.py            # offline-friendly
  HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=0 python inspect_math.py   # allow download
"""

from __future__ import annotations

import os


def show(name, cfg=None):
    from datasets import load_dataset
    tag = f"{name}" + (f" [{cfg}]" if cfg else "")
    try:
        ds = load_dataset(name, cfg) if cfg else load_dataset(name)
        for sp, d in ds.items():
            cols = d.column_names
            print(f"  {tag} {sp}: {len(d)} rows  cols={cols}")
            row = d[0]
            for k in cols:
                v = str(row[k])
                print(f"      {k}: {v[:90]}")
            break  # one split's sample is enough
        return True
    except Exception as e:
        print(f"  {tag} ERROR: {str(e)[:160]}")
        return False


def main():
    print("HF_HOME =", os.environ.get("HF_HOME"),
          "| HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    print("\n== FULL MATH candidates ==")
    try:
        from datasets import get_dataset_config_names
        try:
            cfgs = get_dataset_config_names("EleutherAI/hendrycks_math")
            print("  EleutherAI/hendrycks_math configs:", cfgs)
        except Exception as e:
            print("  (configs lookup failed:", str(e)[:100], ")")
    except Exception:
        pass
    show("nlile/hendrycks-MATH-benchmark")
    show("EleutherAI/hendrycks_math", "algebra")
    show("qwedsacf/competition_math")
    print("\n== MATH-500 (reference, already working) ==")
    show("HuggingFaceH4/MATH-500")
    print("\n== ARC-Challenge ==")
    show("allenai/ai2_arc", "ARC-Challenge")


if __name__ == "__main__":
    main()
