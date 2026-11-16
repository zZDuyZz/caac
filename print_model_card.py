"""Print the vendor's own sampling advice for a model from the local Hugging Face cache.

    python print_model_card.py Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B

The decoding protocol of a pool must be the one the model card recommends (or a documented
deviation), and third-party copies of a card can be stale. This reads the README.md and
generation_config.json that `session2_setup.sh` downloaded with the weights, so it shows the
official text of exactly the revision that generates the pools. Works offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def best_practices(readme: str, max_lines: int = 60) -> list[str]:
    """The 'Best Practices' section (up to the next heading of the same or higher level),
    plus any other line that names a sampling parameter."""
    lines = readme.splitlines()
    out, level = [], None
    for ln in lines:
        s = ln.lstrip()
        if s.startswith("#"):
            lv = len(s) - len(s.lstrip("#"))
            if level is not None and lv <= level:
                break
            if level is None and "best practice" in s.lower():
                level = lv
        if level is not None:
            out.append(ln)
    if not out:
        out = [ln for ln in lines if any(k in ln for k in ("presence_penalty", "top_k", "top_p", "temperature"))]
    return out[:max_lines]


def main() -> None:
    from huggingface_hub import snapshot_download
    for repo in sys.argv[1:]:
        print(f"===== {repo}")
        try:
            root = Path(snapshot_download(repo, local_files_only=True))
        except Exception as e:                       # noqa: BLE001
            print(f"  not in the local cache: {e}")
            continue
        print(f"  snapshot: {root}")
        gc = root / "generation_config.json"
        print("  generation_config.json:", json.dumps(json.loads(gc.read_text())) if gc.exists() else "missing")
        rd = root / "README.md"
        if not rd.exists():
            print("  README.md missing")
            continue
        for ln in best_practices(rd.read_text(encoding="utf-8")):
            print("  | " + ln)


if __name__ == "__main__":
    main()
