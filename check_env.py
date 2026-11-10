"""Is this the environment the existing pools were generated in?

    python check_env.py env_reference.txt           # exit 1 and list every difference
    python check_env.py env_reference.txt --pins    # print name==version for what differs

Pools from two machines are only comparable if both ran the same engine, tokenizer and chat
template code. An unpinned dependency (transformers, for one) can move between two rental
days without any error, so the versions are checked instead of assumed. ``--pins`` leaves
torch out: its build is chosen from the GPU driver and must not be reinstalled from PyPI.
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata


def read_reference(path: str) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                name, ver = line.split("==")
                out[name.strip()] = ver.strip()
    return out


def differences(ref: dict[str, str], installed=None) -> dict[str, tuple[str, str | None]]:
    """name -> (wanted, found) for every package that is missing or at another version."""
    get = installed or _installed
    bad = {}
    for name, want in ref.items():
        have = get(name)
        same = have is not None and (have.split("+")[0] if name == "torch" else have) == want
        if not same:
            bad[name] = (want, have)
    return bad


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--pins", action="store_true")
    a = ap.parse_args()
    bad = differences(read_reference(a.reference))
    if a.pins:
        print(" ".join(f"{n}=={w}" for n, (w, _) in bad.items() if n != "torch"))
        return
    for n, (w, h) in bad.items():
        print(f"ENV DIFFERS: {n} is {h}, the existing pools used {w}")
    if bad:
        sys.exit(1)
    print(f"ENV OK: {len(read_reference(a.reference))} packages match {a.reference}")


if __name__ == "__main__":
    main()
