"""`caac` command line.

  caac validate POOL_DIR                       -> data-quality report, levels (exit 1 on errors)
  caac audit POOL_DIR --config CFG [--out R.json] [--md R.md] [--lock LOCK.json]
  caac lock-hash LOCK.json                     -> sha256 of the canonical lock file
  caac plan --risk R [--coverage C] [--alpha A --delta D] | --from-report R.json
                                               -> questions needed before a GO is possible

POOL_DIR holds rows.jsonl[.gz] or rows.csv and meta.json (CAAC-Pool v1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from caac.pool_v1.config import load_config
from caac.pool_v1.io import load_pool_v1
from caac.pool_v1.lock import check_config_against_lock, load_lock, lock_hash
from caac.pool_v1.report import dumps, make_report, to_markdown
from caac.pool_v1.validate import validate


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _input_info(d: Path) -> dict:
    info = {}
    for n in ("rows.jsonl.gz", "rows.jsonl", "rows.csv", "rows.csv.gz", "meta.json"):
        if (d / n).exists():
            info[n] = _sha(d / n)
    return info


def cmd_validate(a) -> int:
    rows, rerr, meta, merr = load_pool_v1(a.pool)
    rep = validate(rows, meta, row_errors=rerr, meta_errors=merr)
    out = rep.to_dict()
    print(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False))
    return 0 if rep.ok else 1


def cmd_audit(a) -> int:
    cfg = load_config(a.config)
    d = Path(a.pool)
    rows, rerr, meta, merr = load_pool_v1(d)
    info = _input_info(d)
    info["config"] = _sha(Path(a.config))
    if a.lock:
        problems = check_config_against_lock(cfg, a.lock, caps=meta.caps if meta else None)
        if problems:
            print("config does not match the protocol lock:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        info["lock"] = lock_hash(a.lock)
        info["lock_status"] = load_lock(a.lock)["status"]
        if info["lock_status"] != "frozen":
            print("WARNING: the protocol lock is a DRAFT: this run is not pre-registered", file=sys.stderr)
    rep = make_report(rows, meta, cfg, row_errors=rerr, meta_errors=merr, input_info=info)
    text = dumps(rep)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    if a.md:
        Path(a.md).write_text(to_markdown(rep), encoding="utf-8", newline="\n")
    if not a.out:
        sys.stdout.write(text)
    v = rep["R1_verdict"]["verdict"]
    print(f"verdict: {v}", file=sys.stderr)
    return 1 if v == "NOT RUN" else 0


def cmd_lock_hash(a) -> int:
    print(lock_hash(a.lock))
    return 0


def cmd_plan(a) -> int:
    from caac.pool_v1.plan import plan
    alpha, delta, risk, cov = a.alpha, a.delta, a.risk, a.coverage
    if a.from_report:
        rep = json.loads(Path(a.from_report).read_text(encoding="utf-8"))
        v, r2 = rep.get("R1_verdict", {}), rep.get("R2_estimates", {})
        if r2.get("risk") is None or not r2.get("coverage") or "alpha" not in v:
            raise ValueError("the report has no chosen policy with served answers (nothing to plan from)")
        alpha, delta = float(v["alpha"]), float(v["delta"])
        risk, cov = float(r2["risk"]), float(r2["coverage"])
    if risk is None:
        raise ValueError("give --risk (expected selective risk) or --from-report")
    out = plan(alpha, delta, risk, cov)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="caac", description="CAAC: certified verdict for small-model-first")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check a pool and report its level")
    v.add_argument("pool")
    v.set_defaults(fn=cmd_validate)
    au = sub.add_parser("audit", help="run the audit and write the report")
    au.add_argument("pool")
    au.add_argument("--config", required=True)
    au.add_argument("--out")
    au.add_argument("--md")
    au.add_argument("--lock", help="protocol lock; the config must match it")
    au.set_defaults(fn=cmd_audit)
    lh = sub.add_parser("lock-hash", help="hash of a protocol lock file")
    lh.add_argument("lock")
    lh.set_defaults(fn=cmd_lock_hash)
    pl = sub.add_parser("plan", help="how many questions are needed before a GO is possible")
    pl.add_argument("--risk", type=float, help="expected selective risk (e.g. 0.05)")
    pl.add_argument("--coverage", type=float, default=1.0, help="share of questions the policy serves")
    pl.add_argument("--alpha", type=float, default=0.10)
    pl.add_argument("--delta", type=float, default=0.10)
    pl.add_argument("--from-report", help="take alpha, delta, risk, coverage from a report JSON")
    pl.set_defaults(fn=cmd_plan)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
