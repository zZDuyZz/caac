"""Cross-source map (C15) and audit vs the naive point-estimate rule (C14).

Everything that decides WHICH configurations are run is read from the protocol lock and from
the structure of the pools (model names, question ids, token counts, prices); nothing here
looks at correctness before the audit is run. For each configuration (pool, pair, cost measure):
  * the pool is restricted to the two models and to the questions both answered;
  * the candidate policies come from the lock, limited to what the data supports
    (k <= draws every question has; k >= 2 only with answer keys; cut = lock fractions of the
    small model's cap);
  * the audit verdict (make_report) and the NAIVE verdict are recorded. Naive rule (lock
    criteria.b): GO iff some candidate policy has empirical risk <= alpha AND empirical saving
    >= s on ALL the data (no split, no interval, no correction for trying several policies).

  python -m caac.pool_v1.crossmap POOLS_DIR --lock protocols/audit_lock_v1.json --out map/
  -> map/map.jsonl (one line per configuration), map/summary.json (criteria a-d), map/reports/
A run with a lock whose status is not "frozen" is marked NOT PRE-REGISTERED in every output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from caac.pool_v1.config import config_from_dict
from caac.pool_v1.io import load_pool_v1
from caac.pool_v1.lock import allowed_cuts, check_config_against_lock, load_lock, lock_hash, sensitivity_cells
from caac.pool_v1.replay import replay
from caac.pool_v1.report import dumps, make_report


# ------------------------------------------------------------------ structure only
def unit_price(rows, model) -> float | None:
    """Median billed price per token (cost_usd / (in + out tokens)) of a model: a property of
    the price list, read from cost and token columns only."""
    v = [r.cost_usd / ((r.out_tokens or 0) + (r.in_tokens or 0)) for r in rows
         if r.model == model and r.draw == 0 and r.cost_usd is not None
         and ((r.out_tokens or 0) + (r.in_tokens or 0)) > 0]
    return statistics.median(v) if v else None


def max_unit_price(rows, model) -> float | None:
    # rows with out_tokens = 0 (a billed prompt with no output) count too: leaving them out
    # could UNDER-estimate the largest price and break the cost bound
    v = [r.cost_usd / ((r.out_tokens or 0) + (r.in_tokens or 0)) for r in rows
         if r.model == model and r.cost_usd is not None
         and ((r.out_tokens or 0) + (r.in_tokens or 0)) > 0]
    return max(v) if v else None


def common_questions(rows, a, b) -> set:
    qa = {r.qid for r in rows if r.model == a}
    qb = {r.qid for r in rows if r.model == b}
    return qa & qb


def source_of(pool_name: str) -> str:
    return pool_name.split("_", 1)[0]


def benchmark_of(pool_name: str) -> str:
    return pool_name.split("_", 1)[1] if "_" in pool_name else pool_name


def included(pool_name: str, L: dict) -> tuple[bool, str]:
    inc = L["benchmarks"]
    b = benchmark_of(pool_name)
    for pat in inc.get("exclude_from_primary", []):
        if b == pat or b.startswith(pat + "_"):
            return False, f"excluded by lock: {pat}"
    return True, ""


def pairs_for(pool_name, rows, L) -> list[dict]:
    """Primary pairs = the lock's declared same-family pairs present in the pool.
    Secondary pairs = every (cheaper, dearer) pair by unit price (USD) or by nominal
    parameter count (token pools), as the lock's rule says. Each with n common questions."""
    src = source_of(pool_name)
    models = sorted({r.model for r in rows})
    out = []
    declared = [tuple(p) for p in L["pairs"]["primary"].get(src, [])]
    params = L["pairs"].get("nominal_params_b", {})
    price = {m: unit_price(rows, m) for m in models}
    seen = set()
    for a, b in declared:
        if a in models and b in models:
            out.append({"small": a, "big": b, "tier": "primary"})
            seen.add((a, b))
    for a in models:
        for b in models:
            if a == b or (a, b) in seen:
                continue
            if a in params and b in params:
                if params[a] < params[b]:
                    out.append({"small": a, "big": b, "tier": "secondary"})
            elif price.get(a) is not None and price.get(b) is not None and price[a] < price[b]:
                out.append({"small": a, "big": b, "tier": "secondary"})
    n_min = L["pairs"]["n_min"]
    keep = []
    for p in out:
        n = len(common_questions(rows, p["small"], p["big"]))
        p["n_common"] = n
        if n >= n_min:
            keep.append(p)
    return keep


def measures_for(pool_name, rows, meta, pair, L) -> list[dict]:
    """Cost measures of a configuration, from the lock's cost rules."""
    src = source_of(pool_name)
    rule = L["cost"]["measures"][src]
    out = []
    for m in rule:
        if m == "usd":
            if not all(r.cost_usd is not None for r in rows if r.model in (pair["small"], pair["big"])):
                continue
            local = set(L["cost"].get("locally_run_models", {}).get(src, []))
            if pair["small"] in local or pair["big"] in local:
                continue        # lock v2: a price the source estimated by an unstated method
            ps, pb = max_unit_price(rows, pair["small"]), max_unit_price(rows, pair["big"])
            caps = meta.caps or {}
            if ps is None or pb is None or pair["small"] not in caps or pair["big"] not in caps:
                continue
            in_s = max((r.in_tokens or 0) for r in rows if r.model == pair["small"])
            in_b = max((r.in_tokens or 0) for r in rows if r.model == pair["big"])
            cap_s = max(caps[pair["small"]], max((r.out_tokens or 0) for r in rows if r.model == pair["small"]))
            cap_b = max(caps[pair["big"]], max((r.out_tokens or 0) for r in rows if r.model == pair["big"]))
            c_max = (cap_s + in_s) * ps + (cap_b + in_b) * pb
            out.append({"name": "usd", "cost": {"measure": "usd", "c_max": c_max}})
        elif m == "tokens_lambda1":
            out.append({"name": "tokens_lambda1", "cost": {"measure": "tokens", "lambda": 1.0}})
        elif m == "tokens_params":
            params = L["pairs"].get("nominal_params_b", {})
            if pair["small"] in params and pair["big"] in params:
                out.append({"name": "tokens_params",
                            "cost": {"measure": "tokens", "lambda": params[pair["big"]] / params[pair["small"]]}})
        else:
            raise ValueError(f"lock: unknown cost measure rule {m!r}")
    return out


def policy_grid(rows, meta, pair, L) -> dict:
    P = L["policies"]
    small = [r for r in rows if r.model == pair["small"]]
    per_q = defaultdict(int)
    for r in small:
        per_q[r.qid] += 1
    k_min = min(per_q.values())
    keys = bool(meta.answer_keys) and meta.answer_equivalence != "none"
    ks = [k for k in P["k"] if k <= k_min and (k == 1 or keys)]
    cap = (meta.caps or {}).get(pair["small"])
    cuts = [None] + (sorted(allowed_cuts(P, cap)) if cap else [])
    return {"k": ks, "tau": P["tau"], "cut": cuts, "on_unsure": P["on_unsure"]}


# ------------------------------------------------------------------ the naive rule
def naive_go(rr, alpha, s) -> tuple[bool, str | None]:
    """GO iff some policy has empirical risk <= alpha and empirical saving >= s on all data
    (the cheapest such policy is named)."""
    inp = rr.audit_input
    B = sum(inp.big_cost)
    best = None
    for p in inp.policies:
        sv = sum(p.served)
        if sv == 0 or B <= 0:
            continue
        risk = sum(p.error) / sv
        saving = 1 - sum(p.cost) / B
        if risk <= alpha and saving >= s:
            if best is None or saving > best[0]:
                best = (saving, p.name)
    return (best is not None), (best[1] if best else None)


# ------------------------------------------------------------------ run
DEFAULT_TIERS = ("primary", "secondary")


def list_configs(pools_dir, lock_path, only_pools=None, tiers=None) -> list[dict]:
    """Dry run: the configurations a run WOULD audit (structure only: no replay, no audit,
    no correctness read). For checking the lock before freezing it."""
    L = load_lock(lock_path)
    out = []
    for d in sorted(p for p in Path(pools_dir).iterdir() if (p / "meta.json").exists()):
        if only_pools and d.name not in only_pools:
            continue
        rows, rerr, meta, merr = load_pool_v1(d)
        if rerr or merr or meta is None:
            continue
        inc, why = included(d.name, L)
        for pair in pairs_for(d.name, rows, L):
            q = common_questions(rows, pair["small"], pair["big"])
            sub = [r for r in rows if r.model in (pair["small"], pair["big"]) and r.qid in q]
            grid = policy_grid(sub, meta, pair, L)
            tier = pair["tier"] if inc else "excluded"
            if tiers is not None and tier not in tiers:
                continue
            for ms in measures_for(d.name, sub, meta, pair, L):
                out.append({"pool": d.name, "small": pair["small"], "big": pair["big"],
                            "tier": tier, "n": pair["n_common"],
                            "measure": ms["name"], "policies": len(grid["k"]) * len(grid["cut"]) *
                            (len(grid["tau"]) if max(grid["k"]) > 1 else 1)})
    return out


def run(pools_dir, lock_path, out_dir, tiers=DEFAULT_TIERS, only_pools=None, log=print) -> dict:
    """Audit every configuration of the given tiers. LLM-judged benchmarks form the tier
    'excluded' and are audited only when asked (tiers including "excluded"): the lock lists
    them with counts, their labels carry the judge's error. Primary configurations are also
    run on the lock's sensitivity grid (alpha x s, same delta), reported beside the main cell."""
    L = load_lock(lock_path)
    lh = lock_hash(lock_path)
    status = L["status"]
    out = Path(out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    lines = []
    for d in sorted(p for p in Path(pools_dir).iterdir() if (p / "meta.json").exists()):
        if only_pools and d.name not in only_pools:
            continue
        rows, rerr, meta, merr = load_pool_v1(d)
        if rerr or merr or meta is None:
            log(f"{d.name}: unreadable pool, skipped ({(rerr or merr)[:1]})")
            continue
        inc, why = included(d.name, L)
        for pair in pairs_for(d.name, rows, L):
            tier = pair["tier"] if inc else "excluded"
            if tier not in tiers:
                continue
            q = common_questions(rows, pair["small"], pair["big"])
            sub = [r for r in rows if r.model in (pair["small"], pair["big"]) and r.qid in q]
            grid = policy_grid(sub, meta, pair, L)
            for ms in measures_for(d.name, sub, meta, pair, L):
                a = L["audit"]
                cfgd = {"small": pair["small"], "big": pair["big"], "alpha": a["alpha"], "delta": a["delta"],
                        "s": a["s"], "salt": a["salt"], "cost_test": a["cost_test"], "cost": ms["cost"],
                        "policies": grid}
                cfg = config_from_dict(cfgd)
                bad = check_config_against_lock(cfg, lock_path, caps=meta.caps)
                if bad:
                    raise ValueError(f"{d.name} {pair} {ms['name']}: config outside the lock: {bad}")
                rep = make_report(sub, meta, cfg, input_info={"pool": d.name, "lock": lh, "lock_status": status})
                text = dumps(rep)
                rid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                fn = f"{d.name}__{pair['small']}__{pair['big']}__{ms['name']}.json".replace("/", "_")
                (out / "reports" / fn).write_text(text, encoding="utf-8", newline="\n")
                v = rep["R1_verdict"]["verdict"]
                ng, npol, rr = None, None, None
                if v != "NOT RUN":
                    rr = replay(sub, meta, cfg)
                    ng, npol = naive_go(rr, cfg.alpha, cfg.s)
                sens = None
                if tier == "primary" and v != "NOT RUN":
                    sens = {}
                    for al, sv in sensitivity_cells(L):
                        cg = config_from_dict(dict(cfgd, alpha=al, s=sv))
                        bad = check_config_against_lock(cg, lock_path, caps=meta.caps, sensitivity=True)
                        if bad:
                            raise ValueError(f"{d.name} {pair} sensitivity: {bad}")
                        rp = make_report(sub, meta, cg, input_info={"pool": d.name, "lock": lh, "lock_status": status,
                                                                    "sensitivity_cell": True})
                        tx = dumps(rp)
                        sf = fn[:-5] + f"__a{al}_s{sv}.json"
                        (out / "reports" / "sensitivity").mkdir(exist_ok=True)
                        (out / "reports" / "sensitivity" / sf).write_text(tx, encoding="utf-8", newline="\n")
                        sng, _ = naive_go(rr, al, sv)
                        sens[f"alpha={al},s={sv}"] = {"verdict": rp["R1_verdict"]["verdict"], "naive_go": sng,
                                                     "report_sha16": hashlib.sha256(tx.encode("utf-8")).hexdigest()[:16]}
                cert = (rep.get("R4_break_even") or {}).get("certification") or {}
                lines.append({"pool": d.name, "source": source_of(d.name), "benchmark": benchmark_of(d.name),
                              "small": pair["small"], "big": pair["big"], "tier": tier, "excluded_reason": why or None,
                              "measure": ms["name"], "n": pair["n_common"], "verdict": v,
                              "chosen": rep["R1_verdict"].get("chosen_on_half_a"),
                              "naive_go": ng, "naive_policy": npol,
                              "questions_needed": cert.get("questions_needed"),
                              "reachable": cert.get("reachable"),
                              "cost_bound_ratio": (rep.get("R5_data_quality") or {}).get("cost_bound_ratio"),
                              "report": fn, "report_sha16": rid, "lock_status": status,
                              "sensitivity": sens})
        log(f"{d.name}: done")
    lines.sort(key=lambda x: (x["source"], x["benchmark"], x["tier"], x["small"], x["big"], x["measure"]))
    with open(out / "map.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in lines:
            f.write(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n")
    summ = summarize(lines, L, lh)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summ


def summarize(lines, L, lh) -> dict:
    """Lock criteria a-d per tier. Pairs share models and benchmarks: counts are of
    configurations, not independent experiments (models / benchmarks are listed too)."""
    out = {"lock_hash": lh, "lock_status": L["status"],
           "pre_registered": L["status"] == "frozen",
           "note": "counts are of configurations (pool x pair x measure); they share models and questions"}
    for tier in sorted({x["tier"] for x in lines}):
        X = [x for x in lines if x["tier"] == tier]
        run_ = [x for x in X if x["verdict"] != "NOT RUN"]
        verdicts = defaultdict(int)
        for x in X:
            verdicts[x["verdict"]] += 1
        a = verdicts["GO"] / len(run_) if run_ else None
        b = sum(1 for x in run_ if x["naive_go"] and x["verdict"] != "GO")
        bycfg = defaultdict(set)
        for x in run_:
            bycfg[(x["pool"], x["small"], x["big"])].add(x["verdict"])
        c = sum(1 for v in bycfg.values() if len(v) > 1)
        inc_ = [x for x in run_ if x["verdict"] == "INCONCLUSIVE"]
        qn = [x["questions_needed"] for x in inc_ if x["questions_needed"]]
        out[tier] = {"configurations": len(X), "verdicts": dict(verdicts),
                     "a_share_go": a,
                     "b_naive_go_but_audit_not": b,
                     "b_naive_go_total": sum(1 for x in run_ if x["naive_go"]),
                     "c_pairs_whose_verdict_changes_with_measure": c,
                     "c_pairs_with_several_measures": sum(1 for x in bycfg.values()),
                     "d_questions_needed_inconclusive": {"n": len(qn), "median": statistics.median(qn) if qn else None,
                                                         "min": min(qn) if qn else None, "max": max(qn) if qn else None,
                                                         "inconclusive": len(inc_),
                                                         "unreachable_with_these_candidates":
                                                             sum(1 for x in inc_ if x.get("reachable") is False)},
                     "models": len({x["small"] for x in X} | {x["big"] for x in X}),
                     "benchmarks": len({x["pool"] for x in X})}
        bys = defaultdict(lambda: defaultdict(int))
        for x in X:
            bys[x["source"]][x["verdict"]] += 1
        out[tier]["verdicts_by_source"] = {k: dict(v) for k, v in sorted(bys.items())}
        cells = defaultdict(lambda: defaultdict(int))
        for x in X:
            for c, r in (x.get("sensitivity") or {}).items():
                cells[c][r["verdict"]] += 1
                cells[c]["naive_go_but_audit_not"] += int(bool(r["naive_go"]) and r["verdict"] != "GO")
        if cells:
            out[tier]["sensitivity"] = {k: dict(v) for k, v in sorted(cells.items())}
    out["reporting"] = L.get("reporting")
    return out


def _tiers(t):
    return {"both": DEFAULT_TIERS, "all": ("primary", "secondary", "excluded")}.get(t, (t,))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="caac-crossmap")
    ap.add_argument("pools")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out")
    ap.add_argument("--tier", choices=["primary", "secondary", "both", "excluded", "all"], default="both",
                    help="both = primary + secondary (default); excluded = LLM-judged benchmarks only (exploratory)")
    ap.add_argument("--pool", action="append", help="only these pool folders (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="list the configurations only (no audit)")
    a = ap.parse_args(argv)
    if a.dry_run:
        cf = list_configs(a.pools, a.lock, only_pools=a.pool, tiers=_tiers(a.tier))
        for x in cf:
            print(json.dumps(x, sort_keys=True))
        by = defaultdict(int)
        for x in cf:
            by[x["tier"]] += 1
        print(json.dumps({"configurations": len(cf), "by_tier": dict(sorted(by.items()))}), file=sys.stderr)
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run")
    L = load_lock(a.lock)
    if L["status"] != "frozen":
        print("WARNING: the lock is a DRAFT: results are NOT pre-registered", file=sys.stderr)
    s = run(a.pools, a.lock, a.out, tiers=_tiers(a.tier), only_pools=a.pool)
    print(json.dumps(s, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
