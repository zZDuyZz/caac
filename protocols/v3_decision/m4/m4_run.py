"""M4: do the report's predictions come true? (design: DECISION_m4_before_run.md, hashed before any run)

Run from the repository root:
  python protocols/v3_decision/m4/m4_run.py --pools POOLS_DIR --map MAP_REL_JSONL --out OUT_DIR --workers 18

Every report is made by the official make_report with the official configuration of the v3-8 relative map;
only the set of questions changes. Output: OUT_DIR/m4_header.json, OUT_DIR/m4_runs.jsonl (one line per
configuration, sorted), OUT_DIR/m4_files.sha256.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, platform, random, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from caac.pool_v1.config import config_from_dict                                   # noqa: E402
from caac.pool_v1.crossmap import common_questions, measures_for, policy_grid      # noqa: E402
from caac.pool_v1.io import load_pool_v1                                           # noqa: E402
from caac.pool_v1.lock import lock_hash, load_lock                                 # noqa: E402
from caac.pool_v1.report import dumps, make_report                                 # noqa: E402

N_SUBS = (250, 500)
REPS = 10
SOURCES = ("lrb", "lurume")
N_MIN = 1000


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def config_id(x) -> str:
    return f"{x['pool']}|{x['small']}|{x['big']}|{x['measure']}"


def select(map_path):
    out = []
    for l in open(map_path, encoding="utf-8"):
        x = json.loads(l)
        if x["source"] in SOURCES and x["n"] >= N_MIN and x["verdict"] != "NOT RUN" and x.get("claim", "relative") == "relative":
            out.append(x)
    out.sort(key=config_id)
    return out


# ------------------------------------------------------------------ worker
_W = {}


def _init(pools_dir, lock_path):
    _W["pools"] = Path(pools_dir)
    _W["lock_path"] = lock_path
    _W["L"] = load_lock(lock_path)
    _W["lh"] = lock_hash(lock_path)
    _W["cache"] = {}


def _pool(name):
    c = _W["cache"]
    if name not in c:
        c.clear()                                    # configurations arrive grouped by pool
        rows, rerr, meta, merr = load_pool_v1(_W["pools"] / name)
        if rerr or merr or meta is None:
            raise RuntimeError(f"{name}: unreadable pool")
        c[name] = (rows, meta)
    return c[name]


def _build(x):
    """The official configuration of one map line (same code path as crossmap.run)."""
    L = _W["L"]
    rows, meta = _pool(x["pool"])
    pair = {"small": x["small"], "big": x["big"]}
    q = common_questions(rows, pair["small"], pair["big"])
    sub = [r for r in rows if r.model in (pair["small"], pair["big"]) and r.qid in q]
    grid = policy_grid(sub, meta, pair, L)
    ms = [m for m in measures_for(x["pool"], sub, meta, pair, L) if m["name"] == x["measure"]]
    if len(ms) != 1:
        raise RuntimeError(f"{config_id(x)}: measure not found")
    a = L["audit"]
    cfgd = {"small": pair["small"], "big": pair["big"], "alpha": a["alpha"], "delta": a["delta"], "s": a["s"],
            "salt": a["salt"], "cost_test": a["cost_test"], "cost": ms[0]["cost"], "policies": grid,
            "design": a.get("design", "v2"), "claim": "relative", "eps": a["eps"]}
    return sub, meta, config_from_dict(cfgd), sorted(q)


def _report(rows, meta, cfg, pool):
    rep = make_report(rows, meta, cfg, input_info={"pool": pool, "lock": _W["lh"], "lock_status": _W["L"]["status"]})
    return rep, hashlib.sha256(dumps(rep).encode("utf-8")).hexdigest()[:16]


def _pol(rep, name):
    for p in rep["R2_estimates"].get("all_policies") or []:
        if p["policy"] == name:
            return p
    return None


def _brief(rep, sha):
    v = rep["R1_verdict"]
    a = v.get("action") or {}
    cs = v.get("certified_saving") or {}
    return {"verdict": v["verdict"], "sha16": sha, "chosen": v.get("chosen"), "kind": a.get("kind"),
            "type": a.get("type"), "blocking": a.get("blocking"), "below_floor": a.get("below_floor"),
            "questions_needed": a.get("questions_needed"), "S_star": cs.get("value")}


def job(x, n_sub, rep_i, sub_rows=None, built=None):
    sub, meta, cfg, F = built or _build(x)
    cid = config_id(x)
    rng = random.Random(seed_of("M4", cid, n_sub, rep_i))
    S = sorted(rng.sample(F, n_sub))
    Sset = set(S)
    rs, hs = _report([r for r in sub if r.qid in Sset], meta, cfg, x["pool"])
    out = {"n_sub": n_sub, "rep": rep_i, "sub": _brief(rs, hs)}
    v = rs["R1_verdict"]["verdict"]
    H = [q for q in F if q not in Sset]
    if v == "GO":
        Hs = set(H)
        rh, hh = _report([r for r in sub if r.qid in Hs], meta, cfg, x["pool"])
        p = _pol(rh, rs["R1_verdict"]["chosen"]) or {}
        out["follow"] = {"kind": "heldout", "n": len(H), "sha16": hh, "delta_hat": p.get("delta_hat"),
                         "saving": p.get("saving"), "p_risk_above": p.get("p_risk_above"),
                         "p_saving_below": p.get("p_saving_below")}
    elif v == "INCONCLUSIVE" and out["sub"]["type"] == "reachable":
        Q = out["sub"]["questions_needed"]
        if Q is None or Q > len(F):
            out["follow"] = {"kind": "uncheckable", "Q": Q, "n_full": len(F)}
        else:
            rng2 = random.Random(seed_of("M4", cid, n_sub, rep_i, "ext"))
            E = Sset | set(rng2.sample(H, max(0, Q - n_sub)))
            re_, he = _report([r for r in sub if r.qid in E], meta, cfg, x["pool"])
            out["follow"] = {"kind": "extension", "n": len(E), "sha16": he, "verdict": re_["R1_verdict"]["verdict"]}
    else:
        out["follow"] = {"kind": "full_map", "verdict": x["verdict"]}
    return out


def run_config(x):
    t0 = time.time()
    built = _build(x)
    sub, meta, cfg, F = built
    rf, hf = _report(sub, meta, cfg, x["pool"])
    res = {"config": config_id(x), "tier": x["tier"], "n_full": len(F), "full_verdict": x["verdict"],
           "c0_sha16": hf, "c0_ok": hf == x["report_sha16"], "runs": []}
    for n_sub in N_SUBS:            # run even on a C0 mismatch (kept for diagnosis; analysis stops on any mismatch)
        for i in range(REPS):
            res["runs"].append(job(x, n_sub, i, built=built))
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_jobs(spec):
    """Re-run single (config, n_sub, rep) jobs (verification)."""
    x, n_sub, i = spec
    return config_id(x), n_sub, i, job(x, n_sub, i)


def header(args, configs):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "protocols"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:                                # noqa: BLE001
        commit, dirty = None, None
    code = {Path(p).name: sha256_file(p) for p in sorted(glob.glob(str(ROOT / "src/caac/pool_v1/*.py")))}
    pools = sorted({x["pool"] for x in configs})
    data = {p: {"meta.json": sha256_file(Path(args.pools) / p / "meta.json"),
                "rows.jsonl.gz": sha256_file(Path(args.pools) / p / "rows.jsonl.gz")} for p in pools}
    return {"commit": commit, "dirty_src_or_protocols": dirty, "python": sys.version.split()[0],
            "platform": platform.platform(), "lock_hash": lock_hash(args.lock), "map_sha256": sha256_file(args.map),
            "script_sha256": sha256_file(__file__), "code_sha256": code, "data_sha256": data,
            "n_configs": len(configs), "n_subs": N_SUBS, "reps": REPS, "workers": args.workers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--lock", default=str(ROOT / "protocols" / "audit_lock_v3.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="first N configurations only (smoke test)")
    ap.add_argument("--verify", default=None, help="JSON list of [config_id, n_sub, rep] to re-run")
    a = ap.parse_args()
    configs = select(a.map)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.verify:
        want = json.load(open(a.verify))
        byid = {config_id(x): x for x in configs}
        specs = [(byid[c], n, i) for c, n, i in want]
        with Pool(a.workers, initializer=_init, initargs=(a.pools, a.lock)) as P:
            res = P.map(run_jobs, specs, chunksize=1)
        json.dump([{"config": c, "n_sub": n, "rep": i, "result": r} for c, n, i, r in res],
                  open(out / "m4_verify.json", "w"), indent=1, sort_keys=True)
        print(f"verified {len(res)} jobs -> {out / 'm4_verify.json'}")
        return
    if a.limit:
        configs = configs[:a.limit]
    h = header(a, configs)
    json.dump(h, open(out / "m4_header.json", "w"), indent=1, sort_keys=True)
    print(f"M4: {len(configs)} configurations x {len(N_SUBS)} sizes x {REPS} reps, {a.workers} workers; commit {h['commit']}")
    t0, done, results = time.time(), 0, []
    part = open(out / "m4_runs.partial.jsonl", "w", encoding="utf-8")
    with Pool(a.workers, initializer=_init, initargs=(a.pools, a.lock)) as P:
        for r in P.imap_unordered(run_config, configs, chunksize=1):
            results.append(r); done += 1
            part.write(json.dumps(r, sort_keys=True) + "\n"); part.flush()
            if not r["c0_ok"]:
                print(f"C0 MISMATCH {r['config']}: {r['c0_sha16']}", flush=True)
            if done % 50 == 0 or done == len(configs):
                el = time.time() - t0
                print(f"{done}/{len(configs)} configurations, {el/60:.1f} min, eta {el/done*(len(configs)-done)/60:.0f} min", flush=True)
    part.close()
    results.sort(key=lambda r: r["config"])
    with open(out / "m4_runs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            rr = dict(r); rr.pop("seconds", None)
            f.write(json.dumps(rr, sort_keys=True) + "\n")
    bad = sum(not r["c0_ok"] for r in results)
    files = ["m4_header.json", "m4_runs.jsonl"]
    with open(out / "m4_files.sha256", "w", newline="\n") as f:
        for fn in files:
            f.write(f"{sha256_file(out / fn)}  {fn}\n")
    print(f"done: {len(results)} configurations, C0 mismatches: {bad}, {sum(len(r['runs']) for r in results)} runs")
    print(open(out / "m4_files.sha256").read())


if __name__ == "__main__":
    main()
