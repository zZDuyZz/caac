"""Integrity checks, data-quality flags and capability detection for CAAC-Pool v1.

The validator never fixes data. It reports:

  * ERRORS   -- the data cannot be used as is (duplicates, a question in two
                splits, unparsable rows, invalid meta). Audit refuses to run.
  * WARNINGS -- usable, but a conclusion will be weaker or needs a caveat
                (uneven draws, truncation, heavy cost tail, accuracy near 0/1,
                too few questions for a certificate, adapter flags).
  * INFO     -- what was found (counts, which analyses are possible).

Capabilities are detected per model and intersected for a (small, big) pair, so a
report states exactly which conclusions the data allows (package docstring, L0-L4).
Every number here is computed from the rows; thresholds are module constants so a
reader can see and change them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from caac.pool_v1.schema import EQUIVALENCE, FORMAT, GRADER_KINDS, SPLITS, VERSION, Meta, Row

# warning thresholds (documented in the report; not tuned on any result)
TRUNC_WARN = 0.05        # share of generations cut by the cap
CAP_HIT_WARN = 0.02      # share with out_tokens >= cap (when finish is missing)
TAIL_RATIO_WARN = 5.0    # p95 / median of out_tokens
ACC_EXTREME = 0.01       # accuracy below this or above 1 - this
CAPABILITIES = ("risk", "audit", "early_stop", "sample_more", "truncation_check",
                "gpu_cost", "money_cost")


@dataclass(frozen=True, order=True)
class Issue:
    severity: str          # "error" | "warning" | "info"
    code: str
    model: str
    message: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    capabilities: dict[str, list[str]] = field(default_factory=dict)   # model -> sorted caps
    level: dict[str, str] = field(default_factory=dict)                # model -> "L0".."L4"
    stats: dict[str, dict] = field(default_factory=dict)
    pair: dict | None = None

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "issues": [i.__dict__ for i in sorted(self.issues, key=_issue_key)],
                "capabilities": {m: self.capabilities[m] for m in sorted(self.capabilities)},
                "level": {m: self.level[m] for m in sorted(self.level)},
                "stats": {m: self.stats[m] for m in sorted(self.stats)},
                "pair": self.pair}


_SEV = {"error": 0, "warning": 1, "info": 2}


def _issue_key(i: Issue):
    return (_SEV[i.severity], i.code, i.model, i.message)


def certification_floor(alpha: float, delta: float) -> int:
    """Smallest number of served answers with ZERO errors that rejects 'risk > alpha'
    by an exact binomial test at level delta: (1-alpha)^n <= delta."""
    return math.ceil(math.log(delta) / math.log(1.0 - alpha))


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return float(s[k])


def _meta_issues(meta: Meta | None, meta_errors) -> list[Issue]:
    out = [Issue("error", "E_META", "*", e) for e in meta_errors]
    if meta is None:
        if not meta_errors:
            out.append(Issue("error", "E_META", "*", "meta.json missing"))
        return out
    if meta.format != FORMAT or str(meta.version) != VERSION:
        out.append(Issue("error", "E_META", "*",
                         f"format/version is '{meta.format}'/'{meta.version}', expected '{FORMAT}'/'{VERSION}'"))
    if meta.answer_equivalence not in EQUIVALENCE:
        out.append(Issue("error", "E_META", "*",
                         f"answer_equivalence '{meta.answer_equivalence}' not in {EQUIVALENCE}"))
    kind = (meta.grader or {}).get("kind")
    if kind not in GRADER_KINDS:
        out.append(Issue("error", "E_META", "*", f"grader.kind '{kind}' not in {GRADER_KINDS}"))
    for k in ("name", "url", "version_or_sha"):
        if not (meta.source or {}).get(k):
            out.append(Issue("warning", "W_META_SOURCE", "*",
                             f"source.{k} is empty: the report cannot say exactly where the data came from"))
    if kind == "llm-judge":
        out.append(Issue("warning", "W_JUDGE", "*",
                         "correctness comes from an LLM judge: its own error is not covered by the certificate"))
    return out


def _model_caps(rows: list[Row], by_q: dict, meta: Meta | None) -> tuple[set, list[str]]:
    caps = {"risk"}
    notes = []
    has_tok = all(r.out_tokens is not None for r in rows)
    has_cost = all(r.cost_usd is not None for r in rows)
    if has_tok or has_cost:
        caps |= {"audit"}
    if has_tok:
        caps |= {"early_stop"}
    k_min = min(len(d) for d in by_q.values()) if by_q else 0
    equiv_ok = meta is None or meta.answer_equivalence != "none"
    keys_ok = bool(meta and meta.answer_keys) or all(r.answer_key is not None for r in rows)
    if k_min >= 2 and keys_ok and equiv_ok:
        caps |= {"sample_more"}
    has_finish = all(r.finish is not None for r in rows)
    cap = (meta.caps or {}).get(rows[0].model) if (meta and rows) else None
    if has_finish:
        caps |= {"truncation_check"}
    elif cap is not None and has_tok:
        caps |= {"truncation_check"}
        notes.append(f"finish missing: truncation inferred from out_tokens >= cap {cap}")
    if all(r.gpu_s is not None for r in rows):
        caps |= {"gpu_cost"}
    if has_cost:
        caps |= {"money_cost"}
    return caps, notes


def _level(caps: set) -> str:
    """Cumulative summary of the capability set (the set itself is authoritative)."""
    lvl = 0
    for need in (("audit",), ("sample_more",), ("truncation_check",)):
        if all(c in caps for c in need):
            lvl += 1
        else:
            break
    if lvl == 3 and ({"gpu_cost", "money_cost"} & caps):
        lvl = 4
    return f"L{lvl}"


def levels(rows: list[Row], meta: Meta | None = None) -> dict[str, tuple[str, list[str]]]:
    """model -> (level, sorted capabilities)."""
    rep = validate(rows, meta, check_meta=False)
    return {m: (rep.level[m], rep.capabilities[m]) for m in rep.level}


def validate(rows: list[Row], meta: Meta | None = None, *, row_errors=(), meta_errors=(),
             alpha: float = 0.10, delta: float = 0.10, pair: tuple[str, str] | None = None,
             check_meta: bool = True, design: str = "v2") -> Report:
    rep = Report()
    if check_meta:
        rep.issues += _meta_issues(meta, list(meta_errors))
    rep.issues += [Issue("error", "E_PARSE", "*", f"line {ln}: {msg}") for ln, msg in row_errors]

    # ---- whole-pool checks
    seen = Counter((r.qid, r.model, r.draw) for r in rows)
    dups = [k for k, c in seen.items() if c > 1]
    if dups:
        rep.issues.append(Issue("error", "E_DUP", "*",
                                f"{len(dups)} duplicate (qid, model, draw), e.g. {sorted(dups)[0]}"))
    bad_split = sorted({r.split for r in rows if r.split is not None and r.split not in SPLITS})
    if bad_split:
        rep.issues.append(Issue("error", "E_SPLIT", "*", f"split values not in {SPLITS}: {bad_split}"))
    q_splits = defaultdict(set)
    for r in rows:
        if r.split is not None:
            q_splits[r.qid].add(r.split)
    leak = sorted(q for q, s in q_splits.items() if len(s) > 1)
    if leak:
        rep.issues.append(Issue("error", "E_LEAK", "*",
                                f"{len(leak)} question(s) appear in more than one split, e.g. {leak[0]}"))
    q_groups = defaultdict(set)
    for r in rows:
        if r.group is not None:
            q_groups[r.qid].add(r.group)
    mixed = sorted(q for q, g in q_groups.items() if len(g) > 1)
    if mixed:
        rep.issues.append(Issue("warning", "W_GROUP_MIXED", "*",
                                f"{len(mixed)} question(s) carry different group labels across rows, e.g. "
                                f"{mixed[0]}; the report uses the small model's draw 0"))
    neg = [r for r in rows if any(v is not None and v < 0 for v in (r.out_tokens, r.in_tokens, r.gpu_s, r.cost_usd))]
    if neg:
        rep.issues.append(Issue("error", "E_NEG", "*", f"{len(neg)} row(s) with a negative count or cost"))
    unknown = Counter(f for r in rows for f in r.flags if f.startswith("unknown_field:"))
    for f, c in sorted(unknown.items()):
        rep.issues.append(Issue("warning", "W_UNKNOWN_FIELD", "*", f"{f.split(':', 1)[1]}: in {c} row(s), ignored"))

    # ---- per model
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)
    floor = certification_floor(alpha, delta)
    for m in sorted(by_model):
        mr = by_model[m]
        by_q = defaultdict(dict)
        for r in mr:
            by_q[r.qid][r.draw] = r
        n_q = len(by_q)
        ks = [len(d) for d in by_q.values()]
        gaps = sum(1 for d in by_q.values() if sorted(d) != list(range(len(d))))
        if gaps:
            rep.issues.append(Issue("warning", "W_DRAW_GAP", m,
                                    f"{gaps} question(s) whose draws are not 0..k-1 (a draw is missing)"))
        if ks and min(ks) != max(ks):
            rep.issues.append(Issue("warning", "W_UNEVEN", m,
                                    f"draws per question range {min(ks)}..{max(ks)}: only the first {min(ks)} are used"))
        cap_m = (meta.caps or {}).get(m) if meta else None
        over = sum(1 for r in mr if cap_m is not None and r.out_tokens is not None and r.out_tokens > cap_m)
        if over:
            rep.issues.append(Issue("warning", "W_OUT_OVER_CAP", m,
                                    f"{over} row(s) with out_tokens above the cap {cap_m}: the cost bound "
                                    "c_max does not hold; raise the cap in meta.json to the largest observed"))
        greedy = Counter(r.qid for r in mr if r.is_greedy)
        if any(c > 1 for c in greedy.values()):
            rep.issues.append(Issue("error", "E_GREEDY", m, "more than one greedy draw for a question"))
        if any(r.is_greedy and r.draw != 0 for r in mr):
            rep.issues.append(Issue("error", "E_GREEDY", m, "a greedy draw that is not draw 0"))
        if not any(r.is_greedy for r in mr):
            rep.issues.append(Issue("info", "I_FIRST_SAMPLED", m,
                                    "no greedy draw: draw 0 (a sample) is served first; the report says so"))
        first = [d[0] for d in by_q.values() if 0 in d]
        acc = sum(r.correct for r in first) / len(first) if first else float("nan")
        if first and (acc < ACC_EXTREME or acc > 1 - ACC_EXTREME):
            rep.issues.append(Issue("warning", "W_ACC_EXTREME", m,
                                    f"first-answer accuracy {acc:.3f}: risk is near 0 or 1, an audit says little"))
        if design == "v3" and n_q < floor:
            rep.issues.append(Issue("warning", "W_N_FLOOR", m,
                                    f"{n_q} questions < {floor}: GO tests on all questions at level delta/|Pi| and needs "
                                    f">= {floor} served answers with zero errors even for one policy "
                                    f"(alpha={alpha}, delta={delta}); more with several policies; expect INCONCLUSIVE"))
        elif design != "v3" and n_q < 2 * floor:
            rep.issues.append(Issue("warning", "W_N_FLOOR", m,
                                    f"{n_q} questions < {2 * floor}: GO tests on half the data and needs >= {floor} "
                                    f"served answers with zero errors (alpha={alpha}, delta={delta}); expect INCONCLUSIVE"))
        toks = [r.out_tokens for r in mr if r.out_tokens is not None]
        zero = sum(1 for t in toks if t == 0)
        if zero:
            rep.issues.append(Issue("warning", "W_ZERO_TOK", m, f"{zero} generation(s) with out_tokens = 0"))
        tail = None
        if toks:
            med = _quantile(toks, 0.5)
            p95 = _quantile(toks, 0.95)
            tail = (p95 / med) if med > 0 else float("inf")
            if tail > TAIL_RATIO_WARN:
                rep.issues.append(Issue("warning", "W_HEAVY_TAIL", m,
                                        f"out_tokens p95/median = {p95:.0f}/{med:.0f} = {tail:.1f} > {TAIL_RATIO_WARN}: "
                                        "cost bounds are loose, an audit needs more questions"))
        fin = [r.finish for r in mr if r.finish is not None]
        trunc = (sum(1 for f in fin if f == "length") / len(fin)) if fin else None
        if trunc is not None and trunc > TRUNC_WARN:
            rep.issues.append(Issue("warning", "W_TRUNC", m,
                                    f"{trunc:.1%} of generations cut by the cap (finish=length)"))
        cap = (meta.caps or {}).get(m) if meta else None
        cap_hit = None
        if cap is not None and toks:
            cap_hit = sum(1 for t in toks if t >= cap) / len(toks)
            if not fin and cap_hit > CAP_HIT_WARN:
                rep.issues.append(Issue("warning", "W_CAP_HIT", m,
                                        f"{cap_hit:.1%} of generations reach the cap {cap} (finish missing: treated as cut)"))
        declared = bool(meta and meta.answer_keys)
        if declared:
            no_ans = sum(1 for r in mr if r.answer_key is None)
            if no_ans:
                rep.issues.append(Issue("info", "I_NO_ANSWER", m,
                                        f"{no_ans} generation(s) without an extractable answer (not votes)"))
            bad = [r for r in mr if r.answer_key is None and r.correct]
            if bad:
                cut = sum(1 for r in bad if r.finish == "length")
                rep.issues.append(Issue("warning", "W_CORRECT_NO_KEY", m,
                                        f"{len(bad)} generation(s) graded correct but without an answer_key "
                                        f"({cut} of them cut by the cap: a cut answer is not a vote by design)"))
        elif any(r.answer_key is None for r in mr) and ks and min(ks) >= 2:
            miss = sum(1 for r in mr if r.answer_key is None)
            rep.issues.append(Issue("warning", "W_KEY_MISSING", m,
                                    f"{miss} generation(s) without answer_key: SAMPLE MORE (voting) is disabled"))
        flags = Counter(f for r in mr for f in r.flags if not f.startswith("unknown_field:"))
        for f, c in sorted(flags.items()):
            rep.issues.append(Issue("warning", "W_FLAG", m, f"adapter flag '{f}' on {c} generation(s)"))
        caps, notes = _model_caps(mr, by_q, meta)
        for n in notes:
            rep.issues.append(Issue("info", "I_INFERRED", m, n))
        rep.capabilities[m] = sorted(caps, key=CAPABILITIES.index)
        rep.level[m] = _level(caps)
        rep.stats[m] = {"questions": n_q, "generations": len(mr),
                        "draws_min": min(ks) if ks else 0, "draws_max": max(ks) if ks else 0,
                        "first_answer_accuracy": round(acc, 4) if first else None,
                        "out_tokens_median": _quantile(toks, 0.5) if toks else None,
                        "out_tokens_p95": _quantile(toks, 0.95) if toks else None,
                        "tail_ratio": round(tail, 2) if tail not in (None, float("inf")) else tail,
                        "truncated_share": round(trunc, 4) if trunc is not None else None,
                        "cap_hit_share": round(cap_hit, 4) if cap_hit is not None else None}

    # ---- pair
    if pair is not None:
        s, b = pair
        for m in pair:
            if m not in by_model:
                rep.issues.append(Issue("error", "E_PAIR", m, "model of the pair not found in the rows"))
        if s in by_model and b in by_model:
            qs = {r.qid for r in by_model[s]}
            qb = {r.qid for r in by_model[b]}
            only_s, only_b = len(qs - qb), len(qb - qs)
            if only_s or only_b:
                rep.issues.append(Issue("warning", "W_PAIR_MISSING", f"{s}|{b}",
                                        f"{only_s} question(s) only for the small model, {only_b} only for the big one: "
                                        f"the pair uses the {len(qs & qb)} common questions"))
            common = sorted(set(rep.capabilities[s]) & set(rep.capabilities[b]), key=CAPABILITIES.index)
            rep.pair = {"small": s, "big": b, "common_questions": len(qs & qb),
                        "capabilities": common, "level": _level(set(common))}
    rep.issues.sort(key=_issue_key)
    return rep
