"""Offline replay of small-model-first policies on CAAC-Pool v1 rows.

A policy is fixed by four knobs, all chosen BEFORE the audit data is read:

  k          small-model draws used (draws 0..k-1). k >= 2 needs answer keys (L2).
  tau        agreement needed to SERVE: (draws whose key equals the majority key) / k.
  cut        early stop: a draw is stopped after `cut` output tokens; a stopped draw
             has no answer and costs `cut` tokens. None = no early stop. Needs only
             out_tokens and correct (L1).
  on_unsure  what happens when the small model is not confident:
             "escalate" -> the big model's draw 0 is served (its correctness counts);
             "abstain"  -> nothing is served (not an error, but not served).

Per question:
  * a small draw has an answer iff it was not cut by `cut` and finish != "length";
  * with k == 1 the small model is confident iff draw 0 has an answer;
  * with k >= 2 it is confident iff the majority key (ties -> earliest draw) is held
    by >= tau * k draws; the served answer is the first draw with the majority key;
  * cost = sum over used small draws of (tokens used x price_out_small)
           + prompt tokens x price_in_small (once)
           [+ big draw 0 tokens x price_out_big + prompt x price_in_big, if escalated];
  * B (big only) = big draw 0 tokens x price_out_big + prompt x price_in_big.

Draw 0 is the answer the system would serve first. When a source has no greedy
draw, draw 0 is a sample (the validator says so: I_FIRST_SAMPLED).
For usd / gpu_s the per-row measured cost is used; a cut draw is charged in
proportion to the tokens used (an ESTIMATE, flagged in the result).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

from caac.pool_v1.audit import AuditInput, PolicyData
from caac.pool_v1.config import Config
from caac.pool_v1.schema import Meta, Row


# ----------------------------------------------------------------------------- C5
def normalize_key(key: str | None, equivalence: str) -> str | None:
    """Answer equivalence. math: the adapter already canonicalized (unchanged);
    exact: whitespace collapsed + case folded; none: answers cannot be compared."""
    if key is None or equivalence == "none":
        return None
    if equivalence == "math":
        return key
    if equivalence == "exact":
        return " ".join(key.split()).casefold()
    raise ValueError(f"unknown answer_equivalence {equivalence!r}")


@dataclass(frozen=True)
class Policy:
    k: int
    tau: float
    cut: int | None
    on_unsure: str

    @property
    def name(self) -> str:
        c = "none" if self.cut is None else str(self.cut)
        return f"k{self.k}_tau{self.tau:g}_cut{c}_{self.on_unsure}"


def policy_set(cfg: Config) -> list[Policy]:
    """The candidate set Pi, in a fixed order (k=1 ignores tau: one policy, tau=1)."""
    g = cfg.policies
    out = []
    for k, tau, cut, u in product(sorted(g.k), sorted(g.tau), g.cut, g.on_unsure):
        p = Policy(k, 1.0 if k == 1 else float(tau), cut, u)
        if p not in out:
            out.append(p)
    return out


@dataclass
class ReplayResult:
    audit_input: AuditInput
    policies: list[Policy]
    qids: list[str]
    notes: list[str] = field(default_factory=list)
    # per policy, per question, for the report (tokens, not prices)
    small_tokens: dict = field(default_factory=dict)
    big_tokens_escalated: dict = field(default_factory=dict)
    big_only_tokens: list = field(default_factory=list)
    serve_small: dict = field(default_factory=dict)


def _index(rows: list[Row], model: str, split: str | None):
    by_q: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.model == model and (split is None or r.split == split):
            by_q[r.qid].append(r)
    for q in by_q:
        by_q[q].sort(key=lambda r: r.draw)
    return by_q


def _unit_cost(r: Row, used: int, measure: str, price_out: float) -> tuple[float, bool]:
    """Cost of the output part of one draw using `used` tokens; (cost, estimated?)."""
    if measure == "tokens":
        return used * price_out, False
    val = r.cost_usd if measure == "usd" else r.gpu_s
    if val is None:
        raise ValueError(f"measure '{measure}' but row ({r.qid}, {r.model}, draw {r.draw}) has no value")
    full = r.out_tokens or 0
    if used >= full or full == 0:
        return float(val), False
    return float(val) * used / full, True


def _prefill(r: Row, measure: str, price_in: float) -> float:
    return (r.in_tokens or 0) * price_in if measure == "tokens" else 0.0


def replay(rows: list[Row], meta: Meta | None, cfg: Config) -> ReplayResult:
    eq = meta.answer_equivalence if meta else "none"
    split = cfg.audit_split
    S, Bq = _index(rows, cfg.small, split), _index(rows, cfg.big, split)
    if not S:
        raise ValueError(f"no rows for small model {cfg.small!r} (split={split})")
    if not Bq:
        raise ValueError(f"no rows for big model {cfg.big!r} (split={split})")
    qids = sorted(set(S) & set(Bq))
    notes = []
    dropped = len(set(S) ^ set(Bq))
    if dropped:
        notes.append(f"{dropped} question(s) present for only one of the two models were left out")
    pols = policy_set(cfg)
    kmax = max(p.k for p in pols)
    short = [q for q in qids if len(S[q]) < kmax]
    if short:
        raise ValueError(f"{len(short)} question(s) have fewer than k={kmax} small draws "
                         f"(e.g. {short[0]}); lower k in the config")
    if kmax >= 2 and eq == "none":
        raise ValueError("k >= 2 needs comparable answers, but answer_equivalence is 'none'")
    c = cfg.cost
    m = c.measure
    est = False

    big_cost, big_tok = [], []
    for q in qids:
        b = Bq[q][0]
        if b.out_tokens is None and m == "tokens":
            raise ValueError(f"big model row for {q} has no out_tokens")
        uc, e = _unit_cost(b, b.out_tokens or 0, m, c.price_out_big)
        big_cost.append(uc + _prefill(b, m, c.price_in_big))
        big_tok.append(b.out_tokens or 0)

    pdata, stok, btok, ssmall = [], {}, {}, {}
    for p in pols:
        served, error, cost, st, bt, sm = [], [], [], [], [], []
        for qi, q in enumerate(qids):
            draws = S[q][:p.k]
            used_cost, used_tok, keys, ans_rows = 0.0, 0, [], []
            for r in draws:
                if r.out_tokens is None:
                    if p.cut is not None or m == "tokens":
                        raise ValueError(f"small row ({q}, draw {r.draw}) has no out_tokens")
                full = r.out_tokens or 0
                used = full if p.cut is None else min(full, p.cut)
                cut_hit = p.cut is not None and full > p.cut
                uc, e = _unit_cost(r, used, m, c.price_out_small)
                est |= e
                used_cost += uc
                used_tok += used
                has_answer = (not cut_hit) and r.finish != "length"
                keys.append(normalize_key(r.answer_key, eq) if has_answer else None)
                ans_rows.append(r if has_answer else None)
            used_cost += _prefill(draws[0], m, c.price_in_small)
            if p.k == 1:
                confident = ans_rows[0] is not None
                pick = ans_rows[0]
            else:
                counts = defaultdict(int)
                for kk in keys:
                    if kk is not None:
                        counts[kk] += 1
                if counts:
                    top = max(counts.values())
                    maj = next(kk for kk in keys if kk is not None and counts[kk] == top)
                    confident = top >= p.tau * p.k
                    pick = ans_rows[keys.index(maj)]
                else:
                    confident, pick = False, None
            if confident:
                served.append(True)
                error.append(not pick.correct)
                cost.append(used_cost)
                bt.append(0)
                sm.append(True)
            else:
                b = Bq[q][0]
                if p.on_unsure == "escalate":
                    served.append(True)
                    error.append(not b.correct)
                    cost.append(used_cost + big_cost[qi])
                    bt.append(big_tok[qi])
                else:
                    served.append(False)
                    error.append(False)
                    cost.append(used_cost)
                    bt.append(0)
                sm.append(False)
            st.append(used_tok)
        pdata.append(PolicyData(p.name, served, error, cost))
        stok[p.name], btok[p.name], ssmall[p.name] = st, bt, sm

    if c.c_max is not None:
        c_max = float(c.c_max)
    else:  # tokens: the generation caps make the bound
        caps = (meta.caps if meta else {}) or {}
        if cfg.small not in caps or cfg.big not in caps:
            raise ValueError("token cost needs meta.caps for both models (the cost bound comes from "
                             "the generation cap), or give cost.c_max in the config")
        in_max_s = max((S[q][0].in_tokens or 0) for q in qids)
        in_max_b = max((Bq[q][0].in_tokens or 0) for q in qids)
        cap_s = caps[cfg.small] if all(p.cut is None for p in pols) else max(
            min(caps[cfg.small], p.cut or caps[cfg.small]) for p in pols)
        c_max = (kmax * cap_s * c.price_out_small + in_max_s * c.price_in_small
                 + caps[cfg.big] * c.price_out_big + in_max_b * c.price_in_big)
        notes.append("cost bound c_max from the generation caps in meta.json")
    if est:
        notes.append(f"{m}: cost of early-stopped draws is ESTIMATED in proportion to tokens used")
    inp = AuditInput(qids=qids, big_cost=big_cost, policies=pdata, c_max=c_max)
    return ReplayResult(inp, pols, qids, notes, stok, btok, big_tok, ssmall)
