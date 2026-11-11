"""One certification protocol for every method: CAAC and all baselines.

A *family* is a label-free grid of configurations plus the policy each one
runs. Replaying a family over a pool gives an outcome *table* (question x
configuration): served, wrong, and the tokens billed to each model. Calibration,
certification, selection, the price sweep and the re-split experiment are then
sums over rows of that table, so every method is scored by the same code.

Validity of the certificate:
  * Calibration counts come from ONE draw order per question -- the order the
    samples were generated in, which is i.i.d. and never looks at a label. Each
    count is then a sum of independent 0/1 outcomes and the binomial test is
    exact. Averaging over permutations is used only to REPORT held-out numbers.
  * The family and the order of every testing path are fixed before any
    calibration data is seen. A path halts at its first failure, so a first
    point that serves too few questions to ever pass would halt it at once;
    ``prune`` removes such points using the TRAINING split, never calibration. A path is tested by fixed-sequence testing at
    level delta / (number of paths): a one-dimensional threshold grid is one
    path (no multiplicity), a set of unordered schedules is one path per rule
    (plain Bonferroni), and a two-dimensional grid is one path per row.
  * Cost never enters the test. It enters only the choice among certified
    configurations, so the price ratio can be swept without touching the
    guarantee -- provided the family itself does not depend on the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from caac.pool.ledger import cost_profiles
from caac.pool.ltt import min_served_certifiable, pvalue_selective_risk
from caac.pool.schema import PoolRecord

__all__ = ["Family", "Table", "build_table", "prune", "certify", "plugin_select", "pick_max_coverage",
           "pick_min_cost", "summarize", "certified_frontier", "split_cost"]


@dataclass
class Family:
    name: str
    configs: list
    make: Callable                      # config -> fn(small_rec, big_rec | None, order) -> PolicyOutcome
    paths: list[list[int]] | None = None  # None: one path in config order (conservative first)
    needs_big: bool = False
    uses_order: bool = True             # False: the policy never reads the sample order

    def testing_paths(self) -> list[list[int]]:
        return self.paths if self.paths is not None else [list(range(len(self.configs)))]


def split_cost(calls, big_model: str | None) -> tuple[float, float]:
    """(tokens billed to the small model, tokens billed to the big one)."""
    s = b = 0.0
    for c in calls:
        t = c.prefill_tokens + sum(c.decode_tokens)
        if big_model is not None and c.model == big_model:
            b += t
        else:
            s += t
    return s, b


@dataclass
class Table:
    family: str
    qids: list[str]
    configs: list
    paths: list[list[int]]
    served: np.ndarray          # (Q, C) share of draw orders that served (0/1 with one order)
    wrong: np.ndarray           # (Q, C) share that served a wrong answer
    tok_small: np.ndarray       # (Q, C) tokens billed to the small model
    tok_big: np.ndarray         # (Q, C) tokens billed to the big model
    flops: np.ndarray           # (Q, C)
    rounds: np.ndarray          # (Q, C) sequential rounds (latency proxy)
    actions: list = field(default_factory=list)   # per config: {action: share of questions}
    action_q: np.ndarray | None = None            # (Q, C) action taken (one-order tables only)
    waste_small: np.ndarray | None = None         # (Q, C) small-model tokens spent beyond the greedy turn on an unserved question
    waste_big: np.ndarray | None = None           # (Q, C) big-model tokens spent on an unserved question

    def cost(self, lam: float) -> np.ndarray:
        """Cost in small-model tokens when one big-model token costs ``lam`` of them."""
        return self.tok_small + lam * self.tok_big

    def waste(self, lam: float) -> np.ndarray | None:
        """Cost that bought nothing: everything beyond the greedy turn, on questions
        the policy then declined to answer. The greedy turn itself is the price of
        finding out, so it is not counted."""
        if self.waste_small is None:
            return None
        return self.waste_small + lam * self.waste_big


def _flops(calls) -> float:
    """FLOPs from the cost registry; NaN for a model it does not know (tests)."""
    try:
        return cost_profiles(calls)["flops"]
    except KeyError:
        return float("nan")


def build_table(family: Family, qids, small: dict[str, PoolRecord],
                big: dict[str, PoolRecord] | None, orders: list[list[int]] | None = None) -> Table:
    """Replay every configuration on every question.

    ``orders=None`` uses the generation order of the samples -- the setting for
    calibration. A list of permutations averages over them -- for reporting.
    """
    qids = list(qids)
    if not family.uses_order:
        orders = None
    if orders is None:
        width = max((small[q].n_samples for q in qids), default=0)
        orders = [list(range(width))]
    Q, C, R = len(qids), len(family.configs), len(orders)
    shape = (Q, C)
    served, wrong = np.zeros(shape), np.zeros(shape)
    ts, tb, fl, rd = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    ws, wb = np.zeros(shape), np.zeros(shape)
    action_q = np.empty(shape, dtype=object) if R == 1 else None
    mix = [dict() for _ in range(C)]
    fns = [family.make(cfg) for cfg in family.configs]
    for qi, q in enumerate(qids):
        s = small[q]
        b = (big or {}).get(q)
        bm = b.model if b is not None else None
        for ci, fn in enumerate(fns):
            for o in orders:
                out = fn(s, b, o)
                if out.served:
                    served[qi, ci] += 1.0 / R
                    if not out.correct:
                        wrong[qi, ci] += 1.0 / R
                ss, bb = split_cost(out.calls, bm)
                ts[qi, ci] += ss / R
                tb[qi, ci] += bb / R
                if not out.served:
                    first = out.calls[0] if out.calls else None
                    floor = (first.prefill_tokens + sum(first.decode_tokens)
                             if first is not None and first.model == s.model
                             and first.decode_tokens == [s.greedy.decode_tokens] else 0)
                    ws[qi, ci] += max(0.0, ss - floor) / R
                    wb[qi, ci] += bb / R
                fl[qi, ci] += _flops(out.calls) / R
                rd[qi, ci] += len({c.round_index for c in out.calls}) / R
                mix[ci][out.action] = mix[ci].get(out.action, 0.0) + 1.0 / (R * Q)
                if action_q is not None:
                    action_q[qi, ci] = out.action
    return Table(family.name, qids, list(family.configs), family.testing_paths(),
                 served, wrong, ts, tb, fl, rd, mix, action_q, ws, wb)


def prune(family: Family, train: Table, n_calib: int, alpha: float, delta: float,
          slack: float = 1.5) -> Family:
    """Drop configurations whose coverage on the TRAINING split says they would
    serve too few calibration questions to ever be certified. Uses no
    calibration data, so the pruned family is still fixed in advance."""
    paths = family.testing_paths()
    need = slack * min_served_certifiable(alpha, delta / max(1, len(paths))) / max(1, n_calib)
    cov = train.served.mean(0)
    keep = [j for j in range(len(family.configs)) if cov[j] >= need]
    remap = {j: i for i, j in enumerate(keep)}
    new_paths = [[remap[j] for j in path if j in remap] for path in paths]
    new_paths = [pth for pth in new_paths if pth]
    return Family(family.name, [family.configs[j] for j in keep], family.make,
                  new_paths, family.needs_big, family.uses_order)


def _rows(table: Table, rows) -> np.ndarray:
    return np.arange(len(table.qids)) if rows is None else np.asarray(rows)


def certify(table: Table, rows, alpha: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Certified mask and p-values over the configurations, from ``rows`` only.

    Each path is a fixed sequence (conservative first) tested at delta / #paths.
    Counts must be integers: certify a one-order table, never an averaged one.
    """
    r = _rows(table, rows)
    cells = np.concatenate([table.served[r].ravel(), table.wrong[r].ravel()])
    if not np.allclose(cells, np.round(cells)):
        raise ValueError("certify needs a one-order table (0/1 outcomes per question)")
    n = table.served[r].sum(0)
    e = table.wrong[r].sum(0)
    n, e = np.round(n).astype(int), np.round(e).astype(int)
    pv = np.array([pvalue_selective_risk(int(n[j]), int(e[j]), alpha) for j in range(len(n))])
    ok = np.zeros(len(n), dtype=bool)
    level = delta / max(1, len(table.paths))
    for path in table.paths:
        for j in path:
            if n[j] > 0 and pv[j] <= level:
                ok[j] = True
            else:
                break
    return ok, pv


def plugin_select(table: Table, rows, alpha: float) -> np.ndarray:
    """No test at all: every configuration whose calibration risk is <= alpha.
    The yardstick that shows what the guarantee is worth."""
    r = _rows(table, rows)
    n = table.served[r].sum(0)
    e = table.wrong[r].sum(0)
    return (n > 0) & (e <= alpha * n)


def summarize(table: Table, rows, j: int, lam: float = 1.0) -> dict:
    r = _rows(table, rows)
    n = len(r)
    served = float(table.served[r, j].sum())
    wrong = float(table.wrong[r, j].sum())
    return {
        "config": table.configs[j],
        "n": n,
        "coverage": served / n if n else 0.0,
        "selective_risk": wrong / served if served else 0.0,
        "accuracy": (served - wrong) / n if n else 0.0,
        "cost": float(table.cost(lam)[r, j].mean()) if n else 0.0,
        "waste_share": _waste_share(table, r, j, lam),
        "tok_small": float(table.tok_small[r, j].mean()) if n else 0.0,
        "tok_big": float(table.tok_big[r, j].mean()) if n else 0.0,
        "flops": float(table.flops[r, j].mean()) if n else 0.0,
        "rounds": float(table.rounds[r, j].mean()) if n else 0.0,
    }


def _waste_share(table: Table, r, j: int, lam: float) -> float:
    w = table.waste(lam)
    total = float(table.cost(lam)[r, j].sum())
    return float(w[r, j].sum() / total) if w is not None and total > 0 else 0.0


def pick_max_coverage(table: Table, rows, ok: np.ndarray) -> int | None:
    r = _rows(table, rows)
    if not ok.any():
        return None
    cov = table.served[r].mean(0)
    return int(np.flatnonzero(ok)[np.argmax(cov[ok])])


def pick_min_cost(table: Table, rows, ok: np.ndarray, lam: float,
                  min_coverage: float = 0.0) -> int | None:
    """Cheapest certified configuration whose calibration coverage reaches the floor."""
    r = _rows(table, rows)
    cov = table.served[r].mean(0)
    cost = table.cost(lam)[r].mean(0)
    cand = np.flatnonzero(ok & (cov >= min_coverage - 1e-12))
    if len(cand) == 0:
        return None
    return int(cand[np.lexsort((-cov[cand], cost[cand]))[0]])


def certified_frontier(table: Table, rows, ok: np.ndarray, lam: float) -> list[int]:
    """Certified configurations not beaten on both coverage and cost (calibration)."""
    r = _rows(table, rows)
    cov = table.served[r].mean(0)
    cost = table.cost(lam)[r].mean(0)
    idx = sorted(np.flatnonzero(ok), key=lambda j: (cost[j], -cov[j]))
    front, best = [], -1.0
    for j in idx:
        if cov[j] > best + 1e-12:
            front.append(int(j))
            best = cov[j]
    return front
