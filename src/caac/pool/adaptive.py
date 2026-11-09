"""Adaptive-sampling baselines: spend fewer samples, always answer.

These are the competitors for the ``sample_more`` axis (CAAC_MASTER §11). All
three decide HOW MANY samples to draw and then vote; none of them can abstain
and none can change which model answers, so they move along the cost axis at
fixed coverage 1.0 -- which is exactly why they belong in the Pareto plot and
not in the risk-certification table.

  adaptive_consistency  Aggarwal et al. 2023. Draw one sample at a time; stop as
                        soon as the posterior says the current winner is very
                        likely to stay the winner.
  esc                   Early-Stopping Self-Consistency, Li et al. 2024. Draw in
                        windows; stop at the first unanimous window.
  deepconf_offline      Generate k traces, drop the least confident ones, vote on
                        what is left (optionally weighted by confidence).

Two things this module is careful about, because they are the usual way such a
comparison gets rigged:

1. **Every drawn sample is billed**, including the ones a filter later throws
   away. ``deepconf_offline`` pays for all k traces even though it votes on
   ~eta of them -- offline filtering saves no compute at all, it only buys
   accuracy, and the ledger has to show that.

2. **Sequential draws cost rounds.** Adaptive-Consistency's saving is in tokens
   but its draws are serial, so under the ``rounds`` profile (the latency axis,
   ledger.py) it is the most expensive policy here. Each draw gets its own
   ``round_index``; ESC gets one per window; plain self-consistency gets one.
   A paper that reports only tokens would score these three very differently
   from one that reports latency, and we report both.

Every function takes the pool record plus an explicit ``order`` (the draw
permutation) so the replay harness can average over common random numbers.
"""

from __future__ import annotations

from collections import Counter
from math import comb

from caac.pool.ledger import Call
from caac.pool.policies import PolicyOutcome, _vote
from caac.pool.schema import PoolRecord

__all__ = [
    "beta_win_probability", "adaptive_consistency", "esc", "deepconf_offline",
    "trace_confidence", "ADAPTIVE_BASELINES",
]


def beta_win_probability(n1: int, n2: int) -> float:
    """P(p1 > p2) under independent Beta posteriors with a uniform prior.

    The Dirichlet stopping rule of Adaptive-Consistency, restricted to the top
    two answers (which is what their Beta variant evaluates). For integer
    counts this has a closed form via the incomplete beta's binomial identity,

        P(X > 1/2) = 1 - I_{1/2}(n1+1, n2+1),
        I_{1/2}(a, b) = 2^{-(a+b-1)} * sum_{j=a}^{a+b-1} C(a+b-1, j),

    so it is computed exactly in integer arithmetic -- no scipy, no quadrature,
    and no sampling noise in a stopping rule that decides the cost of every
    question.
    """
    a, b = n1 + 1, n2 + 1
    m = a + b - 1
    tail = sum(comb(m, j) for j in range(a, m + 1))
    return 1.0 - tail / (2.0 ** m)


def _billed(rec: PoolRecord, groups: list[list[int]],
            share_prefill: bool = True) -> list[Call]:
    """One Call per sequential group, each group's completions in parallel.

    ``share_prefill=True`` (default) charges the prompt ONCE across the whole
    sequence. Every group here re-sends the identical prompt, and any real
    server keeps that prefix in its KV cache, so billing it per draw would make
    Adaptive-Consistency look worse than plain self-consistency purely by
    accounting -- the method would be judged on our convention instead of on
    itself. We refuse to win that way.

    Note the asymmetry this creates, and state it in the paper: CAAC's own
    ``sample_more`` DOES re-pay prefill (§9), because there the second round
    follows a different, already-finished greedy call. The baselines therefore
    get the more generous convention and we still report the comparison.
    """
    out = []
    first = True
    for r, g in enumerate(groups):
        if not g:
            continue
        pre = rec.prompt_tokens if (first or not share_prefill) else 0
        out.append(Call(rec.model, pre, [rec.samples[i].decode_tokens for i in g], r))
        first = False
    return out


def _outcome(rec: PoolRecord, drawn: list[int], groups: list[list[int]],
             answer: str | None, action: str, share_prefill: bool = True,
             **meta) -> PolicyOutcome:
    out = PolicyOutcome(True, answer, rec.label_correct(answer),
                        _billed(rec, groups, share_prefill), action)
    out.meta.update({"n_drawn": len(drawn),
                     "n_rounds": sum(1 for g in groups if g)})
    out.meta.update(meta)
    return out


def adaptive_consistency(rec: PoolRecord, order: list[int], threshold: float = 0.95,
                         max_k: int = 8, min_k: int = 2,
                         share_prefill: bool = True) -> PolicyOutcome:
    """Draw one at a time; stop when the leader is ``threshold``-likely to hold.

    Serves the majority of whatever was drawn. Saves tokens on easy questions
    (the first few samples agree) and spends the full budget on hard ones --
    the same intuition as ``sample_more``, but with no notion of risk, so it
    cannot decline to answer.
    """
    canon = rec.canonical_sample_answers()
    idxs = [i for i in order if i < rec.n_samples][:max_k]
    drawn: list[int] = []
    groups: list[list[int]] = []
    stopped_at = None
    for i in idxs:
        drawn.append(i)
        groups.append([i])                      # one draw = one sequential round
        if len(drawn) < max(2, min_k):
            continue
        counts = Counter(a for a in (canon[j] for j in drawn) if a is not None)
        top = counts.most_common(2)
        n1 = top[0][1]
        n2 = top[1][1] if len(top) > 1 else 0
        if beta_win_probability(n1, n2) >= threshold:
            stopped_at = len(drawn)
            break
    answer = _vote([canon[j] for j in drawn])
    return _outcome(rec, drawn, groups, answer, "sample_more", share_prefill,
                    stopped_early=stopped_at is not None, stopped_at=stopped_at,
                    threshold=threshold)


def esc(rec: PoolRecord, order: list[int], window: int = 4,
        max_k: int = 8, share_prefill: bool = True) -> PolicyOutcome:
    """Early-Stopping Self-Consistency: stop at the first unanimous window.

    Cheaper in ROUNDS than Adaptive-Consistency (a window is drawn in parallel)
    and coarser in tokens, because the budget moves in steps of ``window``.
    """
    canon = rec.canonical_sample_answers()
    idxs = [i for i in order if i < rec.n_samples][:max_k]
    drawn: list[int] = []
    groups: list[list[int]] = []
    stopped_at = None
    for s in range(0, len(idxs), window):
        chunk = idxs[s:s + window]
        groups.append(chunk)
        drawn.extend(chunk)
        vals = [canon[j] for j in chunk]
        if len(set(vals)) == 1 and vals[0] is not None:
            stopped_at = len(drawn)
            break
    answer = _vote([canon[j] for j in drawn])
    return _outcome(rec, drawn, groups, answer, "sample_more", share_prefill,
                    stopped_early=stopped_at is not None, stopped_at=stopped_at,
                    window=window)


def trace_confidence(rec: PoolRecord, i: int) -> float:
    """Confidence of one sampled trace.

    Falls back to the completion's mean token probability, which is all an A2
    pool carries. The faithful DeepConf score is the mean over the LOWEST-
    confidence token group of the trace, which needs the per-token sidecar
    (``<pool>.tok.jsonl.gz``, Session 2) -- swap this function then and rerun;
    ``deepconf_offline`` takes it as an argument for exactly that reason.
    """
    c = rec.samples[i].confidence
    return 0.0 if c is None else float(c)


def deepconf_offline(rec: PoolRecord, order: list[int], k: int = 8, eta: float = 0.5,
                     weighted: bool = True, conf_fn=trace_confidence) -> PolicyOutcome:
    """Draw k traces, keep the top ``eta`` by confidence, vote on those.

    Offline filtering buys accuracy, never compute: all k traces are generated
    and therefore billed. ``weighted`` makes the vote confidence-weighted
    (DeepConf's default) rather than one-trace-one-vote.
    """
    canon = rec.canonical_sample_answers()
    idxs = [i for i in order if i < rec.n_samples][:k]
    if not idxs:
        return _outcome(rec, [], [], None, "sample_more", n_kept=0)
    scored = sorted(idxs, key=lambda i: conf_fn(rec, i), reverse=True)
    n_keep = max(1, int(round(eta * len(scored))))
    kept = scored[:n_keep]

    tally: dict[str, float] = {}
    for i in kept:
        a = canon[i]
        if a is None:
            continue
        tally[a] = tally.get(a, 0.0) + (conf_fn(rec, i) if weighted else 1.0)
    answer = max(tally, key=lambda a: (tally[a], a)) if tally else None
    # billed for ALL k, drawn in one parallel round
    return _outcome(rec, idxs, [idxs], answer, "sample_more",
                    n_kept=len(kept), eta=eta, weighted=weighted)


# name -> (fn, knob name) for the sweep in adaptive_baselines.py
ADAPTIVE_BASELINES = {
    "adaptive_consistency": (adaptive_consistency, "threshold"),
    "esc": (esc, "window"),
    "deepconf_offline": (deepconf_offline, "eta"),
}
