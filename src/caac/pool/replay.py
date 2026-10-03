"""Replay harness: run a policy over a whole pool, aggregate the metrics the
paper reports, and build LambdaResult lists for LTT certification.

Metrics per policy run (over all questions):
  coverage        = served / total
  selective_risk  = wrong_served / served      (the quantity LTT bounds)
  accuracy        = correct_served / total     (coverage * (1 - selective_risk))
  mean_cost[prof] = average over ALL questions (served or not) of that profile
  action_dist     = fraction of questions taking each top-level action

Order-dependent policies are averaged over R permutations with common random
numbers (same permutations reused across policies for paired comparison).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from caac.pool.estimators import chow_coverage_bound
from caac.pool.ledger import PROFILE_NAMES, cost_profiles
from caac.pool.ltt import LambdaResult
from caac.pool.policies import PolicyOutcome

__all__ = ["RunMetrics", "run_policy", "make_orders", "pi_pop_after", "lambda_result_from_metrics"]


@dataclass
class RunMetrics:
    n: int
    served: int
    correct_served: int
    wrong_served: int
    mean_cost: dict = field(default_factory=dict)
    action_dist: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.served / self.n if self.n else 0.0

    @property
    def selective_risk(self) -> float:
        return self.wrong_served / self.served if self.served else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_served / self.n if self.n else 0.0


def make_orders(n_samples: int, n_perms: int, seed: int = 0) -> list[list[int]]:
    """Common random permutations of sample indices, shared across policies."""
    rng = random.Random(seed)
    orders = []
    for _ in range(n_perms):
        o = list(range(n_samples))
        rng.shuffle(o)
        orders.append(o)
    return orders


def run_policy(qids, policy_fn, orders=None) -> RunMetrics:
    """policy_fn(qid, order) -> PolicyOutcome. order is None for order-free policies.

    Averages order-dependent outcomes over ``orders`` (list of permutations).
    Cost is averaged over all questions AND permutations.
    """
    n = len(qids)
    served = correct = wrong = 0.0
    cost_acc = {p: 0.0 for p in PROFILE_NAMES}
    action_acc = {}
    reps = orders if orders else [None]

    for qid in qids:
        for order in reps:
            out: PolicyOutcome = policy_fn(qid, order)
            w = 1.0 / len(reps)
            if out.served:
                served += w
                if out.correct:
                    correct += w
                else:
                    wrong += w
            c = cost_profiles(out.calls)
            for p in PROFILE_NAMES:
                cost_acc[p] += c[p] * w
            action_acc[out.action] = action_acc.get(out.action, 0.0) + w

    mean_cost = {p: cost_acc[p] / n for p in PROFILE_NAMES}
    action_dist = {a: v / n for a, v in action_acc.items()}
    return RunMetrics(
        n=n, served=round(served), correct_served=round(correct),
        wrong_served=round(wrong), mean_cost=mean_cost, action_dist=action_dist,
    )


def lambda_result_from_metrics(lam, m: RunMetrics, cost_profile: str = "tokens") -> LambdaResult:
    return LambdaResult(
        lam=lam, n_served=m.served, n_wrong=m.wrong_served,
        coverage=m.coverage, mean_cost=m.mean_cost.get(cost_profile, 0.0),
    )


def pi_pop_after(qids, answer_fn) -> float:
    """Fraction of candidate answers that are correct AFTER an action.

    answer_fn(qid) -> bool (is the candidate answer this action produces correct).
    Feeds the Chow bound: chow_coverage_bound(pi_pop_after(...), alpha).
    """
    if not qids:
        return 0.0
    return sum(answer_fn(qid) for qid in qids) / len(qids)
