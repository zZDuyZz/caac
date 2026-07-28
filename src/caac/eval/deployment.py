"""Deployment measurement under load (phase P7 -- experiment E4).

Drives the controller through a serving engine under a Poisson request stream
and records latency percentiles, throughput, and peak KV, with full controller
and verifier overhead counted.

Measuring under load matters because one effect only appears there: BRANCH
raises memory pressure, which can *reduce* the feasible batch size and pull
throughput down. Summing per-request costs would never reveal it.

The scheduling and statistics work on CPU against the mock backend; a real
result needs a dedicated GPU so the latency numbers are not perturbed by
shared load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from caac.cost.accounting import CostMeter
from caac.utils.logging import get_logger

__all__ = ["LoadTestResult", "run_load_test", "poisson_arrivals"]

log = get_logger(__name__)


def poisson_arrivals(n: int, qps: float, seed: int = 0) -> np.ndarray:
    """Arrival times of ``n`` requests at rate ``qps`` (cumulative exponentials)."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / max(qps, 1e-9), size=n)
    return np.cumsum(gaps)


@dataclass
class LoadTestResult:
    """Per-QPS summary of a load test."""

    qps: float
    n_requests: int
    latency_p50: float
    latency_p95: float
    throughput: float
    mean_cost: float
    overhead_ratio: float
    controller_ratio: float
    accuracy: float
    peak_concurrent: int
    meta: dict = field(default_factory=dict)


def run_load_test(
    run_one,
    problems: list[dict],
    *,
    qps_levels=(1.0, 2.0, 4.0, 8.0),
    weights=None,
    seed: int = 0,
) -> list[LoadTestResult]:
    """Replay ``problems`` at several arrival rates.

    ``run_one(problem) -> (correct: bool, meter: CostMeter, wall_seconds: float)``
    is supplied by the caller so this function stays independent of which
    controller or baseline is under test.

    The simulation is sequential: request service times are measured for real,
    then queueing is derived from the arrival schedule. That captures the
    queue-delay effect that makes tail latency rise with load, without needing
    a full async harness.
    """
    from caac.types import CostWeights

    weights = weights or CostWeights(tokens=1.0)
    results: list[LoadTestResult] = []

    # Measure service time and cost once per problem; reuse across QPS levels.
    services, correctness, meters = [], [], []
    for prob in problems:
        t0 = time.perf_counter()
        correct, meter, wall = run_one(prob)
        elapsed = wall if wall is not None else (time.perf_counter() - t0)
        services.append(elapsed)
        correctness.append(bool(correct))
        meters.append(meter)

    total = CostMeter()
    for m in meters:
        total.merge(m)
    n = len(problems)

    for qps in qps_levels:
        arrivals = poisson_arrivals(n, qps, seed)
        finish = 0.0
        latencies, concurrent = [], 0
        peak = 0
        for arrive, service in zip(arrivals, services):
            start = max(arrive, finish)
            finish = start + service
            latencies.append(finish - arrive)
            # Rough concurrency proxy: requests that arrived before this one ended.
            concurrent = int(np.sum((arrivals >= arrive) & (arrivals < finish)))
            peak = max(peak, concurrent)

        lat = np.array(latencies)
        makespan = float(finish) or 1.0
        results.append(
            LoadTestResult(
                qps=qps,
                n_requests=n,
                latency_p50=float(np.percentile(lat, 50)),
                latency_p95=float(np.percentile(lat, 95)),
                throughput=n / makespan,
                mean_cost=total.total().scalar(weights) / n,
                overhead_ratio=total.overhead_ratio(weights),
                controller_ratio=total.controller_ratio(weights),
                accuracy=sum(correctness) / n,
                peak_concurrent=peak,
            )
        )
        log.info("qps=%.1f  p95=%.3fs  throughput=%.2f/s", qps, results[-1].latency_p95,
                 results[-1].throughput)

    return results
