"""CAAC-Config: what the user decides BEFORE the audit data is read.

A config is a small JSON file. Nothing in it is inferred from the audit data:

{
  "small": "model-id", "big": "model-id",          # roles; the pool itself has no roles
  "alpha": 0.10, "delta": 0.10, "s": 0.20,          # risk level, error level, required saving
  "salt": "caac-audit-v1",                          # fixed A/B split
  "audit_split": "test",                            # rows used for the audit (null = all rows)
  "cost": {
    "measure": "tokens",                            # tokens | usd | gpu_s
    "lambda": 3.0,                                  # price of one big output token in small tokens
    "price_out": {"small": 1.0, "big": 3.0},        # OR explicit per-token prices (any unit)
    "price_in":  {"small": 0.0, "big": 0.0},        # prompt tokens, charged once per model call
    "c_max": null,                                  # required for usd / gpu_s (a bound, not data)
    "also": [{"measure": "gpu_s", "c_max": 60.0}]   # extra measures shown in the report (R3)
  },
  "policies": {"k": [1, 2, 4], "tau": [0.5, 1.0], "cut": [null, 4096],
               "on_unsure": ["escalate"]},
  "cost_test": "betting",                           # betting (default) | bernstein
  "serving_mode": "batch"                           # recorded in the report only
}

Token counts of different model families come from different tokenizers, so a
token is not a unit of money. The price fields are where the user says what one
token of each model costs; ``lambda`` is the shortcut small=1, big=lambda.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MEASURES = ("tokens", "usd", "gpu_s")
ON_UNSURE = ("escalate", "abstain")
COST_TESTS = ("bernstein", "betting")


@dataclass
class CostProfile:
    measure: str = "tokens"
    price_out_small: float = 1.0
    price_out_big: float = 1.0
    price_in_small: float = 0.0
    price_in_big: float = 0.0
    c_max: float | None = None
    lam: float | None = None            # p_big / p_small (reported)
    also: list = field(default_factory=list)   # extra measures for R3: [{"measure", "c_max"}]


@dataclass
class PolicyGrid:
    k: list[int] = field(default_factory=lambda: [1])
    tau: list[float] = field(default_factory=lambda: [1.0])
    cut: list[int | None] = field(default_factory=lambda: [None])
    on_unsure: list[str] = field(default_factory=lambda: ["escalate"])


@dataclass
class Config:
    small: str
    big: str
    alpha: float = 0.10
    delta: float = 0.10
    s: float = 0.20
    salt: str = "caac-audit-v1"
    audit_split: str | None = None
    cost: CostProfile = field(default_factory=CostProfile)
    policies: PolicyGrid = field(default_factory=PolicyGrid)
    cost_test: str = "betting"
    serving_mode: str = "batch"

    def to_dict(self) -> dict:
        c, p = self.cost, self.policies
        return {"small": self.small, "big": self.big, "alpha": self.alpha, "delta": self.delta,
                "s": self.s, "salt": self.salt, "audit_split": self.audit_split,
                "cost": {"measure": c.measure, "lambda": c.lam,
                         "price_out": {"small": c.price_out_small, "big": c.price_out_big},
                         "price_in": {"small": c.price_in_small, "big": c.price_in_big},
                         "c_max": c.c_max, "also": list(c.also)},
                "policies": {"k": list(p.k), "tau": list(p.tau), "cut": list(p.cut),
                             "on_unsure": list(p.on_unsure)},
                "cost_test": self.cost_test, "serving_mode": self.serving_mode}


def _num(d, k, lo=None, hi=None, default=None):
    v = d.get(k, default)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"config: '{k}' must be a number, got {v!r}")
    if (lo is not None and v <= lo) or (hi is not None and v >= hi):
        raise ValueError(f"config: '{k}'={v} outside ({lo}, {hi})")
    return float(v)


def config_from_dict(d: dict) -> Config:
    for k in ("small", "big"):
        if not isinstance(d.get(k), str) or not d[k]:
            raise ValueError(f"config: '{k}' (model id) is required")
    if d["small"] == d["big"]:
        raise ValueError("config: small and big must be different models")
    alpha = _num(d, "alpha", 0, 1, 0.10)
    delta = _num(d, "delta", 0, 1, 0.10)
    s = _num(d, "s", -1, 1, 0.20)
    cd = d.get("cost", {}) or {}
    measure = cd.get("measure", "tokens")
    if measure not in MEASURES:
        raise ValueError(f"config: cost.measure must be one of {MEASURES}")
    po, pi = cd.get("price_out"), cd.get("price_in") or {}
    lam = cd.get("lambda")
    if po is not None and lam is not None:
        raise ValueError("config: give cost.lambda OR cost.price_out, not both")
    if po is None and lam is None and measure == "tokens":
        raise ValueError("config: token cost needs cost.lambda or cost.price_out (tokens of two "
                         "models are not the same unit; the tool does not assume a price)")
    if po is not None:
        ps, pb = _num(po, "small", 0), _num(po, "big", 0)
    elif lam is not None:
        ps, pb = 1.0, _num(cd, "lambda", 0)
    else:
        ps = pb = 1.0
    cost = CostProfile(measure=measure, price_out_small=ps, price_out_big=pb,
                       price_in_small=float(pi.get("small", 0.0)), price_in_big=float(pi.get("big", 0.0)),
                       c_max=cd.get("c_max"), lam=pb / ps)
    for a in cd.get("also", []) or []:
        if a.get("measure") not in MEASURES:
            raise ValueError(f"config: cost.also measure must be one of {MEASURES}")
        if a["measure"] in ("usd", "gpu_s") and not isinstance(a.get("c_max"), (int, float)):
            raise ValueError("config: cost.also entries for usd / gpu_s need c_max")
        cost.also.append({"measure": a["measure"], "c_max": a.get("c_max")})
    if cost.price_in_small < 0 or cost.price_in_big < 0:
        raise ValueError("config: prices must be >= 0")
    if measure in ("usd", "gpu_s") and cost.c_max is None:
        raise ValueError(f"config: measure '{measure}' needs cost.c_max (an upper bound on the "
                         "cost of one question, fixed in advance; the audit needs bounded costs)")
    pd = d.get("policies", {}) or {}
    grid = PolicyGrid(k=list(pd.get("k", [1])), tau=list(pd.get("tau", [1.0])),
                      cut=list(pd.get("cut", [None])), on_unsure=list(pd.get("on_unsure", ["escalate"])))
    if not grid.k or any((not isinstance(k, int)) or k < 1 for k in grid.k):
        raise ValueError("config: policies.k must be integers >= 1")
    if any(not (0.0 <= t <= 1.0) for t in grid.tau):
        raise ValueError("config: policies.tau must be in [0, 1]")
    if any(c is not None and (not isinstance(c, int) or c < 1) for c in grid.cut):
        raise ValueError("config: policies.cut must be null or integers >= 1")
    if any(u not in ON_UNSURE for u in grid.on_unsure):
        raise ValueError(f"config: policies.on_unsure must be in {ON_UNSURE}")
    ct = d.get("cost_test", "betting")
    if ct not in COST_TESTS:
        raise ValueError(f"config: cost_test must be one of {COST_TESTS}")
    unknown = set(d) - {"small", "big", "alpha", "delta", "s", "salt", "audit_split", "cost",
                        "policies", "cost_test", "serving_mode", "notes"}
    if unknown:
        raise ValueError(f"config: unknown key(s) {sorted(unknown)}")
    return Config(small=d["small"], big=d["big"], alpha=alpha, delta=delta, s=s,
                  salt=str(d.get("salt", "caac-audit-v1")), audit_split=d.get("audit_split"),
                  cost=cost, policies=grid, cost_test=ct,
                  serving_mode=str(d.get("serving_mode", "batch")))


def load_config(path: str | Path) -> Config:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
