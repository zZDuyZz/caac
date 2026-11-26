"""Protocol lock: what is fixed BEFORE any audit of public or test data is run.

The lock (JSON) fixes alpha, delta, s, the split salt, the cost test, the allowed
candidate policies, the pair-selection rule and the pre-registered criteria. Its
hash goes into every report. A config used with --lock must stay inside it.
status "draft" = still editable (results NOT pre-registered); "frozen" = final.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from caac.pool_v1.config import Config


def load_lock(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") != "caac-lock":
        raise ValueError("not a caac-lock file")
    if d.get("status") not in ("draft", "frozen"):
        raise ValueError("lock status must be 'draft' or 'frozen'")
    return d


def lock_hash(path) -> str:
    d = load_lock(path)
    canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def allowed_cuts(pol: dict, cap: int) -> set:
    """Cut values a lock allows for a small model with generation cap `cap`: fractions of the
    cap (v1) and, from lock v2, absolute token budgets strictly below the cap."""
    out = {round(f * cap) for f in pol.get("cut_fraction_of_cap", [])}
    out |= {int(t) for t in pol.get("cut_tokens_absolute", []) if int(t) < cap}
    return out


def sensitivity_cells(L: dict) -> list[tuple[float, float]]:
    """(alpha, s) cells of the lock's pre-declared sensitivity grid, main cell excluded."""
    g = L.get("sensitivity") or {}
    main = (L["audit"]["alpha"], L["audit"]["s"])
    return [(al, s) for al in g.get("alpha", []) for s in g.get("s", []) if (al, s) != main]


def sensitivity_cells_relative(L: dict) -> list[tuple[float, float]]:
    """(eps, s) cells of the relative claim's sensitivity grid, main cell excluded."""
    g = L.get("sensitivity") or {}
    main = (L["audit"].get("eps"), L["audit"]["s"])
    return [(e, s) for e in g.get("eps", []) for s in g.get("s", []) if (e, s) != main]


def check_config_against_lock(cfg: Config, path, caps: dict | None = None,
                              sensitivity: bool = False) -> list[str]:
    """Return a list of mismatches (empty = the config is inside the lock).
    sensitivity=True also accepts (alpha, s) from the lock's sensitivity grid (reported
    beside the main result, never replacing it)."""
    L = load_lock(path)
    a, pol = L["audit"], L["policies"]
    bad = []
    keys = ("alpha", "delta", "s", "salt", "cost_test", "design")
    if sensitivity:
        g = L.get("sensitivity") or {}
        if cfg.alpha not in g.get("alpha", []) or cfg.s not in g.get("s", []):
            bad.append(f"(alpha, s) = ({cfg.alpha}, {cfg.s}) not in the lock's sensitivity grid")
        keys = ("delta", "salt", "cost_test", "design")
    for k in keys:
        want = a.get(k, "v2") if k == "design" else a[k]
        if getattr(cfg, k) != want:
            bad.append(f"{k}: config {getattr(cfg, k)!r} != lock {want!r}")
    claims = a.get("claims", ["absolute"])
    if "rel_test" in a:
        from caac.pool_v1.audit import REL_TEST
        if a["rel_test"] != REL_TEST:
            bad.append(f"rel_test: code {REL_TEST!r} != lock {a['rel_test']!r}")
    if cfg.claim not in claims:
        bad.append(f"claim {cfg.claim!r} not in lock claims {claims}")
    if cfg.claim == "relative":
        if sensitivity:
            if cfg.eps not in (L.get("sensitivity") or {}).get("eps", []):
                bad.append(f"eps {cfg.eps} not in the lock's sensitivity grid")
        elif cfg.eps != a.get("eps"):
            bad.append(f"eps: config {cfg.eps!r} != lock {a.get('eps')!r}")
    g = cfg.policies
    if not set(g.k) <= set(pol["k"]):
        bad.append(f"policies.k {g.k} not within lock {pol['k']}")
    if not set(float(t) for t in g.tau) <= set(float(t) for t in pol["tau"]):
        bad.append(f"policies.tau {g.tau} not within lock {pol['tau']}")
    if not set(g.on_unsure) <= set(pol["on_unsure"]):
        bad.append(f"policies.on_unsure {g.on_unsure} not within lock {pol['on_unsure']}")
    cuts = [c for c in g.cut if c is not None]
    if cuts:
        if caps is None or cfg.small not in caps:
            bad.append("policies.cut set but the small model's cap is unknown (meta.caps)")
        else:
            allowed = allowed_cuts(pol, caps[cfg.small])
            for c in cuts:
                if c not in allowed:
                    bad.append(f"policies.cut {c} not in lock-allowed {sorted(allowed)} "
                               f"(fractions {pol.get('cut_fraction_of_cap')} of cap {caps[cfg.small]}, "
                               f"absolute {pol.get('cut_tokens_absolute', [])})")
    if None not in g.cut and pol.get("require_no_cut", True):
        bad.append("policies.cut must include null (no early stop) as a candidate")
    return bad
