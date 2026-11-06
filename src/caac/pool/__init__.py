"""Pool-based replay pipeline (PLAN_V2 v2.1).

Generate ONE sample pool on GPU (build_pool.py), then run every experiment --
self-consistency, cascades, PriceCheck-style serve/abstain, the CAAC 4-action
controller, LTT certification, and the Chow coverage bound -- offline on CPU
by replaying from the pool.

The older within-trajectory controller and oracle-tree code (removed
2026-11-06, kept at git tag pre-cleanup-2026-11-06) targeted an action set that
the P3 pilot showed has thin headroom; nothing here depends on it.
"""
