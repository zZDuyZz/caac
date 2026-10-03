"""Pool-based replay pipeline (PLAN_V2 v2.1).

Generate ONE sample pool on GPU (build_pool.py), then run every experiment --
self-consistency, cascades, PriceCheck-style serve/abstain, the CAAC 4-action
controller, LTT certification, and the Chow coverage bound -- offline on CPU
by replaying from the pool.

This package is independent of the older oracle-tree code (data/compute_tree.py,
core/controller.py) which targeted the within-trajectory action set that P3
showed has thin headroom.
"""
