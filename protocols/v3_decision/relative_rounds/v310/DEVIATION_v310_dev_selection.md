# Deviation v3-10-D1: dev selection rule (27/11/2026, 14:2x, before any confirmation run)

Found after the dev analysis (v310_dev_summary.json), before confirmation data existed:
the pre-registered dev rule chose X1 (score +0.0328; gW2 +0.0810; gW1 -0.0153). The confirmation criterion C
requires W1 mean gain >= -0.01, which the dev rule did not check, so the rule could pick a variant that the
pre-registered confirmation would almost surely reject. This is a design error (author: Claude), not a data issue.

Change (approved by the author of the study, 27/11): the dev eligibility rule additionally requires
gW1 >= -0.01 (the same threshold as confirmation criterion C). Nothing else changes: candidates, scores,
tie rule, confirmation criteria A-D, seeds (bl2 / BL2), held-out pools.
Effect on dev: eligible becomes [H1, H2]; chosen = H1 (H2 within 0.005 and later in simplicity order).
Disclosure in the paper: the dev table for all six candidates, the original rule's choice (X1) and this change.
