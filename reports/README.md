# Phase reports

One report per phase, written when the phase closes, using `TEMPLATE.md`.

The reports exist for three reasons:

1. **The gates are binding.** Each phase has a criterion fixed in advance
   (`docs/PHASES.md`). Writing the outcome down stops the bar from drifting once
   the numbers are visible.
2. **Negative results are results.** If E1 shows the signal is unusable at small
   scale, that is a finding worth publishing, and the report is where it is first
   recorded rather than quietly absorbed.
3. **The paper is assembled from these.** By P8, most of the experimental
   section already exists in the reports.

| Report | Phase | Gate |
|---|---|---|
| `P0_skeleton.md` | code skeleton, CPU | tests green, demo runs |
| `P1_baselines.md` | GPU backend + baseline reproduction | baselines match published numbers |
| `P2_calibration.md` | E1 | at least one signal ECE < 0.10 at >= 1.5B |
| `P3_tree_oracle.md` | compute tree + E0 | oracle gap >= 20% cost at iso-accuracy |
| `P4_pareto.md` | estimators + E2 | dominates at least one strong adaptive baseline |
| `P5_analysis.md` | E3 + ablations | source of the gain identified |
| `P6_transfer.md` | E5 | benefit retained after adapter-only re-fit |
| `P7_deployment.md` | E4 | net saving positive in wall-clock under load |
| `P8_paper.md` | writing | draft complete, artifact released |
