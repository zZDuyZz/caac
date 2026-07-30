# Architecture

## Three layers by model-dependence

**Universal (`core/`)** — the VOC rule, the controller loop, budget management.
Pure math and bookkeeping; identical for every model. The VOC rule is:

    VOC(a | s) = E[V(s')] − U_stop(s) − λ·c(a, s)
    STOP                     if max_{a≠STOP} VOC ≤ τ      (the *when* question)
    argmax_{a≠STOP} VOC      otherwise                    (the *where* question)

**Per-model (`adapters/`)** — maps a specific model's raw signals to a
calibrated belief P(correct). The ONLY model-dependent component. Re-fitting an
adapter is how the framework moves to a new model.

**Shared (`policy/`)** — gain and cost estimators plus the VOC policy. They work
on the calibrated belief and tier-0/1 features, so they transfer across models.

## Why the split matters

If the belief is calibrated, "0.7" means the same thing on a 0.5B and a 70B
model. So a policy that reads only the belief (and model-invariant features)
does not need to know which model it is driving. All model dependence is
isolated in the adapter. This is the basis of the multi-model claim, and E5
measures how much of the benefit survives when only the adapter is re-fit.

## Feature tiering (hard constraint)

- tier-0 (universal): belief, step fraction, budget fraction, n_branches, n_verify
- tier-1 (self-normalising): entropy slope, answer stability
- tier-2 (model-specific): raw token confidence, raw entropy scale

`FeatureExtractor.policy_features` returns only tier-0/1 and refuses tier-2.
`test_feature_tiering.py` fails if this is ever violated.

## Overhead-honest accounting

`CostMeter` attributes every unit of compute to a source: reasoning, verifier,
branch, controller. `total()` sums all four; `overhead()` is everything but
reasoning; `controller_ratio()` reports the controller's share. Claims use the
net total, never the gross reasoning-only figure.

## Scheduler flags

Overhead optimisations live in `scheduler/`, separate from decision logic, so
each is a togglable flag measured independently in ablation:
- lazy evaluation — skip VOC for actions far from the margin
- adaptive spacing — widen decision spacing when belief is stable
  (calibration-gated: only safe once the E1 gate passes)
- CPU-parallel — hide controller cost behind generation (needs P7 to measure)
