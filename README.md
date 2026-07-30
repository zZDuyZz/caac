# CAAC — Cost-Aware Adaptive Computation for Efficient LLM Reasoning

Deciding **when** and **where** to spend inference-time compute under a limited
computational budget, by comparing the *value of computation* against its cost.

A controller wraps any open-weight reasoning model and, at each step, chooses
among `{STOP, CONTINUE, VERIFY, BRANCH}` — spending compute only where the
expected reduction in answer uncertainty justifies its cost.

> This is the reference implementation skeleton. The full decision layer runs
> on CPU with a mock backend; GPU is only needed to run real models (phase P1+).

---

## Install

```bash
git clone <repo> && cd caac
pip install -e .            # CPU-only core (numpy/scipy/sklearn)
pip install -e ".[gpu]"     # adds torch/transformers/vllm for phases P1+
pip install -e ".[dev]"     # adds pytest
```

## Quickstart (no GPU)

```python
from caac import CAAC

controller = CAAC.from_pretrained("mock-model", backend="mock")
answer = controller.solve("What is 17 * 23?", budget=2048)

print(controller.last_meter.summary(controller.weights))   # cost breakdown
print([d.action.value for d in controller.last_decisions])  # the allocation trace
```

The mock backend simulates an easy and a hard query differently, so the
adaptive behaviour is visible without a GPU:

```
easy   answer=7    cost=192   steps=4   {'verify': 2, 'continue': 1, 'stop': 1}
hard   answer=42   cost=320   steps=5   {'verify': 2, 'continue': 3}
```

The easy query stops early; the hard one keeps spending. That difference is the
whole point of the framework, and it is produced by the VOC rule, not a
hand-tuned threshold.

Or from the terminal:

```bash
caac solve "2 + 3 = ?" --budget 1024
caac evaluate --n 20
```

Swap `backend="mock"` → `backend="vllm"` (with a config in `configs/models/`)
to run a real model once you have a GPU. **No decision-layer code changes.**

## Run the oracle bound (E0, CPU)

```bash
python experiments/run_e0_oracle.py --n 20
```

## Architecture

Three layers, separated by how model-dependent they are:

```
Serving engine (vLLM / SGLang)         ← generates tokens, gives logprobs
        │  logprob, hidden state (free during decoding)
        ▼
Scheduler        (src/caac/scheduler)  ← budget, decision spacing, flags
        ▼
Decision: VOC    (src/caac/core)       ← argmax VOC + lazy eval; no params
        ▼
Estimators                             ← belief (adapter, per-model)
  adapter  (src/caac/adapters)            gain, cost (policy, shared)
  policy   (src/caac/policy)
```

Only the **adapter** depends on the model. The **policy** works on the
calibrated belief and is shared across models — so moving to a new model means
re-fitting a small adapter (`controller.calibrate(...)`), not retraining.

**Feature tiering** is a hard constraint (`test_feature_tiering.py`): the shared
policy may read only model-invariant features (tier-0/1); model-specific signals
(tier-2) go to the adapter. This is what keeps the policy transferable.

**Overhead-honest accounting** (`src/caac/cost`): every unit of compute is
attributed to a source (reasoning / verifier / branch / controller), and the
controller's own cost is always counted — no hidden overhead.

## Phase map

| Phase | What | GPU? | Status in this repo |
|---|---|---|---|
| **P0** | Core, VOC, cost, mock, tests, oracle, API/CLI | no | complete |
| **P1** | vLLM/HF backend, PRM, baseline reproduction | yes | code ready, needs GPU |
| **P2** | E1 calibration study (gate) | yes | runner ready |
| **P3** | Compute-tree collection + E0 (gate) | yes | runner ready |
| **P4** | Estimator training, E2 Pareto sweep | yes | runner ready |
| **P5** | E3 allocation structure, ablations | yes | runner ready |
| **P6** | E5 transfer / adaptation curve | yes | runner ready |
| **P7** | E4 deployment under load (dedicated GPU) | yes | runner ready |
| **P8** | Paper, artifact release | no | `reports/` templates |

Every runner accepts `--backend mock` and works on CPU, so the harness is
validated before any GPU time is spent. Swapping to `--backend vllm` changes no
decision-layer code.

See `docs/PHASES.md` for the gate criteria and what to do when one fails.

## Experiments

```bash
python experiments/run_e0_oracle.py                  # oracle bound (CPU)
python experiments/run_e1_calibration.py             # E1, the first gate
python experiments/run_e2_pareto.py                  # E2, the main result
python experiments/run_e3_allocation.py              # E3, tests H2
python experiments/run_e5_transfer.py                # E5, multi-model claim
python experiments/run_e6_profile.py                 # E6, deployment profiles
python experiments/run_e4_deployment.py              # E4, under load
```

A note on what the mock shows: with the placeholder `HeuristicGain`, CAAC does
*not* beat every baseline on the mock, and the E2 output says so. That is
intended. The harness is not tuned to flatter the method; the gain model has to
be trained on a real compute tree (P4) before the comparison means anything.

## Reports

One report per phase, from `reports/TEMPLATE.md`. Gate criteria are fixed in
advance in `docs/PHASES.md` so they cannot drift once results are visible, and
by P8 most of the experimental section already exists in these reports.

## Tests

```bash
pytest            # ~30 tests, all CPU, all green
```

## Layout

```
src/caac/
  types.py            core domain types (pure numpy)
  core/               VOC rule, controller loop, budget (universal)
  adapters/           per-model: calibration + correctness → belief
  policy/             shared: gain, cost, VOC policy, baselines
  signals/            uncertainty signals + tiered features
  cost/               overhead-honest accounting + profiles
  scheduler/          overhead flags (spacing, lazy, parallel)
  backends/           mock (CPU) | hf | vllm
  verifier/           mock / self / prm
  baselines/          published-baseline reproductions (DEER, DeepConf, s1, SC@k)
  data/               compute tree + oracle, collection, benchmark loaders
  eval/               calibration, pareto, answer matching, deployment
  api.py              CAAC facade   ·   cli.py  CLI
experiments/          E0 (CPU) + E1–E6 runners
configs/              models / profiles / experiments
tests/                CPU test suite
docs/                 architecture, quickstart, reproducing
```
