# Phase P2 Report — E1 Calibration Study (Gate 1)

**Date:** 10/09/2026
**GPU usage:** ~1 GPU-hour
**Estimated cost:** ~$1

---

## 1. Objective

P2 is the **first formal go/no-go gate** of the project. The objective is to determine whether a small model (1.5B) provides sufficiently reliable uncertainty signals to support the downstream **VOC rule and CAAC controller**.

### Pre-registered gate

At least **one of four signals must achieve ECE < 0.10 after post-hoc calibration** at model scale ≥1.5B.

If no signal passes, the project would pivot toward a **calibration-for-control study** rather than proceeding with the full controller.

---

## 2. Experimental Setup

| Component    | Configuration                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| Model        | `Qwen/Qwen2.5-1.5B-Instruct`                                                                                  |
| Backend      | `vllm==0.7.3`                                                                                                 |
| Benchmark    | GSM8K test split                                                                                              |
| Signals      | Mean token confidence, min token confidence, sequence entropy, answer stability                               |
| Calibrators  | Identity, temperature scaling, isotonic regression                                                            |
| Initial run  | `n=25`                                                                                                        |
| Official run | `n=100`                                                                                                       |
| Main command | `python experiments/run_e1_calibration.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --n-problems 100` |

The experiment was intentionally run twice. The initial `n=25` run revealed suspiciously perfect isotonic calibration (`ECE=0.000`), motivating a larger `n=100` run before making the gate decision.

---

## 3. Results

### 3.1 Initial run — n=25

All four signals obtained `ECE=0.000` with isotonic regression.

This result was treated as **preliminary and unreliable**, because isotonic regression is non-parametric and can overfit small calibration sets.

The `n=25` results were therefore **not used for the final gate decision**.

### 3.2 Official run — n=100

| Signal                    | Calibrator      |        ECE |  Brier |   AURC |
| ------------------------- | --------------- | ---------: | -----: | -----: |
| mean_token_confidence     | identity        |     0.2115 | 0.2434 | 0.2165 |
| **mean_token_confidence** | **temperature** | **0.0640** | 0.1976 | 0.2165 |
| mean_token_confidence     | isotonic        |     0.0000 | 0.1884 | 0.2229 |
| min_token_confidence      | identity        |     0.4046 | 0.3851 | 0.2507 |
| min_token_confidence      | temperature     |     0.1544 | 0.2298 | 0.2507 |
| min_token_confidence      | isotonic        |     0.0000 | 0.1984 | 0.2489 |
| sequence_entropy          | identity        |     0.2554 | 0.2683 | 0.2125 |
| **sequence_entropy**      | **temperature** | **0.0544** | 0.1966 | 0.2125 |
| sequence_entropy          | isotonic        |     0.0000 | 0.1882 | 0.2207 |
| answer_stability          | identity        |     0.2446 | 0.2847 | 0.2571 |
| answer_stability          | temperature     |     0.1913 | 0.2464 | 0.2571 |
| answer_stability          | isotonic        |     0.0000 | 0.2053 | 0.2609 |

### Gate result

**PASS**

Two signals satisfy the pre-registered criterion:

* `mean_token_confidence` + temperature scaling: **ECE = 0.0640**
* `sequence_entropy` + temperature scaling: **ECE = 0.0544**

The gate decision is based on **temperature scaling only**; isotonic results are excluded from the gate interpretation because of the observed overfitting concern.

---

## 4. Gate Outcome

| Criterion                 | Result    |
| ------------------------- | --------- |
| ≥1 signal with ECE < 0.10 | **PASS**  |
| Model scale ≥1.5B         | **PASS**  |
| Signals passing           | **2 / 4** |
| Final gate                | **PASS**  |

**Verdict: PASS**

**Decision:** Proceed to **P3 — compute-tree collection + E0 oracle analysis**.

P3 is the second gate and the final major feasibility check before committing most of the project's GPU budget.

---

## 5. Key Observations

### 5.1 Calibration substantially improves selected signals

Raw signals were poorly calibrated:

* Mean token confidence: ECE **0.2115**
* Min token confidence: ECE **0.4046**
* Sequence entropy: ECE **0.2554**
* Answer stability: ECE **0.2446**

Temperature scaling reduced ECE below 0.10 for **2/4 signals**.

This supports the core CAAC assumption that uncertainty estimates should be **calibrated before being used by a cost-aware controller**, rather than directly treating raw confidence as belief.

### 5.2 Small-sample results can be misleading

`n=25` produced apparently excellent temperature-calibration results for signals that did not remain below the gate threshold at `n=100`.

In particular:

* `min_token_confidence`: ~0.055 → **0.1544**
* `answer_stability`: ~0.043 → **0.1913**

This confirms that the initial small-sample result was overly optimistic and justifies using `n=100` as the official gate measurement.

### 5.3 Isotonic regression shows a systematic overfitting warning

Isotonic regression produced **ECE = 0.0000 for all four signals** at both `n=25` and `n=100`.

This is not interpreted as perfect calibration. The result is consistent with overfitting from a non-parametric calibrator on the current sample size.

**Decision:** isotonic regression is not used to establish the gate. A proper train/calibration split and substantially larger sample size will be required before relying on it in later experiments.

### 5.4 Signal-family differences

At `n=100`, the two probability-distribution-based signals performed better after temperature scaling:

* Mean token confidence: **ECE = 0.0640**
* Sequence entropy: **ECE = 0.0544**

The other two signals remained above the gate:

* Min token confidence: **ECE = 0.1544**
* Answer stability: **ECE = 0.1913**

This motivates prioritizing **mean token confidence and sequence entropy** as candidate inputs for the correctness estimator in later phases.

---

## 6. Problems and Fixes

| Problem                                           | Root Cause                                          | Fix                                                          | Status                                       |
| ------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| All signals initially skipped; false gate failure | E1 still used the synthetic single-class benchmark  | Switch E1 to `load_benchmark('gsm8k', ...)`                  | Fixed                                        |
| Isotonic ECE = 0.000 at `n=25`                    | Non-parametric calibrator can overfit small samples | Re-run at `n=100`; exclude isotonic from gate interpretation | **Open — use proper split / larger n later** |
| `verifier_score` relies on MockVerifier           | Real PRM not yet tuned                              | Keep mock verifier; evaluate real PRM in P4                  | **Open — carried from P1**                   |

---

## 7. Cost & Overhead

| Run       |    Size |  Wall-clock | GPU usage |
| --------- | ------: | ----------: | --------: |
| Initial   |  `n=25` |     ~13 min |    ~0.2 h |
| Official  | `n=100` |     ~44 min |    ~0.7 h |
| **Total** |       — | **~57 min** |  **~1 h** |

**Estimated total cost:** ~$1 on an RTX 4090 instance.

E1 uses the calibration harness rather than the full `CostMeter` accounting used in E2, so the figures above are based primarily on observed wall-clock usage.

---

## 8. Artifacts

* [x] `outputs/e1_calibration.json` — `n=25`, preliminary
* [x] `outputs/e1_calibration.json` — `n=100`, official result
* [ ] Figures — deferred to P4/P5 when multiple model scales are available
* [ ] Reproduction commit hash — record after committing this report

---

## 9. Next Phase — P3

P3 will collect **compute trees** on Qwen2.5-1.5B and evaluate the **E0 oracle bound** using real execution traces.

### P3 gate

Target:

**Oracle gap ≥ 20% cost at iso-accuracy** relative to the strongest baseline measured in P1 (DeepConf / self-consistency).

### Signal priority

Based on P2, the primary candidate signals for the later correctness estimator will be:

1. `mean_token_confidence`
2. `sequence_entropy`

`min_token_confidence` and `answer_stability` will remain available for comparison but will not be treated as equally promising by default.

**P2 status: PASS → proceed to P3.**
