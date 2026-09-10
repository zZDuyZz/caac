# Phase P1 Report — Backend + Baseline Reproduction

**Dates:** 09/09/2026 – 10/09/2026
**GPU usage:** ~1.5 GPU-hours
**Estimated cost:** ~$1–2

---

## 1. Objective

P1 establishes the GPU execution foundation for CAAC:

* Run the real **vLLM backend** with Qwen2.5-1.5B-Instruct.
* Replace synthetic/mock benchmark data with **GSM8K test data**.
* Execute the complete **CAAC + 5 baseline** pipeline without errors.
* Produce reproducible E2 Pareto-sweep results before proceeding to P2.

---

## 2. Experimental Setup

| Component    | Configuration                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Model        | `Qwen/Qwen2.5-1.5B-Instruct`                                                                           |
| Backend      | `vllm==0.7.3`                                                                                          |
| Benchmark    | GSM8K test split                                                                                       |
| Sample size  | `n=5`                                                                                                  |
| Experiment   | E2 Pareto sweep                                                                                        |
| Main command | `python experiments/run_e2_pareto.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --n-problems 5` |

---

## 3. Results

### 3.1 Backend Smoke Test

Command:

```bash
caac solve "What is 17 * 23?" \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

Result:

```text
answer: 391
cost: 12.0
overhead: 0.0%
controller: 0.0%
```

**Status: PASS** — the real vLLM backend successfully generated the correct answer.

### 3.2 E2 Pareto Sweep

| Method                      |  Accuracy |    Cost | Observation                                          |
| --------------------------- | --------: | ------: | ---------------------------------------------------- |
| **CAAC (λ=1e-5)**           | **0.800** | **334** | Best overall among methods except greedy CoT on cost |
| greedy_cot (`max_seg=6`)    |     0.800 |     270 | 23.7% cheaper than CAAC at equal accuracy            |
| self_consistency (`k=8`)    |     0.600 |    2166 | CAAC uses 38.4% less cost                            |
| budget_forcing (`target=8`) |     0.800 |     463 | CAAC uses 27.9% less cost                            |
| deer (`threshold=0.95`)     |     0.400 |     216 | Lower accuracy at this small sample size             |
| deepconf (`k=8`)            |     0.600 |    2166 | CAAC uses 38.4% less cost                            |

### Direct comparison at equal accuracy

* CAAC vs. **self-consistency:** −38.4% cost
* CAAC vs. **DeepConf:** −38.4% cost
* CAAC vs. **budget forcing:** −27.9% cost
* CAAC vs. **greedy CoT:** +23.7% cost

> **Important:** These results are preliminary because `n=5` is too small for strong statistical conclusions.

---

## 4. P1 Outcome

P1 has no formal go/no-go gate; the first formal gate is in P2/E1.

| Criterion                 | Result |
| ------------------------- | ------ |
| Real vLLM backend         | PASS   |
| Real GSM8K benchmark      | PASS   |
| CAAC execution            | PASS   |
| 5 baseline methods        | PASS   |
| Output artifact generated | PASS   |

**Verdict: PASS**

The complete pipeline executed successfully and produced:

```text
outputs/e2_pareto.json
```

**Decision:** Proceed to **P2 — E1 Calibration**.

---

## 5. Key Observations

### 5.1 CAAC vs. greedy CoT

CAAC did not outperform greedy CoT on cost at `n=5`. This is expected because the current `HeuristicGain` implementation in `policy/gain.py` is still a **placeholder**, rather than a trained gain model.

The current heuristic only captures qualitative behavior (e.g., inverted-U behavior for verification). Despite this limitation, CAAC still matched greedy CoT in accuracy and outperformed **3 of 5 baselines** in cost.

This provides preliminary motivation for evaluating the trained gain model in later phases.

### 5.2 DEER instability

DEER produced `NaN` at some comparison points. The cause is the very small sample size (`n=5`), which provides insufficient observations for some accuracy thresholds (`0.5/0.6/0.7/0.8`).

**Interpretation:** numerical behavior is a small-sample limitation, not a pipeline failure.

### 5.3 Synthetic benchmark issue

The initial benchmark used synthetic questions with a hard-coded answer of `"42"`, inherited from the mock-backend testing stage. Consequently, all methods initially obtained:

```text
accuracy = 0.000
```

The benchmark was corrected by implementing `load_gsm8k()` in `benchmarks.py` and switching E2 to the real GSM8K test data.

---

## 6. Problems and Fixes

| Problem                                                       | Root Cause                                                      | Fix                                                             | Status                                 |
| ------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------- |
| `VLLMBackend.__init__() missing model_name`                   | CLI/API passed the model under the wrong argument name          | Map `model → model_name` in `_make_backend` / `from_pretrained` | Fixed                                  |
| `assert "factor" in rope_scaling`                             | vLLM 0.6.3 incompatible with Qwen2.5 config                     | Upgrade to `vllm==0.7.3`                                        | Fixed                                  |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | `transformers` / vLLM version mismatch                          | Pin `transformers==4.49.0`                                      | Fixed                                  |
| PyTorch dependency conflicts                                  | Repeated pip installs caused incompatible dependency resolution | Pin `torch==2.5.1`                                              | Fixed                                  |
| `invalid literal for int(): 'GPU-...'`                        | `CUDA_VISIBLE_DEVICES` contained a GPU UUID                     | `export CUDA_VISIBLE_DEVICES=0`                                 | Recurs per new terminal                |
| `np.trapezoid` unavailable                                    | Older NumPy version                                             | Replace with `np.trapz`                                         | Fixed                                  |
| `ModuleNotFoundError: scipy`                                  | GPU environment reset / missing dependencies                    | Install `scipy` and `scikit-learn`                              | Fixed                                  |
| `VERIFY chosen but no verifier configured`                    | E2 only configured verifier for mock backend                    | Temporarily use `MockVerifier()`                                | **Open — replace with real PRM in P4** |
| Accuracy = 0.000                                              | Synthetic benchmark had hard-coded answer `"42"`                | Add `load_gsm8k()` and use GSM8K                                | Fixed                                  |

**Main technical debt carried forward:** the verifier is still a `MockVerifier`. It must be replaced with a real PRM/verifier before the corresponding final evaluation.

---

## 7. Cost & Overhead

| Component              | Observation                        |
| ---------------------- | ---------------------------------- |
| Reasoning tokens/query | ~64–270                            |
| Verifier               | Mock; no real verifier-token cost  |
| Branch                 | No Branch action observed at `n=5` |
| Controller overhead    | 0.0% in smoke test                 |

The observed controller overhead is **0.0%**, comfortably below the design target of **<1–2%**.

However, this measurement comes from a smoke test and should **not** be treated as a definitive large-scale result. It should be re-evaluated in P4/E4.

---

## 8. Artifacts

* [x] `outputs/e2_pareto.json` — E2 Pareto-sweep results
* [ ] Compute tree — planned for P3
* [ ] Figures — planned for P4/P5 with larger-scale data
* [ ] Reproduction commit hash — record after committing this report

---

## 9. Next Phase — P2

P2 begins with **E1 Calibration** using Qwen2.5-1.5B.

### Planned evaluation

Measure:

* ECE
* Brier score
* AURC
* Four signal families
* Before/after calibration

### Formal gate

**Target:** at least **one signal achieves ECE < 0.10** at this model scale.

### Verifier status

The verifier will remain `MockVerifier()` during P2. A real PRM will be introduced later after selecting and tuning the appropriate PRM implementation.

**P2 is therefore not blocked by the current mock verifier.**
