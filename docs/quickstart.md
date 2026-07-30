# Quickstart

## CPU (no GPU) — run the whole decision layer

```python
from caac import CAAC

c = CAAC.from_pretrained("mock-model", backend="mock", verifier="mock")
ans = c.solve("What is 17 * 23?", budget=2048)
print(c.last_meter.summary(c.weights))
```

## Switch deployment profile (no retraining)

```python
c.set_profile("latency")      # or "throughput", "token_only"
```

## Calibrate the adapter for a new model

```python
# dataset: iterable of (raw_signal, correct_label) pairs
data = [(0.9, 1), (0.2, 0), ...]
c.calibrate(data, n_examples=500)   # touches ONLY the adapter
```

## Oracle bound (E0)

```bash
python experiments/run_e0_oracle.py --n 20
```

## Moving to GPU (phase P1)

1. `pip install -e ".[gpu]"`
2. Fill `src/caac/backends/vllm.py` (interface is fixed).
3. Use a config in `configs/models/`, e.g. `qwen2.5-1.5b.yaml`.
4. `CAAC.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", backend="vllm")`.

The decision layer, cost accounting, calibration and Pareto tools are unchanged.
