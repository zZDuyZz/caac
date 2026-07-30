# Phases, gates, and what each one costs

Ordering principle: run the experiments that can kill the project first, and
write the paper last. E1 and E0 can change what the paper is about, so they come
before the expensive work rather than after it.

| Phase | Work | GPU | Est. GPU-h | Gate to proceed |
|---|---|---|---|---|
| **P0** | Skeleton, VOC, cost accounting, mock, tests | no | 0 | tests green, demo runs |
| **P1** | vLLM/HF backend, PRM, reproduce baselines | yes | 150-250 | baselines match published numbers within tolerance |
| **P2** | **E1 calibration study** | yes | 80-120 | **at least one signal ECE < 0.10 at >= 1.5B** |
| **P3** | Compute-tree collection, **E0 oracle** | yes | 450-800 | **oracle gap >= 20% cost at iso-accuracy** |
| **P4** | Train estimators, CAAC-Analytic, **E2 Pareto** | yes | 250-400 | dominates at least one strong adaptive baseline |
| **P5** | **E3** allocation structure, ablations, CAAC-Learned | yes | in P4 | source of the gain identified |
| **P6** | **E5** transfer + adaptation curve | yes | 100-150 | benefit retained after adapter-only re-fit |
| **P7** | **E4** deployment under load (**dedicated GPU**) | yes | 60-100 | net saving positive in wall-clock |
| **P8** | Write paper, release artifact | no | 0 | draft complete |

Total: roughly 1,200-1,900 GPU-hours, which suits 2-4 GPUs of 48-80GB over a
year. P7 needs exclusive access to a GPU, since latency measured on a shared
card is not a measurement.

## If a gate fails

**P2 fails (no usable signal at any scale).** The VOC rule has nothing
trustworthy to compute with, so the controller cannot work as designed. Pivot:
make the calibration study the contribution. It is a real finding about the
lower bound of adaptive computation for small models, and it is publishable.

**P3 fails (oracle gap too small).** There is not enough headroom for any policy
to matter, so beating a baseline would prove little. Rescope: narrow to the
*when* question with the risk-controlled stopping guarantee, which stands on
its own.

**P7 fails (token saving does not become wall-clock saving).** Report it. A
method that looks good per request and not under load is a finding about
adaptive reasoning in serving systems, and hiding it would misrepresent the
framework's central claim.

## Running each phase

```bash
python experiments/run_e1_calibration.py --backend vllm --model <model>
python experiments/run_e0_oracle.py --tree outputs/tree.json
python experiments/run_e2_pareto.py --backend vllm --model <model>
python experiments/run_e3_allocation.py --backend vllm --model <model>
python experiments/run_e5_transfer.py
python experiments/run_e6_profile.py
python experiments/run_e4_deployment.py --profile latency
```

Every runner works on CPU with `--backend mock` first, so the harness can be
validated before any GPU time is spent on it.

## Writing order in P8

Figures and tables first (the numbers are already fixed by then), then results,
method, related work, introduction, and the abstract last. Writing the
introduction before the results exist means rewriting it when they arrive.
