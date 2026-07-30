# Reproducing the paper

Each experiment maps to a deliverable (figure/table). E0 runs on CPU today;
E1–E6 need a GPU (phase P1+) but call CPU-tested analysis code.

| Exp | Deliverable | Where | GPU |
|---|---|---|---|
| E0 | Oracle frontier (upper bound) | `experiments/run_e0_oracle.py` | ❌ |
| E1 | Calibration vs scale (RQ1 gate) | `experiments/run_e1_calibration.py` | ✅ |
| E2 | Accuracy–cost Pareto (main table) | `experiments/run_e2_pareto.py` | ✅ |
| E3 | VOC(Verify) vs belief (H2 curve) | `experiments/run_e3_allocation.py` | ✅ |
| E4 | Deployment under load (throughput/latency) | `experiments/run_e4_deployment.py` | ✅ |
| E5 | Transfer / adaptation learning curve | `experiments/run_e5_transfer.py` | ✅ |
| E6 | Deployment-profile sweep | `experiments/run_e6_profile.py` | ✅ |

## Go/no-go gates

- **E1**: at least one signal reaches ECE < 0.10 at ≥1.5B after calibration.
- **E0**: oracle beats the strongest baseline by ≥20% cost at iso-accuracy.

If either gate fails, rescope (see proposal §Risks) before spending GPU on P4+.
