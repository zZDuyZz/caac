# Phase P<N> report — <title>

**Dates:** <start> to <end>   **GPU-hours used:** <n>   **Cost:** <$>

## 1. What this phase had to establish
<One paragraph. State the question this phase answers, and the gate or success
criterion agreed in advance. Copy it from docs/PHASES.md rather than restating
it loosely, so the bar is not moved after seeing results.>

## 2. What was run
| Item | Setting | Command |
|---|---|---|
| model | | |
| benchmark | | |
| experiment | | `python experiments/run_eX...` |

## 3. Results
<Tables and figures. Report the numbers that decide the gate first; supporting
numbers after.>

## 4. Gate outcome
- **Criterion:** <e.g. at least one signal with ECE < 0.10 at >= 1.5B>
- **Observed:** <number>
- **Verdict:** PASS / FAIL
- **Decision:** <continue to P<N+1> / rescope / pivot, per docs/PHASES.md>

## 5. What surprised us
<Anything that did not match the expectation registered in the proposal. Record
it even when it is inconvenient; this is where the interesting findings come
from, and it protects against quietly adjusting the story later.>

## 6. Problems and fixes
| Problem | Cause | Fix | Still open? |
|---|---|---|---|

## 7. Cost accounting
| Source | Tokens | Forward passes | Wall-clock |
|---|---|---|---|
| reasoning | | | |
| verifier | | | |
| branch | | | |
| controller | | | |
<Report net as well as gross. A saving that disappears once overhead is included
must be stated as such.>

## 8. Artifacts produced
- [ ] outputs/<file>.json
- [ ] compute tree at <path> (size, n nodes)
- [ ] figures: <list>
- [ ] config/commit hash to reproduce: <sha>

## 9. Next phase
<What P<N+1> starts with, and anything it must not assume.>
