# Making your own CAAC-Pool and running the audit

1. One row per generation in `rows.csv` (or `rows.jsonl[.gz]`): see `pool_v1_template.csv`.
   Required: `qid, model, draw, correct`. Each optional column unlocks more (levels L0-L4):
   `out_tokens` -> audit and early stop; >= 2 draws with `answer_key` -> SAMPLE MORE;
   `finish` -> truncation checks; `gpu_s` / `cost_usd` -> measured cost.
2. `meta.json` next to it: source, grader, answer equivalence (`math` | `exact` | `none`), caps.
3. `config.json`: which model is small / big, alpha, delta, s, price ratio, candidate policies.
   Fix it BEFORE looking at the audit data.

```
mkdir mypool && cp pool_v1_template.csv mypool/rows.csv && cp meta_example.json mypool/meta.json
caac validate mypool
caac audit mypool --config config_example.json --out report.json --md report.md
```
The template has only two questions, so the verdict is INCONCLUSIVE; the report gives the minimum
number of questions for any verdict (44 at alpha = delta = 0.1) and, once a policy qualifies, an
estimate of how many questions it needs.
