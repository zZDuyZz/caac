# #170 Baselines: CAAC vs the usual ways to decide (relative claim)

- `DECISION_bl_before_run.md` + `DECISION_bl_hash.txt`: methods, worlds, metrics, predictions, readings and
  limits, hashed (sha256 fa4c4228...) before any baseline was run.
- `bl_methods.py`: vectorised rules B1 naive, B2 CLT, B3 CLT + Bonferroni, B4 Hoeffding + Bonferroni,
  B5 LTT with Hoeffding-Bentkus p-values, B6 CRC, B7 hold-out, and CAAC (official v3 relative audit).
- `bl_ref.py`: independent plain-Python implementation (betting from the official `audit._betting_p`).
- `bl_check.py`: X1 (300 random draws, every method, vectorised == plain Python), X2 (HB p-values vs scipy),
  and vectorised CAAC == `relative/sim_rel.py` 'bonf' on 120 draws.
- `bl_w1.py`: synthetic world (S1-S11 of `sim_rel.py`, S12-S14 from the design), 1000 replicates per cell.
- `bl_w2.py`: plasmode world on the 2,807 M4 configurations (each pool = population, i.i.d. draws with
  replacement, n in {250, 500, 1000}, 200 replicates), plus W3 (all methods on full data) and X0 (vectorised
  CAAC on full data reproduces the official map verdict and chosen policy for every configuration).
- `bl_analyze.py`: summary (written before the results were read). Addition beyond the design, made before
  reading results and marked descriptive: per-configuration maximum false-GO rate and the number of
  configurations above delta.
- `bl_diag_legs.py`: POST-HOC diagnostic (not in the design): which CAAC leg fails when CLT + Bonferroni says GO
  and CAAC does not (n = 500).

Run (from anywhere; the data are the M4 inputs):

    python protocols/v3_decision/baselines/bl_check.py
    python protocols/v3_decision/baselines/bl_w1.py 1000 OUT/bl_w1_1000.json 2
    python protocols/v3_decision/baselines/bl_w2.py POOLS_M4 MAP_REL38.jsonl OUT 2
    python protocols/v3_decision/baselines/bl_analyze.py OUT/bl_w1_1000.json OUT OUT/bl_summary.json

`results/`: outputs of the run of 27/11 (container, 2 cores). `bl_w2.jsonl.gz` is `bl_w2.jsonl` compressed
(byte-identical after decompression). Checks passed: X0 2,807/2,807; X1 0 mismatches; X2 max rel. diff 1.4e-12;
X3 20 random W1 cells rerun identical; `bl_analyze.py` on `results/` reproduces `bl_summary.json` byte for byte;
after moving the scripts into the repository (only the default repository path in `bl_methods.py` changed,
so the script hashes in `bl_w2_header.json` refer to the pre-move files), `bl_check.py` passes and 20
configurations rerun with `bl_w2.py` are identical.

Main results (relative claim, eps 0.02, s 0.20, delta 0.10):
- W1: largest false-GO rate: naive 1.00, CLT 0.686, CRC 0.998, CLT + Bonferroni 0.040, hold-out 0.075,
  Hoeffding / LTT-HB 0 (power 0 in every cell), CAAC 0.007. CLT + Bonferroni has higher power than CAAC in
  every cell.
- W2 (579 of 2,807 configurations have a feasible policy). Configurations with false-GO rate > delta at n = 250:
  naive 284, CRC 213, CLT 30, CLT + Bonferroni 1, hold-out 1, CAAC 0 (largest CAAC rate 0.035). Power at
  n = 250/500/1000: CLT + Bonferroni .45/.57/.67, CAAC .25/.39/.52, hold-out .43/.53/.62, LTT-HB .05/.10/.19.
  The configuration where CLT + Bonferroni fails (lrb_arcc_test, Llama-3.1-8B-Instruct -> Llama-3.1-8B-UltraMedical,
  tokens): the uncut policy saves 0.172 < 0.20 with heavy-tailed costs; CLT + Bonferroni says GO with it in 33%
  (n = 250) and 16% (n = 500) of studies; CAAC 0%.
- Next (approved 27/11, not started): raise CAAC power (exact fixed-n paired test for the quality leg, mixture
  betting for the cost leg), as amendment v3-9 with a design hashed first.
