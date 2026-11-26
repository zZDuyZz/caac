# Outcome of the pre-registered simulation rules (read 26/11 ~17:45, after both runs finished)
Runs: sim_rel_1000 (rev. 1, 969 s) and sim_rel2_1000 (rev. 2, 950 s); 11 scenarios x n in {150..2400} x 1000 reps.
(d) vectorised betting == _betting_p: max diff 1.1e-16.  PASS
(a) validity: largest wrong-verdict rate of every variant (bonf, +fs2; bet, harm, mix): 0.021 <= 0.10.  PASS
Choice of the GO risk leg (rev. 2 rule: mean GO power over the claim-true cells, Bonferroni):
  bet 0.0407, harm 0.0000, mix 0.0275  ->  bet  (REL_TEST = "bet")
  Note: harm is 0 because every scenario written before the run has a harm rate P(D = +1) above eps;
  the exact harm test can only help when the harm rate is below eps. Reported as is.
(b) M1 two-direction fixed sequence: fs2 - bonf in S9 = (0, 0, -0.004, -0.004, +0.008) for bet;
  criterion needs > +0.05 in 3 of 5 n  ->  NOT ADOPTED (negative result, reported).
(c) relative claim: (a) holds for bonf in S1-S11 -> PROPOSED for lock v3 (the user approved the relative
  claim on 26/11 16:54, eps = 0.02 main).
Power is LOW at eps = 0.02 when small and big disagree often on the questions the small model serves
(S1: 0.315 at n = 2400): expect INCONCLUSIVE with an evidence gap on public pools, not GO.
