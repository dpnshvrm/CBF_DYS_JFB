# Coupled double integrators (N=3): results snapshot

Snapshot of the logs pulled at commit `e626958` (2026-10-27 13:22). Setting for all runs: N=3, d=0.7, gamma=1, T=10, dt=0.2, omega ramp 20 -> 500 (+5 every 10 epochs), seed 0, batch 32, Adam lr 1e-3, DYS (zeta 0.5, tol 5e-3, max 5000 iterations) or CVXPY (SCS eps 1e-3, 1000 iterations).

L = running cost, G = terminal cost (unweighted), as in the paper's Table 1; lower is better. The position part of G cannot go below 0.245 for any safe policy (PLAN.md, Sec. 6), so G values around 0.25-0.27 are near the optimum.

**Two kinds of numbers appear below; do not mix them.** (A) *Held-out evaluation* on 256 fresh initial states, from `eval.json` / `eval_best_safe.json`; only finished runs have it. (B) *Training-batch values* from `log.csv` (one batch of 32 per epoch, noisy). The CVXPY runs are still training, so they only have (B).

## Table 1 (A): held-out evaluation, paper style (L, G)

| Method | Network | (L, G) | min pair dist. (d = 0.7) | min h (true d) | G_pos - floor | status |
|---|---|---|---|---|---|---|
| DYS-JFB (ours), default schedule | best-safe (epoch 340) | (4.791, 0.262) | 0.708 | 0.011 | +0.016 | done |
| DYS-JFB (ours), default schedule | final (epoch 3000) | (35.86, 0.219) | 0.096 | -0.481 | -0.118 | done |
| DYS-JFB, omega capped at 300 (ablation) | best-safe (epoch 271) | (4.720, 0.267) | 0.708 | 0.011 | +0.014 | done |
| DYS-JFB, omega capped at 300 (ablation) | final (epoch 3000) | (102.98, 0.829) | 0.320 | -0.388 | +0.082 | done |
| CVXPY-Layers (IFT), hard QP | best-safe / final | pending | pending | pending | pending | training, epoch 1057 of 3000 |
| CVXPY-Layers (IFT + reg.), quadratic slack | best-safe / final | pending | pending | pending | pending | training, epoch 1006 of 3000 |

A negative `G_pos - floor` is impossible for a safe policy, so those rows violated the distance constraint.

## Table 1b (A'): fixed-state evaluation of the best-safe networks, each with its own filter

256 identical initial states (fixed seed 12345, drawn on the CPU) for every row, produced by `evaluate_run.py`. The CVXPY runs were still training, so their rows describe the best-safe network saved at evaluation time (the checkpoint is overwritten whenever a better safe epoch appears; its epoch was not recorded, but it is at least epoch 1003 for hard and 761 for slack, the best-safe epochs in the logs pulled at 13:22), not a finished run. dt = 0.2 is the training step; dt = 0.02 re-runs the same network with a ten times finer control step.

**dt = 0.2 (training step)**

| Method | best-safe epoch | (L, G) | min pair dist. (d = 0.7) | min h (true d) | safe rollouts | G_pos - floor | filter failures / max DYS iters | s / epoch (training) | training time to best-safe epoch |
|---|---|---|---|---|---|---|---|---|---|
| DYS-JFB (ours) | 340 | (4.79, 0.265) | 0.708 | 0.011 | 256 / 256 | +0.017 | 336 iters | 0.56 | 7 min (exact) |
| CVXPY (IFT), hard QP | >= 1003 | (4.36, 0.252) | 0.709 | 0.013 | 256 / 256 | +0.007 | 0 failures | 11.10 | >= 182 min (3.0 h) |
| CVXPY (IFT + reg.), quadratic slack | >= 761 | (4.55, 0.252) | 0.709 | 0.012 | 256 / 256 | +0.007 | 0 failures | 11.57 | >= 145 min (2.4 h) |

**dt = 0.02 (same networks, ten times finer control step)**

| Method | best-safe epoch | (L, G) | min pair dist. (d = 0.7) | min h (true d) | safe rollouts | G_pos - floor | filter failures / max DYS iters | s / epoch (training) | training time to best-safe epoch |
|---|---|---|---|---|---|---|---|---|---|
| DYS-JFB (ours) | 340 | (4.44, 0.265) | 0.708 | 0.011 | 256 / 256 | +0.018 | 336 iters | 0.56 | 7 min (exact) |
| CVXPY (IFT), hard QP | >= 1003 | (4.02, 0.251) | 0.708 | 0.011 | 256 / 256 | +0.006 | 0 failures | 11.10 | >= 182 min (3.0 h) |
| CVXPY (IFT + reg.), quadratic slack | >= 761 | (4.17, 0.252) | 0.708 | 0.011 | 256 / 256 | +0.007 | 0 failures | 11.57 | >= 145 min (2.4 h) |

Times come from `log.csv`: `s / epoch` is the mean seconds per training epoch after epoch 200 (shared server, so rough); `training time to best-safe epoch` is the cumulative logged time up to that epoch. For the CVXPY runs the epoch (and hence the time) is a lower bound, as explained above. Evaluation time is not included.

How to read it: all three best-safe networks stay at least 0.708 apart in every one of 256 rollouts, at both step sizes. CVXPY's networks reach a slightly lower G (0.252 against 0.265, i.e. closer to the packed optimum 0.245) and a slightly lower L than JFB's. Caveats: one seed; the JFB checkpoint was selected from noisy single-batch scores (the score fluctuates by about 40 between neighbouring epochs, against 0.1 for hard CVXPY), so a better JFB network may exist that the rule did not pick; the CVXPY networks come from unfinished runs.

## Table 2 (B): training-log values for all methods

Best-safe epoch = the epoch with the lowest `running + 500 * terminal` among epochs whose training rollouts never violated the true distance d (min h >= 0). 'Last 50' = mean over the final 50 logged epochs.

| Method | epochs done | best-safe epoch | (L, G) at best-safe epoch | last-50 mean (L, G) | worst min h in last 50 | violating epochs | CVXPY failures | s / epoch (after epoch 200) |
|---|---|---|---|---|---|---|---|---|
| DYS-JFB (ours), default schedule | 3000 | 340 | (4.722, 0.255) | (37.61, 0.228) | -0.485 | 1653 of 3000 | n/a | 0.56 |
| DYS-JFB, omega capped at 300 (ablation) | 3000 | 271 | (4.709, 0.254) | (103.48, 0.857) | -0.404 | 1087 of 3000 | n/a | 0.57 |
| CVXPY-Layers (IFT), hard QP | 1057 | 1003 | (4.334, 0.252) | (4.377, 0.252) | 0.012 | 0 of 1057 | 0 | 11.10 |
| CVXPY-Layers (IFT + reg.), quadratic slack | 1006 | 761 | (4.550, 0.252) | (7.768, 0.277) | 0.011 | 0 of 1006 | 0 | 11.57 |

## Reading the tables

* Best-safe scores (running + 500 G; training batches, one seed): DYS-JFB (ours), default schedule: 132.0; DYS-JFB, omega capped at 300 (ablation): 131.9; CVXPY-Layers (IFT), hard QP: 130.3; CVXPY-Layers (IFT + reg.), quadratic slack: 130.6. Gaps of about 1% are within the noise.
* The DYS-JFB final network is unsafe (violations in most epochs after about epoch 500). The best-safe checkpoints of the default-schedule JFB run and of both CVXPY runs are safe in all 256 fixed rollouts, also at dt = 0.02 (Table 1b; not checked for the omega-capped run). The CVXPY runs have no violation so far.
* Cost: about 0.5 s per epoch for JFB against 11-12 s for CVXPY, on a shared server (rough).
* Still to fill in: re-run `evaluate_run.py` on the CVXPY runs when they finish (best-safe and final networks), JFB seeds 1 and 2, and the final networks at dt = 0.02.

