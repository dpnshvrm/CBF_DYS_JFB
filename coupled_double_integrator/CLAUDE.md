# Coupled double-integrator experiment (N=3) -- hand-off for Claude Code

Read `PLAN.md` in this folder for the math (it uses the notation of the paper's tex). This file is the working context.

## Rules for this folder
* Use the conda env `implicitOC` (Python 3.9). Check `python -c "import sys; print(sys.executable)"` points into it. **Never install, upgrade or create packages/environments.**
* Only create or edit files inside `coupled_double_integrator/`. Do **not** modify anything else in the repo (`utils.py`, `train.py`, `CVXPyLayers/`, ...). Import from the repo's `utils.py` only.
* No git commands (no commit/checkout/reset). This folder is untracked on purpose.
* Be honest about what was run vs. not run; quote actual script output.

## Files
* `double_integrator_coupled.py` -- the problem: dynamics, one shared pairwise HOCBF per pair of agents (`BallCBF`, swappable via `AutogradCBF`), initial distribution (the repo's), costs, rollout with the DYS safety filter (`compute_loss`), diagnostics, plotting.
* `train_coupled.py` -- training (DYS-JFB; `--grad_mode ad` for the AD baseline). Writes `runs/<name>/{args.json,log.csv,eval.json,*.png,*.pth}`.
* `test_double_integrator_coupled.py` -- sanity tests (all 7 pass in implicitOC: constraint row matches finite differences of the barrier, autograd == closed form, rows touch two agents, DYS removes violations, initial states safe, equilateral packing floor, free flight).
* `diagnose_rollout.py` -- read-only diagnostic for a trained network (see below).

## The decided problem (PLAN.md Section 4b)
N=3 double integrators, state z^(i)=(p,v), start equally spaced on a circle of radius sqrt(8) about (1.5,1.5) (jitter 0.05, zero velocity), **common target** at (1.5,1.5), no obstacles. Pair barrier h_ij = |p_i-p_j|^2 - d^2 (d=0.7), relative degree 2, HOCBF h2 = h'' + 2*gamma*h' + gamma^2*h >= 0 (gamma=1), one row per pair -> A(z) in R^{3x6}. Control interval dt=0.2, T=10, RK4, controls held constant over each interval. Terminal weight omega ramps 20 -> 500. Because of the common target the safe optimum is an equilateral triangle of side d; the position part of G cannot go below d^2/2 = 0.245 (proved in PLAN.md Sec. 6).

## Current status / open problem: trained policies violate safety
A 3000-epoch run (d=0.7, gamma=1, dt=0.2, DYS tol 5e-3, max_iter 5000) shows: `min_h` reaches about -0.49 (= -d^2, i.e. two agents on top of each other) from epoch ~141 on; `min_psi1` (= h1) is negative from epoch 17 and reaches -5.3; terminal cost G ~ 0.22-0.25, i.e. at/below the safe floor 0.245 -> the optimizer exploits the violation; `active_frac` ~ 0.99 (the raw network output violates the constraints at almost every step); DYS hit max_iter in 8 epochs. The rollout plot shows pairwise distances falling to 0.2-0.35 in the last second.

What has been established (numpy replica of DYSProjector + the rows, no torch):
* Constraint rows and DYS update are correct for smooth controllers (min h = +0.013, constraints active at the end, exact QP and DYS agree).
* With large noisy nominal controls the state leaves h1>=0 even when the exact QP is satisfied at every step (zero-order-hold step of 0.2 s too coarse); smaller dt helps in that test but not monotonically.
* The QP is always feasible for unbounded controls (u_i = M(p_i - p_mean) works), including collinear starts.

Hypotheses, most likely first: (1) the 0.2 s zero-order-hold step is too coarse and the policy exploits it; (2) the raw network output is unpenalised and unbounded; (3) DYS is under-converged (stops when the step change < tol, which does not bound the constraint violation); (4) a bug (unlikely: tests pass).

## Next steps (in this order)
1. Run `python diagnose_rollout.py --run_dir runs/<the 3000-epoch run> --tight` (add `--double` if float32 is suspect). It rolls the same trained network out at dt = 0.2, 0.1, 0.05, 0.02 and prints min h / min h1, G, max controls, and how well DYS solves the QP. Report the table. If the 3000-epoch run folder is not on this machine, run a fresh `train_coupled.py` first (see log columns in `log.csv`).
2. Based on the result, propose (do not silently apply) one fix: shorter control interval; safety margin on d (the repo's obstacles use eps_safe=0.15); bound the raw network output (e.g. tanh scaling as the quadcopter branch of `train.py` does); tighter or relative DYS tolerance.
3. Only after safety is restored, look at the packing metrics in `eval.json` (`pos_cost_minus_floor`, `side_error_max`, `min_pair_distance`) and compare DYS-JFB vs `--grad_mode ad`.

## Style
Explain math plainly (equation, then what the code does, then how they line up). Flag decision points and ask before changing the design. Keep terminology stable; no internal jargon.

## Update 2026-10-27 (after reading both 3000-epoch runs, dt 0.2 and dt 0.1)
Finding: both runs are safe for epochs ~50-500 (min_h ~ +0.01, constraints active) and collapse during the omega ramp
(first min_h < -0.05 at omega ~ 345-365, epochs ~650-700); halving dt delays but does not prevent it. DYS hits max_iter
in most of the collapsed epochs. min_h = -0.49 = -d^2 means two agents sit on the same point, where the pair row of K is
zero and the QP is infeasible. best_control_net.pth was only saved once omega = 500, i.e. from the unsafe phase.
New (not yet run by Claude chat; run the tests first):
* `train_coupled.py --margin eps` (CBF enforces d+eps; "true" safety always uses d), `--alpha_terminal_final W` (omega cap, already existed),
  `best_safe_control_net.pth` = lowest running+W_final*terminal among epochs with min_h_true >= 0, evaluated in `eval_best_safe.json`.
  Log has two new columns at the end: min_h_true, proj_fail.
* `--projector {dys,cvxpy_hard,cvxpy_slack}`: CVXPY-Layers baselines in `cvxpy_projector.py` (SCS eps 1e-3, max_iters 1000,
  quadratic slack penalty 1e4 as in both repos; the paper's eq. (reg_qp) says linear -- unresolved which produced the table).
* `test_cvxpy_projector.py` compares CVXPY vs tightly-solved DYS on the same states.
Rules unchanged: no tanh bound (only the quadcopters use it in the repo), dt stays 0.2 for main results.
