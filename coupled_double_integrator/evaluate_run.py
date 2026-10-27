"""
Held-out evaluation of trained networks on a FIXED set of initial states, with the safety filter each run was
trained with (DYS for the JFB runs, CVXPY-Layers for the CVXPY runs).  Read-only: loads weights, never trains.

Why fixed states: every network and every method is scored on exactly the same N_eval initial states
(drawn on the CPU from a fixed seed with the repo's initial distribution), so the numbers are comparable.

Example (repo root, in the server env):
    python coupled_double_integrator/evaluate_run.py \
        --run_dir coupled_double_integrator/runs/N3_d0.7_jfb_default_v2_seed0 \
                  coupled_double_integrator/runs/N3_d0.7_cvxpy_hard_seed0 \
        --weights best_safe_control_net.pth --dts 0.2,0.02

Reported per run and per control step dt (dt = 0.2 is the training step, dt = 0.02 is a near-continuous-time
check of the same network):  (L, G), min pair distance, min h (true d), fraction of rollouts that never come closer
than d, G_pos - floor, largest raw / filtered control, filter failures (CVXPY) or max DYS iterations.
Writes eval_fixed_<weights-stem>.json into each run folder and prints a markdown table.
"""
import os
import sys
import json
import math
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
from double_integrator_coupled import CoupledDoubleIntegrators, device
from utils import DYSProjector, ControlNet

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", nargs="+", required=True)
parser.add_argument("--weights", default="best_safe_control_net.pth", help="file inside each run folder")
parser.add_argument("--dts", default="0.2,0.02", help="comma-separated control steps")
parser.add_argument("--n_eval", type=int, default=256)
parser.add_argument("--eval_seed", type=int, default=12345)
args = parser.parse_args()
dts = [float(s) for s in args.dts.split(",")]


def fixed_initial_states(prob, n, seed):
    """Same distribution as prob.sample_initial_condition, but drawn on the CPU from a fixed seed
    (identical states on every machine)."""
    g = torch.Generator().manual_seed(seed)
    angles = -2 * math.pi * torch.arange(prob.n) / prob.n
    z0 = torch.zeros(n, prob.n, 4)
    z0[:, :, 0] = prob.center[0].item() + prob.radius * torch.cos(angles)
    z0[:, :, 1] = prob.center[1].item() + prob.radius * torch.sin(angles)
    z0[:, :, :2] += prob.z0_std * torch.randn(n, prob.n, 2, generator=g)
    return z0.reshape(n, -1).to(device)


def build_projector(cfg, prob):
    kind = cfg.get("projector", "dys")
    if kind == "dys":
        return DYSProjector(alpha=cfg["dys_alpha"], grad_mode="jfb").to(device), kind
    from cvxpy_projector import CvxpyProjector
    return CvxpyProjector(prob.control_dim, prob.n_pair, slack=(kind == "cvxpy_slack"),
                          slack_penalty=cfg.get("slack_penalty", 1e4), eps=cfg.get("scs_eps", 1e-3),
                          max_iters=cfg.get("scs_max_iters", 1000)).to(device), kind


rows = []
for run_dir in args.run_dir:
    cfg = json.load(open(os.path.join(run_dir, "args.json")))
    wpath = os.path.join(run_dir, args.weights)
    if not os.path.exists(wpath):
        print(f"[skip] {wpath} does not exist", flush=True)
        continue
    prob = CoupledDoubleIntegrators(n_agent=cfg["n_agent"], d=cfg["d"], gamma=cfg["gamma"], z0_std=cfg["z0_std"],
                                    vel_weight=cfg["vel_weight"], margin=cfg.get("margin", 0.0))
    net = ControlNet(input_dim=prob.state_dim + 1, hidden_dim=cfg["hidden_dim"], output_dim=prob.control_dim,
                     n_blocks=cfg["n_blocks"]).to(device)
    net.load_state_dict(torch.load(wpath, map_location=device))
    net.eval()
    proj, kind = build_projector(cfg, prob)
    z0 = fixed_initial_states(prob, args.n_eval, args.eval_seed)
    results = {}
    for dt in dts:
        nt = int(round(cfg["T"] / dt))
        n_fail0 = getattr(proj, "n_fail", 0)
        with torch.no_grad():
            total, run_c, term_c, info = prob.compute_loss(
                lambda z, t: net(z, t), z0, nt, dt, proj, cfg["alpha_running"], cfg["alpha_terminal_final"],
                max_iter=cfg["dys_max_iter"], tol=cfg["dys_tol"], return_traj=True, track_sigma=False)
            traj = info["traj"]                                              # (B, 4N, nt+1)
            B = traj.shape[0]
            pos = traj.reshape(B, prob.n, 4, -1)[:, :, :2]                   # (B, N, 2, nt+1)
            dist = (pos[:, prob.pair_i] - pos[:, prob.pair_j]).norm(dim=2)   # (B, c, nt+1)
            min_dist = dist.amin(dim=(1, 2))                                 # per rollout
            pm = prob.packing_metrics(info["z_final"])
        results[f"{dt:g}"] = dict(
            running_cost=float(run_c), terminal_cost=float(term_c),
            min_pair_distance=float(min_dist.min()), median_min_pair_distance=float(min_dist.median()),
            min_h_true=float(info["min_h_true"].min()),
            safe_rollout_fraction=float((min_dist >= prob.d - 1e-9).float().mean()),
            pos_cost_minus_floor=pm.get("pos_cost_minus_floor"), side_error_max=pm.get("side_error_max"),
            dys_iters_max=int(info["iters"].max()) if kind == "dys" else None,
            filter_failures=(getattr(proj, "n_fail", 0) - n_fail0) if kind != "dys" else None,
            filter_residual_max=float(info["resid"].max()))
        print(f"{os.path.basename(run_dir.rstrip('/\\\\'))} {args.weights} dt={dt:g}: "
              f"L={results[f'{dt:g}']['running_cost']:.3f} G={results[f'{dt:g}']['terminal_cost']:.4f} "
              f"min_dist={results[f'{dt:g}']['min_pair_distance']:.3f} "
              f"safe_frac={results[f'{dt:g}']['safe_rollout_fraction']:.3f}", flush=True)
    out = dict(run=run_dir, weights=args.weights, filter=kind, n_eval=args.n_eval, eval_seed=args.eval_seed,
               d=prob.d, margin=prob.margin, results=results)
    stem = os.path.splitext(args.weights)[0]
    with open(os.path.join(run_dir, f"eval_fixed_{stem}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    rows.append((os.path.basename(run_dir.rstrip("/\\")), kind, results))

print("\n| run | filter | dt | (L, G) | min pair dist. | min h (true d) | safe rollouts | G_pos - floor | filter fail / DYS iters |")
print("|---|---|---|---|---|---|---|---|---|")
for name, kind, res in rows:
    for dt, r in res.items():
        extra = r["filter_failures"] if kind != "dys" else r["dys_iters_max"]
        gp = r["pos_cost_minus_floor"]
        print(f"| {name} | {kind} | {dt} | ({r['running_cost']:.3g}, {r['terminal_cost']:.3g}) | {r['min_pair_distance']:.3f} | "
              f"{r['min_h_true']:.3f} | {r['safe_rollout_fraction']:.3f} | {gp if gp is None else format(gp, '+.3f')} | {extra} |")
