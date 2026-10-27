"""
Train a centralized feedback controller for N coupled double integrators (shared pairwise HOCBF,
common target) with the DYS-JFB safety layer.  Standalone: only imports utils.py from the repo.

Example (from the repo root, in the `implicitOC` conda env):
    python coupled_double_integrator/train_coupled.py --n_agent 3 --d 0.7 --epochs 3000 --lr_decay 800 --plot_every 200
    # omega capped at 300, safety margin 0.05:
    python coupled_double_integrator/train_coupled.py --alpha_terminal_final 300 --margin 0.05
    # CVXPY-Layers baselines with the SAME network / loss / schedule / seed:
    python coupled_double_integrator/train_coupled.py --projector cvxpy_hard
    python coupled_double_integrator/train_coupled.py --projector cvxpy_slack
"""
import os
import sys
import time
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
from double_integrator_coupled import CoupledDoubleIntegrators, plot_rollout, device
from utils import DYSProjector, ControlNet  # repo's utils.py (path added by double_integrator_coupled)

parser = argparse.ArgumentParser()
parser.add_argument("--n_agent", type=int, default=3)
parser.add_argument("--d", type=float, default=0.7, help="safety distance between every pair")
parser.add_argument("--gamma", type=float, default=1.0, help="HOCBF class-K slope")
parser.add_argument("--z0_std", type=float, default=5e-2)
parser.add_argument("--margin", type=float, default=0.0,
                    help="safety margin eps: the CBF enforces distance >= d + eps (repo obstacles use 0.1-0.15); reports use d")
parser.add_argument("--projector", choices=["dys", "cvxpy_hard", "cvxpy_slack"], default="dys",
                    help="safety filter: DYS (ours) or CVXPY-Layers hard QP / quadratic-slack QP (baselines)")
parser.add_argument("--slack_penalty", type=float, default=1e4, help="lambda in lambda*||s||^2 (cvxpy_slack)")
parser.add_argument("--scs_eps", type=float, default=1e-3)
parser.add_argument("--scs_max_iters", type=int, default=1000)
parser.add_argument("--vel_weight", type=float, default=1.0, help="w in running cost 0.5|u|^2 + w|v|^2 (repo code: 1)")
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay", type=int, default=800, help="halve lr every this many epochs")
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--T", type=float, default=10.0)
parser.add_argument("--dt", type=float, default=0.2)
parser.add_argument("--alpha_running", type=float, default=1.0)
parser.add_argument("--alpha_terminal_start", type=float, default=20.0)
parser.add_argument("--alpha_terminal_final", type=float, default=500.0)
parser.add_argument("--ramp_step", type=float, default=5.0)
parser.add_argument("--ramp_every", type=int, default=10, help="epochs between omega increases")
parser.add_argument("--dys_alpha", type=float, default=0.5, help="DYS step size (zeta)")
parser.add_argument("--dys_max_iter", type=int, default=5000)
parser.add_argument("--dys_tol", type=float, default=5e-3)
parser.add_argument("--grad_mode", choices=["jfb", "ad"], default="jfb")
parser.add_argument("--load_path", type=str, default=None, help="start from these network weights")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--plot_every", type=int, default=0, help="save a rollout plot every k epochs (0: only at the end)")
parser.add_argument("--eval_batch", type=int, default=256)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
_mode = args.grad_mode if args.projector == "dys" else args.projector
_tag = (f"_m{args.margin:g}" if args.margin > 0 else "") + (f"_w{args.alpha_terminal_final:g}" if args.alpha_terminal_final != 500.0 else "")
out_dir = args.out_dir or os.path.join(_HERE, "runs", f"N{args.n_agent}_d{args.d}_{_mode}{_tag}_seed{args.seed}")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "args.json"), "w") as fh:
    json.dump(vars(args), fh, indent=2)

prob = CoupledDoubleIntegrators(n_agent=args.n_agent, d=args.d, gamma=args.gamma, z0_std=args.z0_std,
                                vel_weight=args.vel_weight, margin=args.margin)
nt = int(round(args.T / args.dt))
net = ControlNet(input_dim=prob.state_dim + 1, hidden_dim=args.hidden_dim,
                 output_dim=prob.control_dim, n_blocks=args.n_blocks).to(device)
if args.load_path:
    net.load_state_dict(torch.load(args.load_path, map_location=device))
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.projector == "dys":
    proj = DYSProjector(alpha=args.dys_alpha, grad_mode=args.grad_mode).to(device)
else:
    from cvxpy_projector import CvxpyProjector
    proj = CvxpyProjector(prob.control_dim, prob.n_pair, slack=(args.projector == "cvxpy_slack"),
                          slack_penalty=args.slack_penalty, eps=args.scs_eps, max_iters=args.scs_max_iters).to(device)


def u_fn(z, t):
    return net(z, t)


print(f"device={device}  N={prob.n}  pairs c={prob.n_pair}  state dim={prob.state_dim}  control dim={prob.control_dim}"
      f"  net params={sum(p.numel() for p in net.parameters())}  steps={nt}", flush=True)

# Plot initial rollout (epoch 0, untrained network)
if args.plot_every > 0:
    with torch.no_grad():
        z0_init = prob.sample_initial_condition(args.batch_size)
        _, _, _, info_init = prob.compute_loss(u_fn, z0_init, nt, args.dt, proj, args.alpha_running, args.alpha_terminal_start,
                                               max_iter=args.dys_max_iter, tol=args.dys_tol, return_traj=True)
        plot_rollout(prob, info_init["traj"], os.path.join(out_dir, "rollout_epoch0.png"), dt=args.dt)

alpha_terminal = args.alpha_terminal_start
best_loss = float("inf")
best_safe_score = float("inf")      # lowest (running + omega_final * terminal) among epochs with min_h_true >= 0
best_safe_epoch = None
log_path = os.path.join(out_dir, "log.csv")
with open(log_path, "w") as fh:
    fh.write("epoch,total,running,terminal,alpha_terminal,active_frac,min_h,min_psi1,dys_iters_max,dys_res_max,sigma_min_active,time_s,min_h_true,proj_fail\n")

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    optimizer.zero_grad()
    z0 = prob.sample_initial_condition(args.batch_size)
    total, run_c, term_c, info = prob.compute_loss(
        u_fn, z0, nt, args.dt, proj, args.alpha_running, alpha_terminal,
        max_iter=args.dys_max_iter, tol=args.dys_tol, return_traj=(args.plot_every > 0 and epoch % args.plot_every == 0))
    total.backward()
    optimizer.step()
    dt_epoch = time.time() - t0

    row = (epoch, total.item(), run_c.item(), term_c.item(), alpha_terminal,
           float(info["active_frac"].mean()), float(info["min_h"].min()), float(info["min_psi1"].min()),
           int(info["iters"].max()), float(info["resid"].max()), float(info["sigma_min_active"].min()), dt_epoch,
           float(info["min_h_true"].min()), int(getattr(proj, "n_fail", 0)))
    with open(log_path, "a") as fh:
        fh.write(",".join(str(v) for v in row) + "\n")
    print(f"epoch {epoch:5d} total={row[1]:.4e} L={row[2]:.4e} G={row[3]:.4e} w={alpha_terminal:.0f} "
          f"active={row[5]:.2f} min_h={row[6]:.3e} min_psi1={row[7]:.3e} iters={row[8]:4d} res={row[9]:.1e} "
          f"sig_act={row[10]:.2e} h_true={row[12]:.3e} fail={row[13]} t={dt_epoch:.2f}s", flush=True)

    if info["traj"] is not None:
        plot_rollout(prob, info["traj"], os.path.join(out_dir, f"rollout_epoch{epoch}.png"), dt=args.dt)

    if epoch % args.ramp_every == 0 and alpha_terminal < args.alpha_terminal_final:
        alpha_terminal = min(alpha_terminal + args.ramp_step, args.alpha_terminal_final)
    if alpha_terminal >= args.alpha_terminal_final and total.item() < best_loss:
        best_loss = total.item()
        torch.save(net.state_dict(), os.path.join(out_dir, "best_control_net.pth"))
    # safety-aware checkpoint: only epochs whose training rollouts never violate the true distance d are eligible,
    # compared at the FINAL omega so that scores from different points of the ramp are comparable
    if row[12] >= 0:
        score = run_c.item() * args.alpha_running + args.alpha_terminal_final * term_c.item()
        if score < best_safe_score:
            best_safe_score, best_safe_epoch = score, epoch
            torch.save(net.state_dict(), os.path.join(out_dir, "best_safe_control_net.pth"))
    if epoch % args.lr_decay == 0:
        for g in optimizer.param_groups:
            g["lr"] *= 0.5
        print("new lr:", optimizer.param_groups[0]["lr"], flush=True)

torch.save(net.state_dict(), os.path.join(out_dir, "final_control_net.pth"))

# ---- evaluation on fresh initial states ------------------------------------------------------
def evaluate(tag):
    with torch.no_grad():
        z0 = prob.sample_initial_condition(args.eval_batch)
        total, run_c, term_c, info = prob.compute_loss(u_fn, z0, nt, args.dt, proj, args.alpha_running, alpha_terminal,
                                                       max_iter=args.dys_max_iter, tol=args.dys_tol, return_traj=True)
        summary = dict(running_cost=run_c.item(), terminal_cost=term_c.item(),
                       min_h_over_rollouts=float(info["min_h"].min()),
                       min_h_true_over_rollouts=float(info["min_h_true"].min()),
                       min_psi1_over_rollouts=float(info["min_psi1"].min()),
                       active_fraction=float(info["active_frac"].mean()),
                       dys_iters_max=int(info["iters"].max()), dys_residual_max=float(info["resid"].max()),
                       sigma_min_active=float(info["sigma_min_active"].min()),
                       proj_fail=int(getattr(proj, "n_fail", 0)))
        summary.update(prob.packing_metrics(info["z_final"]))
    print(f"EVAL ({tag}):", json.dumps(summary, indent=2))
    return summary, info


summary, info = evaluate("final network")
summary["best_safe_epoch"] = best_safe_epoch
with open(os.path.join(out_dir, "eval.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
plot_rollout(prob, info["traj"], os.path.join(out_dir, "rollout_final.png"), dt=args.dt)

if best_safe_epoch is not None:
    net.load_state_dict(torch.load(os.path.join(out_dir, "best_safe_control_net.pth"), map_location=device))
    summary_b, info_b = evaluate(f"best safe network (epoch {best_safe_epoch})")
    summary_b["best_safe_epoch"] = best_safe_epoch
    with open(os.path.join(out_dir, "eval_best_safe.json"), "w") as fh:
        json.dump(summary_b, fh, indent=2)
    plot_rollout(prob, info_b["traj"], os.path.join(out_dir, "rollout_best_safe.png"), dt=args.dt)
else:
    print("no epoch had min_h_true >= 0: no best_safe checkpoint was saved", flush=True)
print("saved to", out_dir)
