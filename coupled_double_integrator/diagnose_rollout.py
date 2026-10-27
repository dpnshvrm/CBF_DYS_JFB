"""
Why does a trained controller violate the pairwise safety constraint?  Read-only diagnostic.

Rolls out a trained network from the repo's initial distribution and reports, per control interval dt,
  * min h and min h1 over the rollout (safety margins), terminal cost G, largest nominal / filtered control,
  * how well DYS solved the QP: max(K u* - d) after the DYS projection (should be <= ~0),
  * with --tight: the same quantity for a very tight DYS solve, and max|u_dys - u_tight|.
Comparing dt = 0.2 (training) with smaller dt for the SAME network separates
  (a) DYS not solving the QP accurately,  (b) the zero-order-hold time step being too coarse.

Usage (repo root, implicitOC env), e.g.:
    python coupled_double_integrator/diagnose_rollout.py --run_dir coupled_double_integrator/runs/<your run> --tight
"""
import os, sys, json, argparse, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch
from double_integrator_coupled import CoupledDoubleIntegrators, device
from utils import DYSProjector, ControlNet, rk4_step

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", default=None, help="folder with args.json and the weights (default: newest in runs/)")
ap.add_argument("--weights", default="final_control_net.pth")
ap.add_argument("--dts", default="0.2,0.1,0.05,0.02", help="comma-separated control intervals to test")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--dys_tol", type=float, default=5e-3)
ap.add_argument("--tight", action="store_true", help="also solve each active QP with a very tight DYS tolerance")
ap.add_argument("--double", action="store_true", help="run in float64")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--tail", type=int, default=12, help="print the last k steps of the dt=first run")
a = ap.parse_args()

if a.run_dir is None:
    cands = [d for d in glob.glob(os.path.join(_HERE, "runs", "*")) if os.path.exists(os.path.join(d, a.weights))]
    if not cands:
        sys.exit("no run folder with weights found; pass --run_dir")
    a.run_dir = max(cands, key=os.path.getmtime)
cfg = json.load(open(os.path.join(a.run_dir, "args.json")))
dtype = torch.float64 if a.double else torch.float32

prob = CoupledDoubleIntegrators(n_agent=cfg["n_agent"], d=cfg["d"], gamma=cfg["gamma"],
                                z0_std=cfg["z0_std"], vel_weight=cfg["vel_weight"])
net = ControlNet(input_dim=prob.state_dim + 1, hidden_dim=cfg["hidden_dim"], output_dim=prob.control_dim,
                 n_blocks=cfg["n_blocks"]).to(device)
net.load_state_dict(torch.load(os.path.join(a.run_dir, a.weights), map_location=device))
net = net.to(dtype).eval()
proj = DYSProjector(alpha=cfg["dys_alpha"], grad_mode="jfb").to(device)
T = cfg["T"]
print(f"run: {a.run_dir}\n N={prob.n} d={prob.d} gamma={prob.gamma} trained with dt={cfg['dt']}  dys_tol(train)={cfg['dys_tol']}  "
      f"dtype={dtype}\n")


def rollout(dt, tight):
    torch.manual_seed(a.seed)
    z = prob.sample_initial_condition(a.batch).to(dtype)
    nt = int(round(T / dt))
    ti = torch.zeros(1, dtype=dtype, device=device)
    rows = []
    with torch.no_grad():
        for i in range(nt):
            u_nom = net(z, ti)
            K, dv = prob.construct_cbf_constraints(z)
            viol = float((-(dv - K @ u_nom.unsqueeze(-1))).clamp(min=0).max())
            after = float("nan"); after_t = float("nan"); diff = float("nan"); iters = 0
            if viol > 0:
                u, _, info = proj(u_nom, K, dv, max_iter=cfg["dys_max_iter"], tol=a.dys_tol)
                iters = info["iters"]
                after = float((K @ u.unsqueeze(-1) - dv).max())
                if tight:
                    u_t, _, _ = proj(u_nom, K, dv, max_iter=20000, tol=1e-9)
                    after_t = float((K @ u_t.unsqueeze(-1) - dv).max())
                    diff = float((u - u_t).abs().max())
            else:
                u = u_nom
            z = rk4_step(z, u, ti, dt, prob.f)
            ti = ti + dt
            rows.append(dict(t=(i + 1) * dt, min_h=float(prob.barrier(z).min()), min_h1=float(prob.psi1(z).min()),
                             unom=float(u_nom.abs().max()), ustar=float(u.abs().max()), viol_nom=viol,
                             after=after, after_tight=after_t, diff=diff, iters=iters))
    return rows, float(prob.terminal_cost(z))


first = None
print(f"{'dt':>6} {'min h':>9} {'min h1':>9} {'G':>8} {'max|u_nom|':>11} {'max|u*|':>9} {'max(Ku*-d) DYS':>15}"
      + (f" {'max(Ku*-d) tight':>17} {'max|u_dys-u_tight|':>19}" if a.tight else ""))
for s in a.dts.split(","):
    dt = float(s)
    rows, G = rollout(dt, a.tight and first is None)
    if first is None:
        first = (dt, rows)
    fin = lambda k: [r[k] for r in rows if r[k] == r[k]]          # drop NaNs
    line = (f"{dt:6.3f} {min(r['min_h'] for r in rows):9.3f} {min(r['min_h1'] for r in rows):9.3f} {G:8.3f} "
            f"{max(r['unom'] for r in rows):11.2e} {max(r['ustar'] for r in rows):9.2e} "
            f"{(max(fin('after')) if fin('after') else float('nan')):15.2e}")
    if a.tight:
        line += (f" {(max(fin('after_tight')) if fin('after_tight') else float('nan')):17.2e}"
                 f" {(max(fin('diff')) if fin('diff') else float('nan')):19.2e}")
    print(line)

print(f"\nlast {a.tail} steps of the dt={first[0]} rollout (batch minima / maxima):")
print(f"{'t':>6} {'min h':>9} {'min h1':>9} {'max|u_nom|':>11} {'max|u*|':>9} {'max(Ku*-d)':>11} {'DYS it':>7}")
for r in first[1][-a.tail:]:
    print(f"{r['t']:6.2f} {r['min_h']:9.3f} {r['min_h1']:9.3f} {r['unom']:11.2e} {r['ustar']:9.2e} {r['after']:11.2e} {r['iters']:7d}")
neg = [r for r in first[1] if r["min_h1"] < 0]
negh = [r for r in first[1] if r["min_h"] < 0]
print("\nfirst step with h1 < 0:", f"t={neg[0]['t']:.2f}" if neg else "none", "|  first step with h < 0:", f"t={negh[0]['t']:.2f}" if negh else "none")
