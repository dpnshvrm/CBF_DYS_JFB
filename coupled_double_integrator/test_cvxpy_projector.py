"""
Sanity checks for CvxpyProjector (run in the implicitOC env):
    python coupled_double_integrator/test_cvxpy_projector.py
Checks: (1) hard CVXPY filter vs DYS solved tightly on the same states / nominal controls,
(2) slack variant is close to the hard one when the QP is feasible, (3) gradients through the layer are finite,
(4) margin: the CBF uses d + margin while 'true' safety uses d.
Tolerances are loose on purpose (SCS eps = 1e-3); the printed numbers are what matters.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from double_integrator_coupled import CoupledDoubleIntegrators, device
from cvxpy_projector import CvxpyProjector
from utils import DYSProjector

torch.manual_seed(0)
prob = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0)
B = 8
z = prob.sample_initial_condition(B)
# move agents closer so that constraints are active for some samples
pos = z.reshape(B, 3, 4)
pos[:, :, :2] = 1.5 + 0.5 * torch.randn(B, 3, 2, device=device)
pos[:, :, 2:] = 0.5 * torch.randn(B, 3, 2, device=device)
z = pos.reshape(B, -1)
K, dvec = prob.construct_cbf_constraints(z)
u_nom = 3.0 * torch.randn(B, prob.control_dim, device=device)
ok = True

hard = CvxpyProjector(prob.control_dim, prob.n_pair, slack=False).to(device)
slack = CvxpyProjector(prob.control_dim, prob.n_pair, slack=True).to(device)
dys = DYSProjector(alpha=0.5, grad_mode="jfb").to(device)

u_h, _, ih = hard(u_nom, K, dvec)
u_s, _, is_ = slack(u_nom, K, dvec)
u_d, _, idys = dys(u_nom, K, dvec, max_iter=20000, tol=1e-7)
viol = lambda u: float((K @ u.unsqueeze(-1) - dvec).clamp(min=0).max())
print(f"[1] nominal violation {viol(u_nom):.3e} | hard CVXPY {viol(u_h):.3e} (failed={ih['failed']}) | "
      f"slack CVXPY {viol(u_s):.3e} | DYS(tight) {viol(u_d):.3e}")
diff_hd = float((u_h - u_d).abs().max())
print(f"    max|u_cvxpy_hard - u_dys_tight| = {diff_hd:.3e}   (DYS iters {idys['iters']})")
ok &= (diff_hd < 5e-2) and (not ih["failed"])

diff_sh = float((u_s - u_h).abs().max())
print(f"[2] max|u_slack - u_hard| = {diff_sh:.3e}   (expected small when the hard QP is feasible)")
ok &= diff_sh < 5e-2

un = u_nom.clone().requires_grad_(True)
u_g, _, _ = hard(un, K, dvec)
(u_g ** 2).sum().backward()
g_ok = bool(torch.isfinite(un.grad).all())
print(f"[3] gradient through hard layer finite: {g_ok}   |grad|max = {float(un.grad.abs().max()):.3e}")
ok &= g_ok

pm = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0, margin=0.1)
z_close = torch.zeros(1, 12, device=device)
z_close.reshape(1, 3, 4)[:, :, :2] = torch.tensor([[0.0, 0.0], [0.75, 0.0], [0.0, 3.0]], device=device)
h_cbf, h_true = float(pm.barrier(z_close).min()), float(pm.true_barrier(z_close).min())
print(f"[4] margin 0.1: CBF uses d={pm.cbf.d:.2f}; pair at distance 0.75 -> h_cbf={h_cbf:.3f} (<0), h_true={h_true:.3f} (>0)")
ok &= (abs(pm.cbf.d - 0.8) < 1e-9) and h_cbf < 0 < h_true

print("ALL OK" if ok else "SOME CHECKS FAILED")
