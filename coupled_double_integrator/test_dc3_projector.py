"""
Checks for DC3Projector (run in the implicit-oc env, repo root):
    python coupled_double_integrator/test_dc3_projector.py
Prints, on the coupled 3-agent problem with strongly violating nominal controls:
 [1] remaining violation and distance to the exact projection vs number of steps and step size lr
     (exact projection = DYS solved tightly), to choose lr / steps and to see how far a fixed number of steps is from feasible;
 [2] finite gradients through the unrolled correction;
 [3] one violated constraint: the correction converges to the exact projection (sanity check of the formula).
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from double_integrator_coupled import CoupledDoubleIntegrators, device
from dc3_projector import DC3Projector
from utils import DYSProjector

torch.manual_seed(0)
prob = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0)
B = 64
z = prob.sample_initial_condition(B)
pos = z.reshape(B, 3, 4)
pos[:, :, :2] = 1.5 + 0.5 * torch.randn(B, 3, 2, device=device)       # agents close together: constraints active
pos[:, :, 2:] = 0.5 * torch.randn(B, 3, 2, device=device)
z = pos.reshape(B, -1)
K, dvec = prob.construct_cbf_constraints(z)
u_nom = 3.0 * torch.randn(B, prob.control_dim, device=device)
viol = lambda u: (K @ u.unsqueeze(-1) - dvec).clamp(min=0).amax(dim=(1, 2))      # per sample
dys = DYSProjector(alpha=0.5, grad_mode="jfb").to(device)
u_ex, _, _ = dys(u_nom, K, dvec, max_iter=20000, tol=1e-7)
print(f"nominal: max violation {float(viol(u_nom).max()):.3e}   exact projection: max violation {float(viol(u_ex).max()):.1e}")
print(f"spectral norm^2 of K: median {float(torch.linalg.matrix_norm(K, 2).pow(2).median()):.2f}, max {float(torch.linalg.matrix_norm(K, 2).pow(2).max()):.2f}"
      "  (heavy-ball stability needs roughly lr < 3 / (2 ||K||^2))")

print("\n[1] steps | lr | max violation | median violation | finite | median ||u - u_exact||")
for lr in (0.003, 0.01, 0.03, 0.06, 0.1):
    for steps in (5, 10, 100, 1000):
        p = DC3Projector(steps=steps, lr=lr, momentum=0.5).to(device)
        u, _, info = p(u_nom, K, dvec)
        v = viol(u)
        d = (u - u_ex).norm(dim=1)
        print(f"    {steps:5d} | {lr:<5g} | {float(v.max()):.3e} | {float(v.median()):.3e} | {not info['failed']} | {float(d.median()):.3e}")

ok = True
un = u_nom.clone().requires_grad_(True)
u_g, _, _ = DC3Projector(steps=10, lr=0.01).to(device)(un, K, dvec)
(u_g ** 2).sum().backward()
g_ok = bool(torch.isfinite(un.grad).all())
print(f"\n[2] gradient finite: {g_ok}   |grad|max = {float(un.grad.abs().max()):.3e}")
ok &= g_ok

# [3] one violated constraint, very small lr*steps large enough -> exact projection
K1 = K[:1].clone(); d1 = dvec[:1].clone(); un1 = u_nom[:1].clone()
slack = d1 - K1 @ un1.unsqueeze(-1)
mask = (slack < 0).squeeze(-1).squeeze(0)
if int(mask.sum()) >= 1:
    j = int(mask.nonzero()[0])
    Kj, dj = K1[:, j:j + 1], d1[:, j:j + 1]
    a = Kj[0, 0]
    u_proj = un1[0] - ((a @ un1[0] - dj[0, 0, 0]) / (a @ a)) * a
    lr1 = float(0.5 / (2 * (a @ a)))
    u1, _, _ = DC3Projector(steps=200, lr=lr1, momentum=0.0).to(device)(un1, Kj, dj)
    err = float((u1[0] - u_proj).norm())
    print(f"[3] one violated constraint: |u_dc3 - exact projection| = {err:.2e} (expected ~0)")
    ok &= err < 1e-4
# [4] wiring of the soft-constraint term: total(lambda=1) - total(lambda=0) == reported penalty, and ~0 for an exact filter
u_fn = lambda zz, tt: 4.0 * torch.ones(B, prob.control_dim, device=device)
p3 = DC3Projector(steps=3, lr=0.01).to(device)
t0, _, _, i0 = prob.compute_loss(u_fn, z, 5, 0.2, p3, 1.0, 20.0, return_traj=False, track_sigma=False, viol_penalty=0.0)
t1, _, _, i1 = prob.compute_loss(u_fn, z, 5, 0.2, p3, 1.0, 20.0, return_traj=False, track_sigma=False, viol_penalty=1.0)
w_ok = abs(float(t1 - t0) - i1["penalty"]) < 1e-4 * max(1.0, abs(i1["penalty"])) and i1["penalty"] > 0 and i0["penalty"] == 0.0
_, _, _, ie = prob.compute_loss(u_fn, z, 5, 0.2, dys, 1.0, 20.0, max_iter=20000, tol=1e-7, return_traj=False, track_sigma=False, viol_penalty=1.0)
print(f"[4] penalty term: total difference {float(t1 - t0):.4e} vs reported {i1['penalty']:.4e} (3 DC3 steps); exact DYS filter: {ie['penalty']:.2e}  ok={w_ok}")
ok &= w_ok
print("ALL OK" if ok else "SOME CHECKS FAILED")
