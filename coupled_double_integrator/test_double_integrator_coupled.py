"""Sanity checks for double_integrator_coupled.py.  Run:  python coupled_double_integrator/test_double_integrator_coupled.py"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from double_integrator_coupled import CoupledDoubleIntegrators, BallCBF, AutogradCBF, device
from utils import DYSProjector

torch.manual_seed(0)
dt64 = torch.float64


def random_state(prob, B, spread=1.5, vel=2.0):
    """Random positions with all pair distances >= d, random velocities."""
    zs = []
    while len(zs) < B:
        p = torch.randn(prob.n, 2, dtype=dt64) * spread
        dm = min(float((p[i] - p[j]).norm()) for i in range(prob.n) for j in range(i + 1, prob.n))
        if dm >= prob.d:
            v = torch.randn(prob.n, 2, dtype=dt64) * vel
            zs.append(torch.cat([p, v], dim=-1).reshape(-1))
    return torch.stack(zs).to(device)


# 1. constraint row vs. finite differences of h2 = h'' + 2 gamma h' + gamma^2 h along constant-u motion
for n in (3, 4):
    prob = CoupledDoubleIntegrators(n_agent=n, d=0.5, gamma=1.3)
    z = random_state(prob, 20)
    u = torch.randn(20, prob.control_dim, dtype=dt64, device=device)
    K, d = prob.construct_cbf_constraints(z)
    h2_formula = d.squeeze(-1) - (K @ u.unsqueeze(-1)).squeeze(-1)           # >=0 <=> K u <= d

    def h_of_t(t):
        zt = z.clone()
        pos = z.reshape(20, n, 4)[:, :, :2] + z.reshape(20, n, 4)[:, :, 2:] * t + 0.5 * u.reshape(20, n, 2) * t ** 2
        vel = z.reshape(20, n, 4)[:, :, 2:] + u.reshape(20, n, 2) * t
        return prob.barrier(torch.cat([pos, vel], dim=-1).reshape(20, -1))
    e = 1e-3
    hd = (h_of_t(e) - h_of_t(-e)) / (2 * e)
    hdd = (h_of_t(e) - 2 * h_of_t(0.0) + h_of_t(-e)) / e ** 2
    h2_fd = hdd + 2 * prob.gamma * hd + prob.gamma ** 2 * h_of_t(0.0)
    err = float((h2_fd - h2_formula).abs().max())
    assert err < 1e-4, err
    print(f"[1] N={n}: |h2 finite-diff - (d - K u)| max = {err:.2e}")

# 2. generic autograd barrier gives the same rows as the closed form
prob_a = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0)
prob_b = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0, cbf=AutogradCBF(lambda x: (x ** 2).sum(-1) - 0.7 ** 2))
z = random_state(prob_a, 8)
Ka, da = prob_a.construct_cbf_constraints(z)
Kb, db = prob_b.construct_cbf_constraints(z)
print(f"[2] autograd vs closed form: max|dK|={float((Ka - Kb).abs().max()):.2e}  max|dd|={float((da - db).abs().max()):.2e}")
assert torch.allclose(Ka, Kb, atol=1e-8) and torch.allclose(da, db, atol=1e-8)

# 3. each row has exactly two nonzero 2-blocks, opposite in sign
K4 = Ka.reshape(8, prob_a.n_pair, 3, 2)
nz = (K4.abs().sum(-1) > 0).sum(-1)
assert bool((nz == 2).all()), nz
assert torch.allclose(K4.sum(2), torch.zeros_like(K4.sum(2)), atol=1e-12)
print("[3] rows touch exactly two agents with opposite blocks")

# 4. DYS projection satisfies K u* <= d when the nominal control violates it
prob = CoupledDoubleIntegrators(n_agent=3, d=0.7, gamma=1.0)
proj = DYSProjector(alpha=0.5, grad_mode="jfb")
z = random_state(prob, 16, spread=1.0, vel=1.5).float()
K, d = prob.construct_cbf_constraints(z)
u_nom = (torch.randn(16, prob.control_dim, device=device) * 5.0)
u_star, _, info = proj(u_nom, K, d, max_iter=5000, tol=1e-6)
viol_nom = float((K @ u_nom.unsqueeze(-1) - d).amax())
viol_proj = float((K @ u_star.unsqueeze(-1) - d).amax())
print(f"[4] max constraint violation: nominal {viol_nom:.3f} -> projected {viol_proj:.2e}  (DYS iters {info['iters']}, converged={info['converged']})")
assert viol_nom > 0 and viol_proj < 1e-3

# 5. initial states are safe: h > 0 and h_1 > 0
z0 = prob.sample_initial_condition(256)
print(f"[5] initial min h = {float(prob.barrier(z0).min()):.3f}, min h1 = {float(prob.psi1(z0).min()):.3f}")
assert float(prob.barrier(z0).min()) > 0 and float(prob.psi1(z0).min()) > 0

# 6. equilateral triangle of side d at the target gives position cost d^2/2 (the floor)
d0 = prob.d
ang = math.pi / 2 + 2 * math.pi * torch.arange(3) / 3
tri = (d0 / math.sqrt(3)) * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1).to(device) + prob.center
zT = torch.cat([tri, torch.zeros(3, 2, device=device)], dim=-1).reshape(1, -1)
pm = prob.packing_metrics(zT)
print("[6] equilateral packing:", {k: round(v, 6) for k, v in pm.items()})
assert abs(pm['pos_cost'] - d0 ** 2 / 2) < 1e-5 and abs(pm['pos_cost_minus_floor']) < 1e-5 and pm['side_error_max'] < 1e-5

# 7. dynamics: RK4 free flight with zero control matches p + v t
from utils import rk4_step
z = z0[:2]
zt = z
ti = torch.zeros(1, device=device)
for _ in range(10):
    zt = rk4_step(zt, torch.zeros(2, prob.control_dim, device=device), ti, 0.2, prob.f)
pos0, vel0 = prob.split_state(z)
pos1, _ = prob.split_state(zt)
print(f"[7] free-flight error = {float((pos1 - (pos0 + vel0 * 2.0)).abs().max()):.2e}")
print("ALL TESTS PASSED")
