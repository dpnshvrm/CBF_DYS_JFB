"""
Coupled multi-agent double integrators with ONE SHARED pairwise HOCBF per pair of agents.

Standalone experiment: nothing in the repo is modified; we only import
DYSProjector / rk4_step from the repo's utils.py.

Notation (see PLAN.md):
    agent i:   z^(i) = (p^(i), v^(i)) in R^4,   control u^(i) in R^2,   p' = v, v' = u
    pair (i,j): dp = p^(i) - p^(j),  dv = v^(i) - v^(j),  h_ij = h(dp) >= 0   (default h = |dp|^2 - d^2)
    HOCBF (relative degree 2, linear class-K functions with slope gamma):
        h2 = h'' + 2*gamma*h' + gamma^2*h >= 0
        h'  = grad_h(dp)^T dv
        h'' = dv^T Hess_h(dp) dv + grad_h(dp)^T (u^(i) - u^(j))
    which is the affine constraint  K(z) u <= d(z)  with one row per pair:
        -grad_h(dp)^T (u^(i) - u^(j))  <=  dv^T Hess_h dv + 2*gamma*grad_h^T dv + gamma^2*h
Everything about the barrier lives in a CBF object (BallCBF below); to change the barrier, pass a
different object with methods h(dp), grad(dp), hess(dp) (e.g. AutogradCBF(h_fn)).
"""
import os
import sys
import math

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from utils import DYSProjector, rk4_step  # noqa: E402  (repo's utils.py)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------------------------
# Barrier definitions: h as a function of the relative position dp = p_i - p_j, shape (..., 2)
# ----------------------------------------------------------------------------------------------
class BallCBF:
    """h(dp) = |dp|^2 - d^2 with closed-form gradient 2 dp and Hessian 2 I."""

    def __init__(self, d):
        self.d = float(d)

    def h(self, dp):
        return (dp ** 2).sum(-1) - self.d ** 2

    def grad(self, dp):
        return 2.0 * dp

    def hess(self, dp):
        eye = torch.eye(dp.shape[-1], device=dp.device, dtype=dp.dtype)
        return 2.0 * eye.expand(*dp.shape[:-1], -1, -1)


class AutogradCBF:
    """
    Any smooth barrier h_fn(dp) -> scalar (written so that it works on a single vector of shape (2,)
    and, with `.sum(-1)`-style reductions, on batches (..., 2)). Gradient and Hessian by torch.func.
    Example:  AutogradCBF(lambda x: (x.abs() ** 4).sum(-1) - 0.7 ** 4)
    """

    def __init__(self, h_fn):
        self.h_fn = h_fn

    def h(self, dp):
        return self.h_fn(dp)

    def grad(self, dp):
        g = torch.func.vmap(torch.func.grad(self.h_fn))(dp.reshape(-1, dp.shape[-1]))
        return g.reshape(dp.shape)

    def hess(self, dp):
        H = torch.func.vmap(torch.func.hessian(self.h_fn))(dp.reshape(-1, dp.shape[-1]))
        return H.reshape(*dp.shape[:-1], dp.shape[-1], dp.shape[-1])


# ----------------------------------------------------------------------------------------------
# The problem
# ----------------------------------------------------------------------------------------------
class CoupledDoubleIntegrators:
    def __init__(self, n_agent=3, d=0.7, gamma=1.0, center=(1.5, 1.5), radius=math.sqrt(8.0),
                 z0_std=5e-2, cbf=None, vel_weight=1.0, margin=0.0):
        """
        n_agent    number of agents N
        d          safety distance (used by the default BallCBF)
        gamma      slope of the linear class-K functions alpha_1 = alpha_2 = gamma * s
        center     common target p_target (and centre of the initial circle)
        radius     radius of the initial circle (repo: sqrt(8))
        z0_std     std of the Gaussian jitter on the initial positions (repo: 5e-2)
        vel_weight weight w in the running cost 0.5|u|^2 + w|v|^2   (repo code: w = 1)
        margin     safety margin eps >= 0: the CBF enforces distance >= d + eps (the repo's obstacle examples use
                   eps_safe = 0.1-0.15); the quantities reported as 'true' safety (min_h_true, min_pair_distance)
                   always refer to the margin-free distance d.  Default 0 = no margin.
        """
        self.n = int(n_agent)
        self.d = float(d)
        self.gamma = float(gamma)
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.radius = float(radius)
        self.z0_std = float(z0_std)
        self.vel_weight = float(vel_weight)
        self.margin = float(margin)
        self.cbf = cbf if cbf is not None else BallCBF(d + self.margin)

        I, J = torch.triu_indices(self.n, self.n, offset=1)
        self.pair_i, self.pair_j = I.to(device), J.to(device)
        self.n_pair = int(I.numel())                 # c = N(N-1)/2
        self.state_dim = 4 * self.n                  # n in the tex
        self.control_dim = 2 * self.n                # m in the tex
        self.p_target = self.center.view(1, 2)       # common target for every agent

    # ---- state helpers --------------------------------------------------------------------
    def split_state(self, z):
        zv = z.reshape(z.shape[0], self.n, 4)
        return zv[:, :, :2], zv[:, :, 2:]            # pos (B,N,2), vel (B,N,2)

    def pair_differences(self, z):
        pos, vel = self.split_state(z)
        dp = pos[:, self.pair_i] - pos[:, self.pair_j]   # (B,c,2)
        dv = vel[:, self.pair_i] - vel[:, self.pair_j]
        return dp, dv

    # ---- dynamics: z' = A z + B u, written directly ---------------------------------------
    def f(self, z, u, t):
        zv = z.reshape(z.shape[0], self.n, 4)
        out = torch.cat([zv[:, :, 2:], u.reshape(z.shape[0], self.n, 2)], dim=-1)
        return out.reshape(z.shape[0], -1)

    # ---- barriers -------------------------------------------------------------------------
    def barrier(self, z):
        dp, _ = self.pair_differences(z)
        return self.cbf.h(dp)                                              # (B,c)

    def true_barrier(self, z):
        """Margin-free safety value |dp|^2 - d^2 (>= 0 iff every pair is at least d apart). Ball barrier only;
        for a custom barrier it falls back to the barrier itself."""
        if isinstance(self.cbf, BallCBF):
            dp, _ = self.pair_differences(z)
            return (dp ** 2).sum(-1) - self.d ** 2
        return self.barrier(z)

    def psi1(self, z):
        """h_1 = h' + gamma*h."""
        dp, dv = self.pair_differences(z)
        return (self.cbf.grad(dp) * dv).sum(-1) + self.gamma * self.cbf.h(dp)

    def construct_cbf_constraints(self, z):
        """K(z) u <= d(z) with one row per pair.  Returns K (B,c,2N), d (B,c,1)."""
        dp, dv = self.pair_differences(z)
        h = self.cbf.h(dp)
        g = self.cbf.grad(dp)                                              # (B,c,2)
        H = self.cbf.hess(dp)                                              # (B,c,2,2)
        hdot = (g * dv).sum(-1)
        rhs = torch.einsum('bci,bcij,bcj->bc', dv, H, dv) + 2 * self.gamma * hdot + self.gamma ** 2 * h

        B = z.shape[0]
        K4 = torch.zeros(B, self.n_pair, self.n, 2, device=z.device, dtype=z.dtype)
        rows = torch.arange(self.n_pair, device=z.device)
        K4[:, rows, self.pair_i] = -g          # block of agent i
        K4[:, rows, self.pair_j] = g           # block of agent j
        return K4.reshape(B, self.n_pair, 2 * self.n), rhs.unsqueeze(-1)

    # ---- initial distribution rho (the repo's, with N agents) ------------------------------
    def sample_initial_condition(self, batch_size):
        angles = -2 * math.pi * torch.arange(self.n, device=device) / self.n
        z0 = torch.zeros(batch_size, self.n, 4, device=device)
        z0[:, :, 0] = self.center[0] + self.radius * torch.cos(angles)
        z0[:, :, 1] = self.center[1] + self.radius * torch.sin(angles)
        z0[:, :, :2] += self.z0_std * torch.randn(batch_size, self.n, 2, device=device)
        return z0.reshape(batch_size, -1)

    # ---- costs ----------------------------------------------------------------------------
    def terminal_cost(self, z):
        """G = 0.5 sum_i |p_i - p_target|^2 + 0.5 sum_i |v_i|^2, averaged over the batch."""
        pos, vel = self.split_state(z)
        g = 0.5 * ((pos - self.p_target.unsqueeze(0)) ** 2).sum(dim=(1, 2)) + 0.5 * (vel ** 2).sum(dim=(1, 2))
        return g.mean()

    # ---- diagnostics ----------------------------------------------------------------------
    def sigma_min_active(self, K, slack, act_tol=1e-3):
        """
        Smallest singular value of the rows of K whose constraint is active after projection
        (slack = d - K u* <= act_tol). Returns the minimum over the batch (inf if nothing is active).
        """
        mask = (slack.squeeze(-1) <= act_tol)                              # (B,c)
        k = mask.sum(dim=1)                                                # (B,)
        if int(k.max()) == 0:
            return float('inf')
        sv = torch.linalg.svdvals(K * mask.unsqueeze(-1).to(K.dtype))      # (B, min(c,2N)) descending
        idx = (k.clamp(min=1, max=sv.shape[1]) - 1).unsqueeze(1)
        sig = sv.gather(1, idx).squeeze(1)
        sig = torch.where(k > 0, sig, torch.full_like(sig, float('inf')))
        return float(sig.min())

    def packing_metrics(self, z):
        """Terminal packing diagnostics (meaningful for the ball barrier): see PLAN.md Section 6."""
        pos, _ = self.split_state(z)
        e_pos = 0.5 * ((pos - self.p_target.unsqueeze(0)) ** 2).sum(dim=(1, 2))            # (B,)
        out = dict(
            pos_cost=float(e_pos.mean()),
            centroid_offset=float((pos.mean(dim=1) - self.p_target).norm(dim=-1).mean()),
        )
        if isinstance(self.cbf, BallCBF):
            dist = (pos[:, self.pair_i] - pos[:, self.pair_j]).norm(dim=-1)                # (B,c)
            floor = (self.n - 1) * self.d ** 2 / 4.0                                       # bound of PLAN.md Sec. 6
            floor_m = (self.n - 1) * (self.d + self.margin) ** 2 / 4.0                     # floor when the margin is enforced
            out.update(pos_cost_floor=floor, pos_cost_minus_floor=float(e_pos.mean()) - floor,
                       pos_cost_floor_with_margin=floor_m,
                       side_error_max=float((dist - self.d).abs().max(dim=1).values.mean()),
                       min_pair_distance=float(dist.min()))
        return out

    # ---- rollout with the CBF-QP safety filter solved by DYS --------------------------------
    def compute_loss(self, u_fn, z0, nt, dt, proj, alpha_running, alpha_terminal,
                     max_iter=5000, tol=5e-3, return_traj=True, track_sigma=True):
        """
        Same structure as compute_loss in double_integrator_multi.py (constraints built from z.detach(),
        projection only if the nominal control violates some row, RK4 step), but for the coupled problem.
        Returns total_cost, running_cost, terminal_cost, info(dict).
        """
        B = z0.shape[0]
        z = z0
        ti = torch.zeros(1, device=z0.device)
        running = 0.0

        traj = None
        if return_traj:
            traj = torch.zeros(B, self.state_dim, nt + 1, device=z0.device)
            traj[:, :, 0] = z0.detach()

        active_frac = torch.zeros(nt)          # fraction of samples whose nominal control violates a row
        iters = torch.zeros(nt)
        resid = torch.zeros(nt)
        min_h = torch.zeros(nt)
        min_h_true = torch.zeros(nt)
        min_psi1 = torch.zeros(nt)
        sig_min = torch.full((nt,), float('inf'))

        for i in range(nt):
            u_nom = u_fn(z, ti)
            assert u_nom.shape == (B, self.control_dim)

            K, dvec = self.construct_cbf_constraints(z.detach())            # K u <= d
            slack_nom = dvec - K @ u_nom.unsqueeze(-1)                       # >= 0 means satisfied
            viol = (-slack_nom).clamp(min=0).amax(dim=(1, 2))                # (B,)
            active_frac[i] = float((viol > 0).float().mean())

            if bool(viol.max() > 0):
                u, _, pinfo = proj(u_nom, K, dvec, max_iter=max_iter, tol=tol)
                iters[i] = pinfo['iters']
                resid[i] = pinfo['final_residual']
                if track_sigma:
                    with torch.no_grad():
                        sig_min[i] = self.sigma_min_active(K, dvec - K @ u.detach().unsqueeze(-1))
            else:
                u = u_nom

            z = rk4_step(z, u, ti, dt, self.f)
            ti = ti + dt
            if return_traj:
                traj[:, :, i + 1] = z.detach()

            with torch.no_grad():
                min_h[i] = float(self.barrier(z).min())
                min_h_true[i] = float(self.true_barrier(z).min())
                min_psi1[i] = float(self.psi1(z).min())
            _, vel = self.split_state(z)
            running = running + dt * 0.5 * (u ** 2).sum(dim=-1) + dt * self.vel_weight * (vel ** 2).sum(dim=(1, 2))

        running_cost = running.mean()
        terminal_cost = self.terminal_cost(z)
        total_cost = alpha_running * running_cost + alpha_terminal * terminal_cost
        info = dict(traj=traj, z_final=z.detach(), active_frac=active_frac, iters=iters, resid=resid,
                    min_h=min_h, min_h_true=min_h_true, min_psi1=min_psi1, sigma_min_active=sig_min)
        return total_cost, running_cost, terminal_cost, info


# ----------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------
def plot_rollout(problem, traj, path, sample=0, dt=0.2):
    """traj: (B, 4N, nt+1) tensor/array.  Left: xy paths of one sample; right: pairwise distances vs d."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr = traj.detach().cpu().numpy() if hasattr(traj, 'detach') else np.asarray(traj)
    n = problem.n
    x = tr[sample].reshape(n, 4, -1)                      # (N, 4, nt+1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.tab10(np.arange(n) % 10)
    for a in range(n):
        ax[0].plot(x[a, 0], x[a, 1], color=colors[a], lw=1.6, label=f'agent {a + 1}')
        ax[0].plot(x[a, 0, 0], x[a, 1, 0], 'o', color=colors[a])
        ax[0].plot(x[a, 0, -1], x[a, 1, -1], '*', color=colors[a], ms=13)
        ax[0].add_patch(plt.Circle((x[a, 0, -1], x[a, 1, -1]), problem.d / 2, fill=False, color=colors[a], ls=':'))
    c = problem.center.cpu().numpy()
    ax[0].plot(c[0], c[1], 'kx', ms=10, label='common target')
    ax[0].set_aspect('equal'); ax[0].set_title('paths (o start, * end, dotted: radius d/2)')
    ax[0].legend(fontsize=8)
    tt = np.arange(x.shape[-1]) * dt
    I, J = problem.pair_i.cpu().numpy(), problem.pair_j.cpu().numpy()
    for k in range(problem.n_pair):
        dist = np.linalg.norm(x[I[k], :2] - x[J[k], :2], axis=0)
        ax[1].plot(tt, dist, label=f'|p{I[k] + 1}-p{J[k] + 1}|')
    ax[1].axhline(problem.d, color='k', ls='--', label='d')
    ax[1].set_xlabel('t'); ax[1].set_title('pairwise distances'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
