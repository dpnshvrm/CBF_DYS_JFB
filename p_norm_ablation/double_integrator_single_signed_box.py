import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from utils import DYSProjector, euler_step, rk4_step, ResBlock, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#### Settings for double integrator dynamics
A = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]).to(device)
B = torch.tensor([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]]).to(device)
                  
def f(z, u, t, A=A, B=B):
    """Dynamics: ż = Az + Bu"""
    output = (A @ z.T + B @ u.T).T
    assert output.shape == z.shape
    return output
    
def lagrangian(u):
    """Running cost"""
    return 0.5 * torch.norm(u, dim=-1)**2
    
def G(z, p_target):
    """Terminal cost"""
    output = 0.5 * torch.norm(z[:, :2] - p_target, dim=1)**2  # terminal cost on position only
    output = output.mean()
    assert output.shape == ()
    return output

#### HOCBF Settings
position_dimension = 2
box_delta = 1e-3   # smoothing for abs
box_beta  = 20.0   # sharpness of smooth max


def barrier_function(z, center, r_obs, eps):
    pos = z[:, :position_dimension]
    dp = pos - center

    a = r_obs + eps
    sx = torch.sqrt(dp[:, 0]**2 + box_delta**2)
    sy = torch.sqrt(dp[:, 1]**2 + box_delta**2)

    qx = sx - a
    qy = sy - a

    h = torch.logsumexp(box_beta * torch.stack([qx, qy], dim=-1),dim=-1) / box_beta
    return h
    

    
def evaluate_barriers(z, centers, r_obs, eps):
    """Returns (batch, num_obstacles) values of ψ₀ = h."""
    return torch.stack([barrier_function(z, c, r_obs, eps) for c in centers], dim=-1)
    
def gamma(x):
    """Class-K function used in both HOCBF lifts (α₁ and α₂)."""
    return 1.0 * x
    
def psi1_function(z, center, r_obs, eps):
    pos = z[:, :2]
    vel = z[:, 2:]
    dp = pos - center

    a = r_obs + eps
    dx = dp[:, 0]
    dy = dp[:, 1]

    sx = torch.sqrt(dx**2 + box_delta**2)
    sy = torch.sqrt(dy**2 + box_delta**2)

    qx = sx - a
    qy = sy - a

    logits = box_beta * torch.stack([qx, qy], dim=-1)
    w = torch.softmax(logits, dim=-1)

    h = torch.logsumexp(logits, dim=-1) / box_beta

    grad_qx = torch.stack([dx / sx, torch.zeros_like(dx)], dim=-1)
    grad_qy = torch.stack([torch.zeros_like(dy), dy / sy], dim=-1)

    grad_h = w[:, 0:1] * grad_qx + w[:, 1:2] * grad_qy
    Lf_h = torch.sum(grad_h * vel, dim=-1)

    return Lf_h + gamma(h)


def evaluate_psi1(z, centers, r_obs, eps):
    """Returns (batch, num_obstacles) values of ψ₁."""
    return torch.stack([psi1_function(z, c, r_obs, eps) for c in centers], dim=-1)
    
def construct_cbf_constraints(z, centers, r_obs, eps):
    batch = z.shape[0]
    pos = z[:, :2]
    vel = z[:, 2:]

    K_list, d_list = [], []

    for center in centers:
        dp = pos - center

        a = r_obs + eps
        dx = dp[:, 0]
        dy = dp[:, 1]

        sx = torch.sqrt(dx**2 + box_delta**2)
        sy = torch.sqrt(dy**2 + box_delta**2)

        qx = sx - a
        qy = sy - a

        logits = box_beta * torch.stack([qx, qy], dim=-1)
        w = torch.softmax(logits, dim=-1)

        h = torch.logsumexp(logits, dim=-1) / box_beta

        grad_qx = torch.stack([dx / sx, torch.zeros_like(dx)], dim=-1)
        grad_qy = torch.stack([torch.zeros_like(dy), dy / sy], dim=-1)
        grad_h = w[:, 0:1] * grad_qx + w[:, 1:2] * grad_qy

        Lf_h = torch.sum(grad_h * vel, dim=-1)
        psi1 = Lf_h + gamma(h)

        # Hessians of qx and qy
        Hx = torch.zeros(batch, 2, 2, device=z.device, dtype=z.dtype)
        Hy = torch.zeros(batch, 2, 2, device=z.device, dtype=z.dtype)
        Hx[:, 0, 0] = box_delta**2 / (sx**3)
        Hy[:, 1, 1] = box_delta**2 / (sy**3)

        diff = (grad_qx - grad_qy).unsqueeze(-1)
        outer = diff @ diff.transpose(-2, -1)

        hess_h = (
            w[:, 0].view(-1, 1, 1) * Hx
            + w[:, 1].view(-1, 1, 1) * Hy
            + box_beta * (w[:, 0] * w[:, 1]).view(-1, 1, 1) * outer
        )

        vel_col = vel.unsqueeze(-1)
        Lf2_h = (vel.unsqueeze(1) @ hess_h @ vel_col).squeeze(-1).squeeze(-1)

        K_i = -grad_h
        d_i = Lf2_h + gamma(Lf_h) + gamma(psi1)

        K_list.append(K_i.unsqueeze(1))
        d_list.append(d_i.view(batch, 1, 1))

    K_cbf = torch.cat(K_list, dim=1)
    d_cbf = torch.cat(d_list, dim=1)
    return K_cbf, d_cbf
        
        
def sample_initial_condition(z0_mean, z0_std, batch_size=64, a=-1, b=0.5, mode="gaussian"):
    """Sample initial conditions"""
    if mode == "gaussian":
        noise = torch.zeros(batch_size, 4, device=device)
        noise[:, :2] = torch.randn(batch_size, 2, device=device) 
        return z0_mean + z0_std * noise
        # return z0_mean + z0_std * torch.randn(batch_size, 4, device=device)
    else:
        noise = torch.zeros(batch_size, 4, device=device)
        noise[:, :2] = (b-a) * torch.rand(batch_size, 2, device=device) +a
        return noise
        
def compute_loss(u, z0, nt, f, p_target, obstacle_centers, r_obstacle, eps_safe,
                 alpha_running, alpha_terminal, proj, verbose=False, return_traj=True, dt=None):
    """
    Rolls out policy u for nt steps under the HOCBF filter, accumulating
    running and terminal costs.

    At each step:
      1. Query policy for u_nom.
      2. Build HOCBF constraints K_cbf u ≤ d_cbf (ψ₂ ≥ 0) at current state.
      3. If u_nom violates any constraint (K_cbf u_nom > d_cbf), solve the QP
         via DYS to obtain u_safe.
      4. Advance the state with the applied control via RK4.

    Args:
        dt: Physical time step. If None, defaults to 1/nt (normalized [0,1] horizon).
    """
    running_cost = 0.
    z  = z0
    h  = dt if dt is not None else 1.0 / nt
    ti = torch.zeros(1, device=z0.device)
    batch_size = z0.shape[0]
    state_dim  = z0.shape[1]

    isprojected         = 0
    max_res_norm_array  = torch.zeros(nt)
    n_iters_array       = torch.zeros(nt)
    barrier_value_array = torch.zeros(nt)
    if return_traj:
        traj = torch.zeros(batch_size, state_dim, nt + 1, device=z0.device)
        traj[:, :, 0] = z0.detach().clone()
    else:
        traj = None

    for i in range(nt):
        u_nom = u(z, ti)
        assert u_nom.shape == (batch_size, 2)

        # Build HOCBF constraints K_cbf u ≤ d_cbf (ψ₂ ≥ 0) at current state
        K_cbf, d_cbf = construct_cbf_constraints(z.detach(), obstacle_centers, r_obstacle, eps_safe)

        # Check whether u_nom violates ψ₂ ≥ 0 for any obstacle / batch element
        psi2_nom   = d_cbf - K_cbf @ u_nom.unsqueeze(-1)   # (batch, n_obs, 1)
        hocbf_viol = (-psi2_nom).clamp(min=0).max().item()
        is_projected = hocbf_viol > 0

        if is_projected:
            current_u, _, info = proj(u_nom, K_cbf, d_cbf,
                                      max_iter=5000, tol=5e-3, verbose=verbose)
            isprojected           = 1
            n_iters_array[i]      = info['iters']
            max_res_norm_array[i] = info['final_residual']

            if verbose:
                psi2_safe = (d_cbf - K_cbf @ current_u.unsqueeze(-1)).min().item()
                print(f't {i*h:.3f}  CBF ACTIVE  '
                      f'ψ₂_nom_min={-hocbf_viol:.2e}  '
                      f'ψ₂_safe_min={psi2_safe:.2e}  '
                      f'iters={info["iters"]:4d}  '
                      f'converged={info["converged"]}')
        else:
            current_u             = u_nom
            n_iters_array[i]      = 0
            max_res_norm_array[i] = 0.0

        z  = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()

        barrier_value_array[i] = evaluate_barriers(z, obstacle_centers, r_obstacle, eps_safe).min().item()
        running_cost = running_cost + h * lagrangian(current_u)

    running_cost  = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost    = alpha_running * running_cost + alpha_terminal * terminal_cost

    assert traj.shape == (batch_size, state_dim, nt + 1) if return_traj else True

    return total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj
    

#### plotter
def plot_trajectory(traj, obstacle_centers, obstacle_radius, p_target,
                    title="Trajectory with Obstacles", eps_safe=0.1, n_traj=10):

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(min(traj.shape[0], n_traj)):
        ax.plot(traj[i, 0, :], traj[i, 1, :], marker='o')

    # Plot obstacles as Lp balls
    for center in obstacle_centers:
        c = center[0]

        # obstacle boundary
        rect = plt.Rectangle((c[0] - obstacle_radius, c[1] - obstacle_radius), 2 * obstacle_radius, 2 * obstacle_radius,
            fill=False, color='red', alpha=0.8)
        ax.add_patch(rect)

        # safety boundary
        r_safe = obstacle_radius + eps_safe
        rect_safe = plt.Rectangle((c[0] - r_safe, c[1] - r_safe), 2 * r_safe, 2 * r_safe,
            fill=False, color='black', linestyle='--', linewidth=2)
        ax.add_patch(rect_safe)
    # Plot target position
    plt.scatter(p_target[:, 0].item(), p_target[:, 1].item(),
                color='green', label='Target', s=400, marker='X')

    # plt.xlabel('Position X')
    # plt.ylabel('Position Y')
    # plt.title(title)
    plt.legend(fontsize=14, markerscale=1.5)
    plt.grid()
    
    plt.axis('equal')
    plt.savefig("trajs.png", bbox_inches="tight", dpi=400)
    
    
