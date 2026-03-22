import torch
import numpy as np

from utils import rk4_step

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_agent = 50

m = 0.5
g = 1.0
T_hover = m * g

STATE_DIM = 12 * n_agent
CONTROL_DIM = 4 * n_agent

START_CENTER = torch.tensor([1.5, 0.0])
START_RADIUS = 0.5
TARGET_CENTER = torch.tensor([1.5, 3.0])
TARGET_RADIUS = 0.5
TARGET_Z = 1.0


def thrust_direction(psi, theta, phi):
    sps, cps = torch.sin(psi), torch.cos(psi)
    sth, cth = torch.sin(theta), torch.cos(theta)
    sph, cph = torch.sin(phi), torch.cos(phi)

    f7 = sps * sph + cps * sth * cph
    f8 = -cps * sph + sps * sth * cph
    f9 = cth * cph
    return torch.stack([f7, f8, f9], dim=-1)


def split_state(z):
    batch = z.shape[0]
    view = z.reshape(batch, n_agent, 12)
    return view[:, :, 0:3], view[:, :, 3:6], view[:, :, 6:9], view[:, :, 9:12]


def f(z, u, t=None):
    batch = z.shape[0]
    z_view = z.reshape(batch, n_agent, 12)
    u_view = u.reshape(batch, n_agent, 4)

    angles = z_view[:, :, 3:6]
    vel = z_view[:, :, 6:9]
    ang_vel = z_view[:, :, 9:12]

    thrust = u_view[:, :, 0]
    tau = u_view[:, :, 1:4]

    psi, theta, phi = angles[:, :, 0], angles[:, :, 1], angles[:, :, 2]
    td = thrust_direction(psi, theta, phi)
    e3 = torch.tensor([0.0, 0.0, 1.0], device=z.device, dtype=z.dtype)

    dp = vel
    dangles = ang_vel
    dv = (thrust / m).unsqueeze(-1) * td - g * e3
    dang_vel = tau

    dz = torch.cat([dp, dangles, dv, dang_vel], dim=-1)
    return dz.reshape(batch, STATE_DIM)


def lagrangian(u):
    return 0.5 * torch.norm(u, dim=-1) ** 2


def G(z, p_target):
    batch = z.shape[0]
    view = z.reshape(batch, n_agent, 12)
    pos = view[:, :, 0:3]
    angles = view[:, :, 3:6]
    vel = view[:, :, 6:9]
    ang_vel = view[:, :, 9:12]

    cost = (
        0.5 * ((pos - p_target.unsqueeze(0)) ** 2).sum(dim=-1).sum(dim=-1)
        + 0.5 * vel.pow(2).sum(dim=-1).sum(dim=-1)
        + 0.5 * angles.pow(2).sum(dim=-1).sum(dim=-1)
        + 0.5 * ang_vel.pow(2).sum(dim=-1).sum(dim=-1)
    )
    return cost.mean()


def gamma(x):
    return 1.0 * x


def barrier_function(z, center, r_obs, eps):
    pos, _, _, _ = split_state(z)
    return ((pos - center) ** 2).sum(dim=-1) - r_obs ** 2 - eps ** 2


def evaluate_barriers(z, centers, r_obs, eps):
    return torch.stack([barrier_function(z, c, r_obs, eps) for c in centers], dim=-1)


def psi1_function(z, center, r_obs, eps):
    pos, _, vel, _ = split_state(z)
    dp = pos - center
    h = dp.pow(2).sum(dim=-1) - r_obs ** 2 - eps ** 2
    return 2.0 * (dp * vel).sum(dim=-1) + gamma(h)


def evaluate_psi1(z, centers, r_obs, eps):
    return torch.stack([psi1_function(z, c, r_obs, eps) for c in centers], dim=-1)


def construct_cbf_constraints(z, centers, r_obs, eps):
    batch = z.shape[0]
    pos, angles, vel, _ = split_state(z)
    psi, theta, phi = angles[:, :, 0], angles[:, :, 1], angles[:, :, 2]
    td = thrust_direction(psi, theta, phi)

    K_list, d_list = [], []
    for center in centers:
        dp = pos - center
        h = dp.pow(2).sum(dim=-1) - r_obs ** 2 - eps ** 2
        lf_h = 2.0 * (dp * vel).sum(dim=-1)
        psi1 = lf_h + gamma(h)
        d_vals = (
            2.0 * vel.pow(2).sum(dim=-1)
            - 2.0 * g * dp[:, :, 2]
            + gamma(lf_h)
            + gamma(psi1)
        )

        for i in range(n_agent):
            K_row = torch.zeros(batch, CONTROL_DIM, device=z.device, dtype=z.dtype)
            K_row[:, 4 * i] = -(2.0 / m) * (dp[:, i, :] * td[:, i, :]).sum(dim=-1)
            K_list.append(K_row.unsqueeze(1))
            d_list.append(d_vals[:, i].view(batch, 1, 1))

    K_cbf = torch.cat(K_list, dim=1)
    d_cbf = torch.cat(d_list, dim=1)
    return K_cbf, d_cbf


def sample_initial_condition(batch_size=64, z0_std=0.03):
    start_center = START_CENTER.to(device)
    angles_xy = -2 * torch.pi * torch.arange(n_agent, device=device) / n_agent

    z0 = torch.zeros(batch_size, STATE_DIM, device=device)
    z0_view = z0.view(batch_size, n_agent, 12)

    z0_view[:, :, 0] = start_center[0] + START_RADIUS * torch.cos(angles_xy)
    z0_view[:, :, 1] = start_center[1] + START_RADIUS * torch.sin(angles_xy)
    z0_view[:, :, 2] = TARGET_Z

    z0_view[:, :, :2] += z0_std * torch.randn(batch_size, n_agent, 2, device=device)
    z0_view[:, :, 2] += 0.05 * z0_std * torch.randn(batch_size, n_agent, device=device)
    return z0


def build_target_positions(runtime_device):
    center = TARGET_CENTER.to(runtime_device)
    angles = -2 * torch.pi * torch.arange(n_agent, device=runtime_device) / n_agent
    p_target = torch.zeros(n_agent, 3, device=runtime_device)
    p_target[:, 0] = center[0] + TARGET_RADIUS * torch.cos(angles)
    p_target[:, 1] = center[1] + TARGET_RADIUS * torch.sin(angles)
    p_target[:, 2] = TARGET_Z
    return p_target


def compute_loss(
    u,
    z0,
    nt,
    f,
    p_target,
    obstacle_centers,
    r_obstacle,
    eps_safe,
    alpha_running,
    alpha_terminal,
    proj,
    verbose=False,
    return_traj=True,
    dt=None,
):
    running_cost = 0.0
    z = z0
    h = dt if dt is not None else 1.0 / nt
    ti = torch.zeros(1, device=z0.device)
    batch_size = z0.shape[0]

    hover_bias = torch.zeros(CONTROL_DIM, device=z0.device, dtype=z0.dtype)
    hover_bias[::4] = T_hover

    isprojected = 0
    max_res_norm_array = torch.zeros(nt)
    n_iters_array = torch.zeros(nt)
    barrier_value_array = torch.zeros(nt)

    if return_traj:
        traj = torch.zeros(batch_size, STATE_DIM, nt + 1, device=z0.device)
        traj[:, :, 0] = z0.detach().clone()
    else:
        traj = None

    for i in range(nt):
        u_nom = u(z, ti) + hover_bias
        K_cbf, d_cbf = construct_cbf_constraints(z.detach(), obstacle_centers, r_obstacle, eps_safe)

        psi2_nom = d_cbf - K_cbf @ u_nom.unsqueeze(-1)
        hocbf_viol = (-psi2_nom).clamp(min=0).max().item()
        is_projected = hocbf_viol > 0

        if is_projected:
            current_u, _, info = proj(u_nom, K_cbf, d_cbf, max_iter=5000, tol=5e-3, verbose=verbose)
            isprojected = 1
            n_iters_array[i] = info['iters']
            max_res_norm_array[i] = info['final_residual']
        else:
            current_u = u_nom
            n_iters_array[i] = 0
            max_res_norm_array[i] = 0.0

        z = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()

        barrier_value_array[i] = evaluate_barriers(z, obstacle_centers, r_obstacle, eps_safe).min().item()
        vel = z.reshape(z.shape[0], n_agent, 12)[:, :, 6:9]
        vel_penalty = 0.5 * vel.pow(2).sum(dim=(1, 2))
        running_cost = running_cost + h * lagrangian(current_u) + 2.0 * h * vel_penalty

    running_cost = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost = alpha_running * running_cost + alpha_terminal * terminal_cost

    return (
        total_cost,
        running_cost,
        terminal_cost,
        isprojected,
        n_iters_array,
        max_res_norm_array,
        barrier_value_array,
        traj,
    )
