"""
Quadrotor Multi-Agent CBF Training
===================================
Trains a ControlNet policy for n_agent quadrotors using HOCBF safety constraints
enforced via the DYS differentiable projector (JFB backprop).

  State per agent  : 12-D  [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
  Control per agent:  4-D  [T, tau_x, tau_y, tau_z]
  CBF              : Position HOCBF, relative degree 2 w.r.t. thrust T
  Hover bias       : Network output is delta from hover; T_i = mg + net[T_i]
"""

from sched import scheduler
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import quadrotor_multi as _qm          # kept as module so we can patch device
from quadrotor_multi import (
    n_agent, STATE_DIM, CONTROL_DIM, T_hover, m, g,
    f, lagrangian, G,
    barrier_function, evaluate_barriers, gamma,
    psi1_function, evaluate_psi1, construct_cbf_constraints,
    sample_initial_condition, compute_loss,
)
from utils import DYSProjector, ControlNet

print(f'n_agent      : {n_agent}')
print(f'STATE_DIM    : {STATE_DIM}')
print(f'CONTROL_DIM  : {CONTROL_DIM}')
print(f'Hover thrust : {T_hover:.3f} N  (m={m} kg, g={g} m/s²)')
print(f'CUDA available: {torch.cuda.is_available()}')


# ── Device selection ──────────────────────────────────────────────────────────
# Options: 'cuda'  'cpu'  'mps' (Apple Silicon)
device = 'cuda'

# ── Validate & patch ──────────────────────────────────────────────────────────
if device.startswith('cuda'):
    if not torch.cuda.is_available():
        print('WARNING: CUDA requested but not available — falling back to CPU.')
        device = 'cpu'
    else:
        n_cuda = torch.cuda.device_count()
        if device == 'cuda':
            device = 'cuda:0'
        else:
            try:
                gpu_idx = int(device.split(':', 1)[1])
            except (IndexError, ValueError):
                print(f'WARNING: Invalid CUDA device "{device}" — using cuda:0.')
                device = 'cuda:0'
            else:
                if gpu_idx < 0 or gpu_idx >= n_cuda:
                    print(f'WARNING: {device} not available (found {n_cuda} CUDA device(s)) — using cuda:0.')
                    device = 'cuda:0'
elif device == 'mps' and not torch.backends.mps.is_available():
    print('WARNING: MPS requested but not available — falling back to CPU.')
    device = 'cpu'

# Patch the module so sample_initial_condition uses the chosen device.
_qm.device = device

print(f'Device: {device}')
if device.startswith('cuda'):
    gpu_idx = torch.device(device).index or 0
    print(f'GPU   : {torch.cuda.get_device_name(gpu_idx)}')
    print(f'VRAM  : {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.1f} GB')


# =============================================================================
# PARAMETERS  —  edit freely
# =============================================================================

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Horizon & timestep
T_horizon = 10.0    # total horizon [s]
dt        = 0.1     # integration timestep [s]
num_steps = int(T_horizon / dt)

# Cost weights
alpha_running  = 1.0
alpha_terminal = 100.0
weight_decay   = 1e-3

# Obstacles  (x, y, z)  — 3-D sphere centres
obstacle_cfg = [
    [0.63, 1.0,  1.0],
    [1.5,  2.5,  1.0],
    [2.37, 1.0,  1.0],
]
obstacle_radius = 0.35   # obstacle hard radius [m]
eps_safe        = 0.15   # safety margin added to radius in CBF

# Training
n_epochs        = int(1e4)
learning_rate   = 1e-4
lr_decay        = 300        # halve lr every this many epochs
batch_size      = 128
z0_std          = 0.05       # std of Gaussian noise on initial xy position
log_every       = 1          # print every N epochs
plot_every      = 10         # plot every N epochs

# alpha_terminal schedule (increment every N epochs up to max)
alpha_sched_every = 100
alpha_sched_step  = 0.0
alpha_sched_max   = 60.0

# Network architecture
hidden_dim = 128
n_blocks   = 3

print(f'Horizon  : {T_horizon}s   dt={dt}s   num_steps={num_steps}')
print(f'Obstacles: {len(obstacle_cfg)} spheres, r={obstacle_radius}, eps={eps_safe}')
print(f'Training : {n_epochs} epochs, batch={batch_size}, lr={learning_rate}')


# =============================================================================
# Setup — obstacles, targets, network
# =============================================================================

# ── Obstacles ─────────────────────────────────────────────────────────────────
obstacle_centers = [
    torch.tensor(c, dtype=torch.float32).view(1, 3).to(device)
    for c in obstacle_cfg
]
obs_np = [c.cpu().numpy() for c in obstacle_centers]   # numpy copy for plotting

# ── Target positions: small cluster at z=1 — above obstacle field ────────────
center_xy    = torch.tensor([1.5, 3.5], device=device)
radius_tgt   = 0.3          # tight cluster above obstacle field
angles_tgt   = -2 * torch.pi * torch.arange(n_agent, device=device) / n_agent
p_target     = torch.zeros(n_agent, 3, device=device)
p_target[:, 0] = center_xy[0] + radius_tgt * torch.cos(angles_tgt)
p_target[:, 1] = center_xy[1] + radius_tgt * torch.sin(angles_tgt)
p_target[:, 2] = 1.0

# ── Network, projector, optimiser ─────────────────────────────────────────────
T_dev_scale = 0.5    # max thrust deviation from hover  (u ∈ [T_hover - scale, T_hover + scale])
tau_scale   = 0.1    # max angular acceleration

net = ControlNet(
    input_dim  = STATE_DIM + 1,
    hidden_dim = hidden_dim,
    output_dim = CONTROL_DIM,
    n_blocks   = n_blocks,
).to(device)

proj = DYSProjector().to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.5)

def u_fn(z, t):
    raw = net(z, t).reshape(z.shape[0], n_agent, 4)
    scaled = torch.cat([
        T_dev_scale * torch.tanh(raw[:, :, 0:1]),
        tau_scale   * torch.tanh(raw[:, :, 1:4]),
    ], dim=-1)
    return scaled.reshape(z.shape[0], CONTROL_DIM)

print(f'Network params : {sum(p.numel() for p in net.parameters()):,}')
print(f'T_hover        : {T_hover:.3f}  (deviation scale ±{T_dev_scale})')
print(f'tau_scale      : ±{tau_scale}')


# ── Environment preview (also used during training) ───────────────────────────
# traj : optional numpy (B, STATE_DIM, T+1) — if given, trajectories are
#        overlaid on top of the environment geometry.

def preview_environment(n_samples=32, traj=None,
                        title='Environment preview  —  before training'):
    with torch.no_grad():
        z0_batch = sample_initial_condition(n_samples, z0_std)

    z0_np  = z0_batch.cpu().numpy()
    tgt_np = p_target.detach().cpu().numpy()

    with torch.no_grad():
        z0_nom = sample_initial_condition(1, z0_std=0.0)
    z0_nom_np = z0_nom.cpu().numpy()

    fig = plt.figure(figsize=(13, 7))

    # ── 3-D view ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab10(range(n_agent))

    if traj is not None:
        # Show trajectories instead of initial-condition clouds
        n_show = min(traj.shape[0], 8)
        for i in range(n_show):
            for a in range(n_agent):
                b = 12 * a
                ax3.plot(traj[i, b,   :], traj[i, b+1, :], traj[i, b+2, :],
                         color=colors[a], alpha=0.6, linewidth=1)
            ax3.scatter(traj[i, b, 0], traj[i, b+1, 0], traj[i, b+2, 0],
                        color=colors[a], s=20, marker='o')
    else:
        # Show perturbed-IC scatter cloud
        for a in range(n_agent):
            b = 12 * a
            ax3.scatter(z0_np[:, b], z0_np[:, b+1], z0_np[:, b+2],
                        color=colors[a], s=8, alpha=0.4)

    # Nominal start markers
    for a in range(n_agent):
        b = 12 * a
        ax3.scatter(z0_nom_np[0, b], z0_nom_np[0, b+1], z0_nom_np[0, b+2],
                    color=colors[a], s=120, marker='o', edgecolors='black',
                    linewidths=0.8, zorder=5, label=f'agent {a}')

    # Target positions
    ax3.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2],
                c='limegreen', s=160, marker='X', zorder=10,
                edgecolors='black', linewidths=0.8, label='target')

    # Dashed lines: start → target
    for a in range(n_agent):
        b = 12 * a
        ax3.plot([z0_nom_np[0, b],   tgt_np[a, 0]],
                 [z0_nom_np[0, b+1], tgt_np[a, 1]],
                 [z0_nom_np[0, b+2], tgt_np[a, 2]],
                 color=colors[a], linestyle=':', linewidth=0.8, alpha=0.6)

    # Obstacle spheres
    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi,     30)
    U, V = np.meshgrid(u_s, v_s)
    sx = np.sin(V) * np.cos(U);  sy = np.sin(V) * np.sin(U);  sz = np.cos(V)

    for c in obs_np:
        cx, cy, cz = c[0, 0], c[0, 1], c[0, 2]
        ax3.plot_surface(cx + obstacle_radius * sx,
                         cy + obstacle_radius * sy,
                         cz + obstacle_radius * sz,
                         color='red', alpha=0.35, linewidth=0)
        ax3.plot_wireframe(cx + (obstacle_radius + eps_safe) * sx,
                           cy + (obstacle_radius + eps_safe) * sy,
                           cz + (obstacle_radius + eps_safe) * sz,
                           color='red', alpha=0.12, linewidth=0.4)

    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.legend(fontsize=7, loc='upper left')
    ax3.view_init(elev=25, azim=45)

    # ── Top-down view (XY) ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(122)

    if traj is not None:
        n_show = min(traj.shape[0], 8)
        for i in range(n_show):
            for a in range(n_agent):
                b = 12 * a
                ax2.plot(traj[i, b, :], traj[i, b+1, :],
                         color=colors[a], alpha=0.6, linewidth=1)
    else:
        for a in range(n_agent):
            b = 12 * a
            ax2.scatter(z0_np[:, b], z0_np[:, b+1],
                        color=colors[a], s=8, alpha=0.4)

    for a in range(n_agent):
        b = 12 * a
        ax2.scatter(z0_nom_np[0, b], z0_nom_np[0, b+1],
                    color=colors[a], s=100, marker='o',
                    edgecolors='black', linewidths=0.8, zorder=5)
        ax2.annotate(f'a{a}', (z0_nom_np[0, b], z0_nom_np[0, b+1]),
                     textcoords='offset points', xytext=(4, 4), fontsize=8)

    ax2.scatter(tgt_np[:, 0], tgt_np[:, 1],
                c='limegreen', s=120, marker='X', zorder=10,
                edgecolors='black', linewidths=0.8, label='target')

    for c in obs_np:
        cx, cy = c[0, 0], c[0, 1]
        ax2.add_patch(plt.Circle((cx, cy), obstacle_radius, color='red', alpha=0.4))
        ax2.add_patch(plt.Circle((cx, cy), obstacle_radius + eps_safe,
                                  fill=False, edgecolor='red',
                                  linewidth=1.2, linestyle='--'))
        ax2.annotate(f'obs\nr={obstacle_radius}', (cx, cy),
                     ha='center', va='center', fontsize=7,
                     color='white', fontweight='bold')

    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title('Top view (XY)')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.35); ax2.legend()

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig('env_preview.png', dpi=120, bbox_inches='tight')
    plt.show()

    if traj is None:
        print(f'Agents        : {n_agent}')
        print(f'Nominal start : z = {z0_nom_np[0, 2]:.2f}')
        print(f'Targets       : z = {tgt_np[0, 2]:.2f}')
        print(f'Obstacles     : {len(obs_np)} spheres, r={obstacle_radius}, '
              f'margin={eps_safe}  (excl. r={obstacle_radius+eps_safe})')


preview_environment()


# =============================================================================
# Sanity Check: Evaluate Loss
# =============================================================================

# Run before training to verify the loss is finite and get a baseline.
# NaN here means the dynamics diverge before any gradient step —
# the diagnostic block below will tell you which agent/state blows up first.

net.eval()
with torch.no_grad():
    z0_check = sample_initial_condition(batch_size, z0_std=0.0)  # noise-free

    total0, run0, term0, proj0, _, _, barrier0, traj0 = compute_loss(
        u_fn, z0_check, num_steps, f, p_target,
        obstacle_centers, obstacle_radius, eps_safe,
        alpha_running, alpha_terminal, proj,
        dt=dt, return_traj=True,
    )

print(f'Initial loss    : {total0.item():.4e}')
print(f'  running cost  : {run0.item():.4e}')
print(f'  terminal cost : {term0.item():.4e}')
print(f'  CBF projected : {bool(proj0)}')
print(f'  min barrier h : {barrier0.min().item():.4e}')

if torch.isnan(total0):
    print()
    print('WARNING: NaN detected — finding first step where state diverges...')
    with torch.no_grad():
        from utils import rk4_step as _rk4
        hover_bias      = torch.zeros(CONTROL_DIM, device=device)
        hover_bias[::4] = T_hover
        z_dbg  = z0_check.clone()
        ti_dbg = torch.zeros(1, device=device)
        dim_names = ['px','py','pz','φ','θ','ψ','vx','vy','vz','p','q','r']

        for step in range(num_steps):
            u_raw  = u_fn(z_dbg, ti_dbg) + hover_bias
            z_next = _rk4(z_dbg, u_raw, ti_dbg, dt, f)

            if torch.isnan(z_next).any():
                bad = torch.isnan(z_next).any(dim=0).nonzero(as_tuple=False).squeeze(-1)
                print(f'  NaN first at step {step+1}  (t = {(step+1)*dt:.2f} s)')
                print(f'  |z| before step : max = {z_dbg.abs().max().item():.3e}')
                print(f'  |u| at step     : max = {u_raw.abs().max().item():.3e}')
                u_view = u_raw.reshape(batch_size, n_agent, 4)
                for a in range(n_agent):
                    T_a   = u_view[0, a, 0].item()
                    tau_a = u_view[0, a, 1:].tolist()
                    print(f'    agent {a}  T={T_a:.3f} N   '
                          f'τ=[{tau_a[0]:.4f}, {tau_a[1]:.4f}, {tau_a[2]:.4f}] Nm')
                for idx in bad.tolist():
                    a, d = idx // 12, idx % 12
                    print(f'    first NaN: agent {a}  →  {dim_names[d]}')
                break
            z_dbg  = z_next
            ti_dbg = ti_dbg + dt
        else:
            print('  State stayed finite — NaN originates in cost computation.')
else:
    print()
    print('Loss is finite — safe to start training.')


# =============================================================================
# Training loop
# =============================================================================

# ── History buffers ───────────────────────────────────────────────────────────
loss_history      = []
run_history       = []
term_history      = []
proj_history      = []
n_iters_history   = []
res_history       = []
barrier_history   = []
grad_norm_history = []

_alpha         = alpha_terminal    # mutable copy for scheduling
traj           = None
best_loss      = float('inf')
best_ckpt_path = 'quadrotor_control_net_best.pth'

net.train()

# ── Loop ──────────────────────────────────────────────────────────────────────
for epoch in range(1, n_epochs + 1):
    t0 = time.time()
    optimizer.zero_grad()

    z0_sample = sample_initial_condition(batch_size, z0_std)

    total_cost, running_cost, terminal_cost, isprojected, \
    n_iters_array, max_res_array, barrier_array, traj = compute_loss(
        u_fn, z0_sample, num_steps, f, p_target,
        obstacle_centers, obstacle_radius, eps_safe,
        alpha_running, _alpha, proj, dt=dt,
    )

    total_cost.backward()
    optimizer.step()

    gnorm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in net.parameters() if p.grad is not None
    ) ** 0.5

    t1 = time.time()

    loss_history.append(total_cost.item())
    run_history.append(running_cost.item())
    term_history.append(terminal_cost.item())
    proj_history.append(isprojected)
    n_iters_history.append(n_iters_array.max().item())
    res_history.append(max_res_array.max().item())
    barrier_history.append(barrier_array.min().item())
    grad_norm_history.append(gnorm)

    if epoch % log_every == 0:
        print(
            f'ep {epoch:4d} | '
            f'total={total_cost.item():.3e}  '
            f'run={running_cost.item():.3e}  '
            f'term={terminal_cost.item():.3e}  '
            f'proj={int(isprojected)}  '
            f'iters={int(n_iters_array.max().item()):4d}  '
            f'res={max_res_array.max().item():.2e}  '
            f'h_min={barrier_array.min().item():.2e}  '
            f'grad={gnorm:.2e}  '
            f't={t1 - t0:.1f}s'
        )

    if epoch % plot_every == 0:
        net.eval()
        with torch.no_grad():
            z0_plot = sample_initial_condition(batch_size, z0_std=0.0)
            _, _, _, _, _, _, _, traj_plot = compute_loss(
                u_fn, z0_plot, num_steps, f, p_target,
                obstacle_centers, obstacle_radius, eps_safe,
                alpha_running, _alpha, proj, dt=dt,
            )
        preview_environment(
            traj=traj_plot.cpu().numpy(),
            title=f'Epoch {epoch}  |  total={total_cost.item():.3e}  term={terminal_cost.item():.3e}',
        )
        net.train()

    if epoch % alpha_sched_every == 0 and _alpha < alpha_sched_max:
        _alpha += alpha_sched_step
        print(f'  alpha_terminal -> {_alpha:.1f}')

    # ── Save best checkpoint ─────────────────────────────────────────────────
    if total_cost.item() < best_loss:
        best_loss = total_cost.item()
        torch.save({
            'model_state_dict'    : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch'               : epoch,
            'loss'                : best_loss,
            'alpha_terminal'      : _alpha,
            'loss_history'        : loss_history,
            'config': {
                'n_agent'     : n_agent,
                'STATE_DIM'   : STATE_DIM,
                'CONTROL_DIM' : CONTROL_DIM,
                'hidden_dim'  : hidden_dim,
                'n_blocks'    : n_blocks,
                'T_horizon'   : T_horizon,
                'dt'          : dt,
            },
        }, best_ckpt_path)
        if epoch % log_every == 0:
            print(f'  ** best checkpoint saved  (loss={best_loss:.3e})  ->  {best_ckpt_path}')

    if epoch % lr_decay == 0:
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
        lr_now = optimizer.param_groups[0]['lr']
        print(f'  lr -> {lr_now:.2e}')

print('Training complete.')


# ── Final save ────────────────────────────────────────────────────────────────
save_path = 'quadrotor_control_net.pth'
torch.save({
    'model_state_dict'    : net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch'               : n_epochs,
    'alpha_terminal'      : _alpha,
    'loss_history'        : loss_history,
    'config': {
        'n_agent'     : n_agent,
        'STATE_DIM'   : STATE_DIM,
        'CONTROL_DIM' : CONTROL_DIM,
        'hidden_dim'  : hidden_dim,
        'n_blocks'    : n_blocks,
        'T_horizon'   : T_horizon,
        'dt'          : dt,
    },
}, save_path)
print(f'Model saved  ->  {save_path}')


# =============================================================================
# Results
# =============================================================================

# ── Training diagnostics ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 7))

ax = axes[0, 0]
ax.semilogy(loss_history, label='total', color='black', linewidth=1.5)
ax.semilogy(run_history,  label='running',  linestyle='--', linewidth=1)
ax.semilogy(term_history, label='terminal', linestyle=':',  linewidth=1)
ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(True, alpha=0.4)

ax = axes[0, 1]
ax.plot(proj_history, color='crimson', linewidth=1)
ax.set_title('CBF projection triggered (0/1)')
ax.set_xlabel('epoch'); ax.grid(True, alpha=0.4)

ax = axes[1, 0]
ax.plot(barrier_history, color='steelblue', linewidth=1)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='h=0')
ax.set_title('Min barrier value h (last batch)')
ax.set_xlabel('epoch'); ax.legend(); ax.grid(True, alpha=0.4)

ax = axes[1, 1]
ax.semilogy(grad_norm_history, color='darkorange', linewidth=1)
ax.set_title('Gradient norm'); ax.set_xlabel('epoch'); ax.grid(True, alpha=0.4)

plt.suptitle('Training diagnostics', fontsize=13)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 3-D trajectory plot ───────────────────────────────────────────────────────
def plot_3d(traj_np, obs_np, obs_r, eps, p_tgt,
            n_show=8, elev=25, azim=45, title='Quadrotor trajectories'):
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(range(n_agent))
    nt = traj_np.shape[2]

    for i in range(min(traj_np.shape[0], n_show)):
        for a in range(n_agent):
            b  = 12 * a
            xs = traj_np[i, b,   :]
            ys = traj_np[i, b+1, :]
            zs = traj_np[i, b+2, :]
            # fade trajectory from light to dark over time
            for k in range(nt - 1):
                alpha = 0.15 + 0.85 * k / nt
                ax.plot(xs[k:k+2], ys[k:k+2], zs[k:k+2],
                        color=colors[a], alpha=alpha, linewidth=1.2)
            ax.scatter(xs[0],  ys[0],  zs[0],  color=colors[a], s=30, marker='o')
            ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[a], s=60, marker='*')

    # Draw obstacle spheres
    u_s = np.linspace(0, 2 * np.pi, 24)
    v_s = np.linspace(0, np.pi,     24)
    U, V   = np.meshgrid(u_s, v_s)
    sx = np.sin(V) * np.cos(U)
    sy = np.sin(V) * np.sin(U)
    sz = np.cos(V)

    for c in obs_np:
        cx, cy, cz = c[0, 0], c[0, 1], c[0, 2]
        ax.plot_surface(cx + obs_r * sx,
                        cy + obs_r * sy,
                        cz + obs_r * sz,
                        color='red', alpha=0.35, linewidth=0)
        ax.plot_wireframe(cx + (obs_r + eps) * sx,
                          cy + (obs_r + eps) * sy,
                          cz + (obs_r + eps) * sz,
                          color='red', alpha=0.12, linewidth=0.4)

    tgt = p_tgt.detach().cpu().numpy()
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
               c='limegreen', s=140, marker='X', label='Target',
               zorder=10, edgecolors='black', linewidths=0.5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('traj_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


obs_np  = [c.cpu().numpy() for c in obstacle_centers]
traj_np = traj.cpu().numpy()

plot_3d(traj_np, obs_np, obstacle_radius, eps_safe, p_target, n_show=8)


# ── Multiple viewing angles (side by side) ────────────────────────────────────
views = [
    (30,  45,  'Isometric'),
    (90,   0,  'Top (XY)'),
    ( 0,   0,  'Front (XZ)'),
    ( 0,  90,  'Side (YZ)'),
]

fig = plt.figure(figsize=(16, 4))
colors = plt.cm.tab10(range(n_agent))
u_s = np.linspace(0, 2 * np.pi, 20)
v_s = np.linspace(0, np.pi,     20)
U, V = np.meshgrid(u_s, v_s)
sx, sy, sz = np.sin(V)*np.cos(U), np.sin(V)*np.sin(U), np.cos(V)

for idx, (elev, azim, ttl) in enumerate(views):
    ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
    for i in range(min(traj_np.shape[0], 6)):
        for a in range(n_agent):
            b = 12 * a
            ax.plot(traj_np[i, b, :], traj_np[i, b+1, :], traj_np[i, b+2, :],
                    color=colors[a], alpha=0.5, linewidth=1)

    for c in obs_np:
        cx, cy, cz = c[0, 0], c[0, 1], c[0, 2]
        ax.plot_surface(cx + obstacle_radius * sx,
                        cy + obstacle_radius * sy,
                        cz + obstacle_radius * sz,
                        color='red', alpha=0.3, linewidth=0)

    tgt = p_target.detach().cpu().numpy()
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
               c='limegreen', s=80, marker='X', zorder=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.tick_params(labelsize=7)

plt.suptitle('Trajectory — four viewpoints', fontsize=12)
plt.tight_layout()
plt.savefig('traj_multiview.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 2-D projections with obstacle circles ────────────────────────────────────
def plot_projections(traj_np, obs_np, obs_r, eps, p_tgt, n_show=8):
    views2d = [
        (0, 1, 'X [m]', 'Y [m]', 'Top view (XY)'),
        (0, 2, 'X [m]', 'Z [m]', 'Front view (XZ)'),
        (1, 2, 'Y [m]', 'Z [m]', 'Side view (YZ)'),
    ]
    colors = plt.cm.tab10(range(n_agent))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (ix, iy, xl, yl, ttl) in zip(axes, views2d):
        for i in range(min(traj_np.shape[0], n_show)):
            for a in range(n_agent):
                b = 12 * a
                ax.plot(traj_np[i, b+ix, :], traj_np[i, b+iy, :],
                        color=colors[a], alpha=0.5, linewidth=1)
                ax.scatter(traj_np[i, b+ix, 0], traj_np[i, b+iy, 0],
                           color=colors[a], s=20, marker='o')
                ax.scatter(traj_np[i, b+ix, -1], traj_np[i, b+iy, -1],
                           color=colors[a], s=40, marker='*')

        for c in obs_np:
            cx, cy = c[0, ix], c[0, iy]
            ax.add_patch(plt.Circle((cx, cy), obs_r,
                                    color='red', alpha=0.4))
            ax.add_patch(plt.Circle((cx, cy), obs_r + eps,
                                    fill=False, edgecolor='red',
                                    linewidth=1.2, linestyle='--'))

        tgt = p_tgt.detach().cpu().numpy()
        ax.scatter(tgt[:, ix], tgt[:, iy], c='limegreen',
                   s=80, marker='X', label='Target', zorder=5)

        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig('traj_projections.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_projections(traj_np, obs_np, obstacle_radius, eps_safe, p_target)


# ── Altitude, barrier, and DYS solver analysis ────────────────────────────────
t_arr = np.linspace(0, T_horizon, num_steps + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Altitude over time
ax = axes[0]
for i in range(min(traj_np.shape[0], 8)):
    for a in range(n_agent):
        ax.plot(t_arr, traj_np[i, 12*a + 2, :],
                color=plt.cm.tab10(a), alpha=0.5, linewidth=1)
ax.axhline(1.0, color='green', linestyle='--', linewidth=1.2, label='z=1 target')
ax.set_title('Altitude z(t)')
ax.set_xlabel('t [s]'); ax.set_ylabel('z [m]')
ax.legend(); ax.grid(True, alpha=0.4)

# Barrier history over training
ax = axes[1]
ax.plot(barrier_history, color='steelblue', linewidth=1)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='h=0')
ax.set_title('Min h over training')
ax.set_xlabel('epoch'); ax.set_ylabel('h_min')
ax.legend(); ax.grid(True, alpha=0.4)

# DYS iterations over training
ax = axes[2]
ax.plot(n_iters_history, color='darkorange', linewidth=1)
ax.set_title('Max DYS solver iters per epoch')
ax.set_xlabel('epoch'); ax.set_ylabel('iters')
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Interactive 3-D visualisation (requires plotly) ───────────────────────────
# Install with:  pip install plotly
try:
    import plotly.graph_objects as go

    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    ]
    n_show_pl = min(traj_np.shape[0], 6)
    plotly_fig = go.Figure()

    for i in range(n_show_pl):
        for a in range(n_agent):
            b  = 12 * a
            col = palette[a % len(palette)]
            plotly_fig.add_trace(go.Scatter3d(
                x=traj_np[i, b,   :].tolist(),
                y=traj_np[i, b+1, :].tolist(),
                z=traj_np[i, b+2, :].tolist(),
                mode='lines',
                line=dict(color=col, width=2),
                opacity=0.7,
                name=f'agent {a}' if i == 0 else None,
                showlegend=(i == 0),
            ))
            # mark end point
            plotly_fig.add_trace(go.Scatter3d(
                x=[traj_np[i, b,   -1]],
                y=[traj_np[i, b+1, -1]],
                z=[traj_np[i, b+2, -1]],
                mode='markers',
                marker=dict(symbol='cross', size=5, color=col),
                showlegend=False,
            ))

    # Obstacle spheres
    phi_g   = np.linspace(0, np.pi,     20)
    theta_g = np.linspace(0, 2*np.pi,  20)
    PH, TH  = np.meshgrid(phi_g, theta_g)
    ox = np.sin(PH) * np.cos(TH)
    oy = np.sin(PH) * np.sin(TH)
    oz = np.cos(PH)

    for c in obs_np:
        cx, cy, cz = c[0, 0], c[0, 1], c[0, 2]
        plotly_fig.add_trace(go.Surface(
            x=(cx + obstacle_radius * ox).tolist(),
            y=(cy + obstacle_radius * oy).tolist(),
            z=(cz + obstacle_radius * oz).tolist(),
            colorscale=[[0, 'red'], [1, 'red']],
            opacity=0.4, showscale=False, name='obstacle',
        ))

    tgt_np = p_target.detach().cpu().numpy()
    plotly_fig.add_trace(go.Scatter3d(
        x=tgt_np[:, 0].tolist(),
        y=tgt_np[:, 1].tolist(),
        z=tgt_np[:, 2].tolist(),
        mode='markers',
        marker=dict(symbol='cross', size=8, color='green'),
        name='target',
    ))

    plotly_fig.update_layout(
        title='Quadrotor trajectories — interactive',
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
        ),
        width=900, height=700,
    )
    plotly_fig.show()

except ImportError:
    print('plotly not installed — skipping interactive visualisation.')
    print('Install with:  pip install plotly')
