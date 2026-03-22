import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import quadrotor_multi_50 as _qm
from quadrotor_multi_50 import (
    CONTROL_DIM,
    STATE_DIM,
    TARGET_RADIUS,
    TARGET_TYPE,
    T_hover,
    build_target_positions,
    compute_loss,
    f,
    n_agent,
    sample_initial_condition,
)
from utils import ControlNet, DYSProjector


device = 'cuda'
if device.startswith('cuda') and not torch.cuda.is_available():
    device = 'cpu'
if device == 'cuda':
    device = 'cuda:0'

_qm.device = device

seed = 7
torch.manual_seed(seed)
np.random.seed(seed)

results_dir = Path('results_quadrotor_50')
results_dir.mkdir(exist_ok=True)

T_horizon = 12.0
dt = 0.1
num_steps = int(T_horizon / dt)

alpha_running = 1.0
alpha_terminal = 120.0

obstacle_cfg = [
    [1.0, 1.2, 1.0],   # left
    [2.0, 1.2, 1.0],   # right
    [1.5, 2.0, 1.0],   # top-center
]
obstacle_radius = 0.28
eps_safe = 0.10

n_epochs = 5000
learning_rate = 1e-4
lr_decay = 600
batch_size = 32
z0_std = 0.05
log_every = 5
plot_every = 100
save_ckpt_every = 250

hidden_dim = 512
n_blocks = 5
T_dev_scale = 0.5
tau_scale = 0.12

print(f'n_agent      : {n_agent}')
print(f'STATE_DIM    : {STATE_DIM}')
print(f'CONTROL_DIM  : {CONTROL_DIM}')
print(f'Device       : {device}')
print(f'Horizon      : {T_horizon}s')
print(f'Batch size   : {batch_size}')
print(f'Target type  : {TARGET_TYPE}')
if device.startswith('cuda'):
    gpu_idx = torch.device(device).index or 0
    print(f'GPU          : {torch.cuda.get_device_name(gpu_idx)}')

obstacle_centers = [
    torch.tensor(c, dtype=torch.float32).view(1, 3).to(device)
    for c in obstacle_cfg
]
p_target = build_target_positions(device)

# Debug: print target positions to verify
print(f'Target positions (first 5):')
for i in range(min(5, n_agent)):
    print(f'  Agent {i}: x={p_target[i,0]:.3f}, y={p_target[i,1]:.3f}, z={p_target[i,2]:.3f}')
print(f'  ...')
print(f'  Agent {n_agent-1}: x={p_target[-1,0]:.3f}, y={p_target[-1,1]:.3f}, z={p_target[-1,2]:.3f}')

net = ControlNet(
    input_dim=STATE_DIM + 1,
    hidden_dim=hidden_dim,
    output_dim=CONTROL_DIM,
    n_blocks=n_blocks,
).to(device)
proj = DYSProjector().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def u_fn(z, t):
    raw = net(z, t).reshape(z.shape[0], n_agent, 4)
    scaled = torch.cat(
        [
            T_dev_scale * torch.tanh(raw[:, :, 0:1]),
            tau_scale * torch.tanh(raw[:, :, 1:4]),
        ],
        dim=-1,
    )
    return scaled.reshape(z.shape[0], CONTROL_DIM)


def save_environment_snapshot(path):
    with torch.no_grad():
        z0 = sample_initial_condition(batch_size=1, z0_std=0.0).cpu().numpy()[0]
    tgt = p_target.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ring_colors = plt.cm.hsv(np.linspace(0, 1, n_agent, endpoint=False))

    # Plot start positions
    for a in range(n_agent):
        idx = 12 * a
        ax.scatter(z0[idx], z0[idx + 1], s=20, color=ring_colors[a], alpha=0.7, marker='o', label='Start' if a == 0 else '')

    # Plot target positions
    for a in range(n_agent):
        ax.scatter(tgt[a, 0], tgt[a, 1], s=30, marker='X', color=ring_colors[a], alpha=0.9, edgecolor='black', linewidth=0.5)

    # If horizontal line, draw connecting line to show formation
    from quadrotor_multi_50 import TARGET_TYPE
    if TARGET_TYPE == 'horizontal_line':
        ax.plot(tgt[:, 0], tgt[:, 1], 'g--', linewidth=2, alpha=0.5, label='Target line')
    for c in obstacle_cfg:
        ax.add_patch(plt.Circle((c[0], c[1]), obstacle_radius, color='red', alpha=0.35))
        ax.add_patch(
            plt.Circle(
                (c[0], c[1]),
                obstacle_radius + eps_safe,
                fill=False,
                edgecolor='red',
                linestyle='--',
                linewidth=1.0,
            )
        )
    ax.set_title(f'50-rotor start and target placement ({TARGET_TYPE})')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_training_curves(loss_history, run_history, term_history, barrier_history, grad_norm_history, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].semilogy(loss_history, label='total', color='black')
    axes[0, 0].semilogy(run_history, label='running', linestyle='--')
    axes[0, 0].semilogy(term_history, label='terminal', linestyle=':')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Loss')

    axes[0, 1].plot(barrier_history, color='steelblue')
    axes[0, 1].axhline(0.0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Min barrier')

    axes[1, 0].semilogy(grad_norm_history, color='darkorange')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Gradient norm')

    steps = np.arange(len(loss_history))
    axes[1, 1].plot(steps, np.array(loss_history) / np.maximum(1, steps + 1), color='purple')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Smoothed total loss proxy')

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def save_trajectory_projection(traj_np, path, n_show=2):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 8))
    ring_colors = plt.cm.tab10(range(n_agent))
    tgt = p_target.detach().cpu().numpy()
    nt = traj_np.shape[2]

    # 3D view (left panel)
    ax3d = fig.add_subplot(121, projection='3d')
    for i in range(min(n_show, traj_np.shape[0])):
        for a in range(n_agent):
            idx = 12 * a
            xs = traj_np[i, idx, :]
            ys = traj_np[i, idx + 1, :]
            zs = traj_np[i, idx + 2, :]

            # Fade trajectory from light to dark over time
            for k in range(nt - 1):
                alpha = 0.15 + 0.85 * k / nt
                ax3d.plot(xs[k:k+2], ys[k:k+2], zs[k:k+2],
                         color=ring_colors[a % 10], alpha=alpha, linewidth=1.2)

            # Start and end markers
            ax3d.scatter(xs[0], ys[0], zs[0], color=ring_colors[a % 10], s=30, marker='o')
            ax3d.scatter(xs[-1], ys[-1], zs[-1], color=ring_colors[a % 10], s=60, marker='*')

    # Target positions
    ax3d.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c='limegreen', marker='X', s=140,
                label='Target', zorder=10, edgecolors='black', linewidths=0.5)

    # 3D obstacle spheres with high-quality rendering
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u, v)
    x_sphere = np.sin(V) * np.cos(U)
    y_sphere = np.sin(V) * np.sin(U)
    z_sphere = np.cos(V)

    for c in obstacle_cfg:
        # Solid obstacle
        ax3d.plot_surface(c[0] + obstacle_radius * x_sphere,
                         c[1] + obstacle_radius * y_sphere,
                         c[2] + obstacle_radius * z_sphere,
                         color='red', alpha=0.4, linewidth=0,
                         shade=True, antialiased=True)
        # Safety margin wireframe
        ax3d.plot_wireframe(c[0] + (obstacle_radius + eps_safe) * x_sphere,
                           c[1] + (obstacle_radius + eps_safe) * y_sphere,
                           c[2] + (obstacle_radius + eps_safe) * z_sphere,
                           color='red', alpha=0.15, linewidth=0.5, linestyle='--')

    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('3D Trajectory')
    ax3d.set_box_aspect([1, 1, 0.5])  # Equal aspect for X,Y; shorter Z
    ax3d.view_init(elev=25, azim=45)
    ax3d.legend()

    # Top view XY (right panel)
    ax2d = fig.add_subplot(122)
    for i in range(min(n_show, traj_np.shape[0])):
        for a in range(n_agent):
            idx = 12 * a
            ax2d.plot(traj_np[i, idx, :], traj_np[i, idx + 1, :], color=ring_colors[a], alpha=0.45, linewidth=0.8)
            # Start position
            ax2d.scatter(traj_np[i, idx, 0], traj_np[i, idx + 1, 0],
                        color=ring_colors[a], s=20, marker='o', alpha=0.7)

    ax2d.scatter(tgt[:, 0], tgt[:, 1], c='limegreen', marker='X', s=40, label='target')

    for c in obstacle_cfg:
        ax2d.add_patch(plt.Circle((c[0], c[1]), obstacle_radius, color='red', alpha=0.35))
        ax2d.add_patch(
            plt.Circle(
                (c[0], c[1]),
                obstacle_radius + eps_safe,
                fill=False,
                edgecolor='red',
                linestyle='--',
                linewidth=1.0,
            )
        )
        ax2d.text(c[0], c[1], f'obs\nr={obstacle_radius:.2f}',
                 ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax2d.set_title('Top view (XY)')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


save_environment_snapshot(results_dir / 'placement_50.png')

# Generate epoch 0 trajectory to visualize initial (untrained) behavior
print('Generating epoch 0 trajectory visualization...')
net.eval()
with torch.no_grad():
    z0_init = sample_initial_condition(batch_size=2, z0_std=0.0)
    _, _, _, _, _, _, _, traj_init = compute_loss(
        u_fn,
        z0_init,
        num_steps,
        f,
        p_target,
        obstacle_centers,
        obstacle_radius,
        eps_safe,
        alpha_running,
        alpha_terminal,
        proj,
        dt=dt,
    )
    save_trajectory_projection(
        traj_init.detach().cpu().numpy(),
        results_dir / 'traj_xy_epoch_0000.png',
    )
print(f'Epoch 0 visualization saved to {results_dir / "traj_xy_epoch_0000.png"}')

loss_history = []
run_history = []
term_history = []
barrier_history = []
grad_norm_history = []
n_iters_history = []
res_history = []

best_loss = float('inf')
best_ckpt_path = results_dir / 'quadrotor_control_net_50_best.pth'
traj = None

net.train()
for epoch in range(1, n_epochs + 1):
    t0 = time.time()
    optimizer.zero_grad()

    z0_sample = sample_initial_condition(batch_size=batch_size, z0_std=z0_std)
    total_cost, running_cost, terminal_cost, _, n_iters_array, max_res_array, barrier_array, traj = compute_loss(
        u_fn,
        z0_sample,
        num_steps,
        f,
        p_target,
        obstacle_centers,
        obstacle_radius,
        eps_safe,
        alpha_running,
        alpha_terminal,
        proj,
        dt=dt,
    )

    total_cost.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()

    grad_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in net.parameters()
        if p.grad is not None
    ) ** 0.5

    loss_history.append(total_cost.item())
    run_history.append(running_cost.item())
    term_history.append(terminal_cost.item())
    barrier_history.append(barrier_array.min().item())
    grad_norm_history.append(grad_norm)
    n_iters_history.append(n_iters_array.max().item())
    res_history.append(max_res_array.max().item())

    if epoch % log_every == 0 or epoch == 1:
        print(
            f'ep {epoch:4d} | total={total_cost.item():.3e} '
            f'run={running_cost.item():.3e} term={terminal_cost.item():.3e} '
            f'iters={int(n_iters_array.max().item()):4d} res={max_res_array.max().item():.2e} '
            f'h_min={barrier_array.min().item():.2e} grad={grad_norm:.2e} '
            f't={time.time() - t0:.1f}s'
        )

    if total_cost.item() < best_loss:
        best_loss = total_cost.item()
        torch.save(
            {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'config': {
                    'n_agent': n_agent,
                    'STATE_DIM': STATE_DIM,
                    'CONTROL_DIM': CONTROL_DIM,
                    'hidden_dim': hidden_dim,
                    'n_blocks': n_blocks,
                    'T_horizon': T_horizon,
                    'dt': dt,
                    'batch_size': batch_size,
                    'obstacle_cfg': obstacle_cfg,
                    'obstacle_radius': obstacle_radius,
                    'eps_safe': eps_safe,
                },
            },
            best_ckpt_path,
        )

    if epoch % lr_decay == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        print(f"  lr -> {optimizer.param_groups[0]['lr']:.2e}")

    if epoch % save_ckpt_every == 0:
        ckpt_path = results_dir / f'quadrotor_control_net_50_epoch_{epoch:04d}.pth'
        torch.save(
            {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': total_cost.item(),
                'loss_history': loss_history,
                'config': {
                    'n_agent': n_agent,
                    'STATE_DIM': STATE_DIM,
                    'CONTROL_DIM': CONTROL_DIM,
                    'hidden_dim': hidden_dim,
                    'n_blocks': n_blocks,
                    'T_horizon': T_horizon,
                    'dt': dt,
                    'batch_size': batch_size,
                    'obstacle_cfg': obstacle_cfg,
                    'obstacle_radius': obstacle_radius,
                    'eps_safe': eps_safe,
                },
            },
            ckpt_path,
        )
        print(f'  Checkpoint saved -> {ckpt_path}')

    if epoch % plot_every == 0:
        save_training_curves(
            loss_history,
            run_history,
            term_history,
            barrier_history,
            grad_norm_history,
            results_dir / 'training_curves_50.png',
        )
        if traj is not None:
            save_trajectory_projection(
                traj.detach().cpu().numpy(),
                results_dir / f'traj_xy_epoch_{epoch:04d}.png',
            )

torch.save(
    {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': n_epochs,
        'loss_history': loss_history,
    },
    results_dir / 'quadrotor_control_net_50_final.pth',
)

save_training_curves(
    loss_history,
    run_history,
    term_history,
    barrier_history,
    grad_norm_history,
    results_dir / 'training_curves_50.png',
)
if traj is not None:
    save_trajectory_projection(traj.detach().cpu().numpy(), results_dir / 'traj_xy_final.png')

np.save(results_dir / 'n_iters_history.npy', np.asarray(n_iters_history))
np.save(results_dir / 'residual_history.npy', np.asarray(res_history))

print('Training complete.')
print(f'Best checkpoint: {best_ckpt_path}')
print(f'Results dir    : {results_dir}')
