import torch
import matplotlib.pyplot as plt
import time
import os
import psutil
import argparse
import importlib
from utils import DYSProjector, ControlNetPerAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--problem", choices=["quadcopter_30_finetune", "quadcopter_100_finetune"], required=True)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--lr_decay", type=int, default=600)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grad_mode", choices=["jfb", "ad"], default="jfb")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--n_blocks", type=int, default=3)
parser.add_argument("--T", type=float, default=10.0)
parser.add_argument("--dt", type=float, default=0.2)
parser.add_argument("--alpha_terminal_final", type=float, default=10.0)
parser.add_argument("--use_schedule", action="store_true", default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
problem = args.problem
process = psutil.Process(os.getpid())

T = args.T
dt = args.dt
num_steps = int(T / dt)
alpha_running = 1
alpha_terminal = 2e1
alpha_terminal_final = args.alpha_terminal_final
log_every = 1
batch_size = 32
plot_freq = 50

import quadcopter_multi
quadcopter_multi.n_agent = 30 if problem == "quadcopter_30_finetune" else 100
importlib.reload(quadcopter_multi)

from quadcopter_multi import n_agent, CONTROL_DIM, f, sample_initial_condition, compute_loss, plot_trajectory

z0_std = 2e-2 if problem == "quadcopter_30_finetune" else 1e-2
obstacle_centers = [torch.tensor([0.63, 1.0, 1.0], device=device).view(1, 3), torch.tensor([1.5, 2.5, 0.8], device=device).view(1, 3), torch.tensor([2.37, 1.0, 1.0], device=device).view(1, 3)]
obstacle_radius = 0.35
eps_safe = 0.15

p_target = torch.zeros(n_agent, 3, device=device)
p_target[:, 0] = torch.linspace(0.2, 2.8, n_agent, device=device)
p_target[:, 1] = 3.5
p_target[:, 2] = 1.5

if not args.use_schedule:
    alpha_terminal = alpha_terminal_final

net = ControlNetPerAgent(agent_state_dim=12, hidden_dim=args.hidden_dim, control_dim=4, n_blocks=args.n_blocks, n_agent=n_agent, n_obs=len(obstacle_centers), T_total=T).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250, min_lr=1e-8)
proj = DYSProjector(grad_mode=args.grad_mode).to(device)
print('Number of parameters in control net:', sum(p.numel() for p in net.parameters()))

T_dev_scale = 0.5
tau_scale = 0.1

def u_fn(z, t):
    raw = net(z, t, p_target=p_target, obstacle_centers=obstacle_centers).reshape(z.shape[0], n_agent, 4)
    scaled = torch.cat([T_dev_scale * torch.tanh(raw[:, :, 0:1]), tau_scale * torch.tanh(raw[:, :, 1:4])], dim=-1)
    return scaled.reshape(z.shape[0], CONTROL_DIM)

loss_history = []
run_history = []
term_history = []
proj_history = []
best_loss = float("inf")
best_terminal_cost = float("inf")

for epoch in range(1, args.epochs + 1):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    optimizer.zero_grad()
    z0_sample = sample_initial_condition(batch_size, z0_std)
    total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj = compute_loss(u_fn, z0_sample, num_steps, f, p_target, obstacle_centers, obstacle_radius, eps_safe, alpha_running, alpha_terminal, proj, dt=dt)
    total_cost.backward()
    optimizer.step()
    scheduler.step(terminal_cost)

    total_grad_norm = 0.0
    for param in net.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    end_time = time.time()
    memory_MB = process.memory_info().rss / 1024 / 1024
    gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    loss_history.append(total_cost.item())
    run_history.append(running_cost.item())
    term_history.append(terminal_cost.item())
    proj_history.append(isprojected)

    if epoch % log_every == 0:
        print(f"epoch {epoch:4d} total={total_cost.item():.4e}  L={running_cost.item():.4e}  G={terminal_cost.item():.4e}  proj={int(isprojected)}  res={max_res_norm_array.max().item():.2e}  h={barrier_value_array.min().item():.2e}  iters={int(n_iters_array.mean())}  grad={total_grad_norm:.2e}  t={end_time - start_time:.2f}s  mem={memory_MB:.1f}MB  gpu_mem={gpu_max_memory_MB:.1f}MB")

    if args.plot and epoch % plot_freq == 0:
        print("  Plotting trajectory...")
        plot_trajectory(traj.cpu().numpy(), [c.cpu().numpy() for c in obstacle_centers], obstacle_radius, p_target, eps_safe=eps_safe)

    if epoch % 20 == 0 and args.use_schedule and alpha_terminal < alpha_terminal_final:
        alpha_terminal += 5
        print("new alpha_terminal:", alpha_terminal)

    if alpha_terminal >= alpha_terminal_final and total_cost.item() < best_loss:
        best_loss = total_cost.item()
        torch.save(net.state_dict(), f"best_control_net_{problem}.pth")
        print("save best model")

    if terminal_cost.item() < best_terminal_cost:
        best_terminal_cost = terminal_cost.item()
        torch.save(net.state_dict(), f"best_terminal_control_net_{problem}.pth")
        print(f"save best terminal model (G={best_terminal_cost:.4e})")

    if epoch % args.lr_decay == 0:
        if os.path.exists(f"best_control_net_{problem}.pth"):
            net.load_state_dict(torch.load(f"best_control_net_{problem}.pth"))
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            print("new lr:", param_group['lr'])

print("Training complete.")
torch.save(net.state_dict(), f"final_control_net_{problem}.pth")
print(f"Model saved to final_control_net_{problem}.pth")

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].semilogy(loss_history, label="total")
axes[0].semilogy(run_history, label="running", linestyle="--")
axes[0].semilogy(term_history, label="terminal", linestyle=":")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(proj_history)
axes[1].set_title("Projection triggered")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("0/1")
plt.tight_layout()
plt.savefig(f"training_curve_{problem}.png", bbox_inches="tight", dpi=400)