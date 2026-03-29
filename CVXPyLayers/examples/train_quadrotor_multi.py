"""
Example: Train CBF controller for multi-quadrotor system with 3D obstacles.

This demonstrates:
    - Multi-agent quadrotor dynamics (12-state per agent)
    - HOCBF (Higher-Order CBF, relative degree 2)
    - 3D spherical obstacles
    - Non-linear dynamics with Euler angles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import json
from pathlib import Path

# Import from parent repo
from quadcopter_multi import (
    n_agent, m, g, T_hover, STATE_DIM, CONTROL_DIM,
    split_state, thrust_direction, f, lagrangian, G,
    barrier_function, evaluate_barriers, psi1_function, evaluate_psi1,
    construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
)
from utils import DYSProjector, ControlNet


def main():
    # Parse arguments (matching train.py interface)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=int, default=600)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========================================================================
    # Configuration
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING: Multi-Quadrotor + 3D Obstacles (HOCBF)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Number of agents: {n_agent}")
    print(f"State dim per agent: 12 (position, angles, velocities, angular rates)")
    print(f"Control dim per agent: 4 (thrust, angular accelerations)")
    print(f"Total state dim: {STATE_DIM}")
    print(f"Total control dim: {CONTROL_DIM}")

    # Time parameters
    T = 10.0
    dt = 0.2
    num_steps = int(T / dt)
    print(f"\nTime horizon: T={T}s, dt={dt}s, steps={num_steps}")

    # Obstacles (3D spherical)
    obstacle_center_1 = torch.tensor([0.63, 1.0, 1.0]).view(1, 3).to(device)
    obstacle_center_2 = torch.tensor([1.5, 2.5, 0.8]).view(1, 3).to(device)
    obstacle_center_3 = torch.tensor([2.37, 1.0, 1.0]).view(1, 3).to(device)
    obstacle_centers = [obstacle_center_1, obstacle_center_2, obstacle_center_3]
    obstacle_radius = 0.35
    eps_safe = 0.15

    print(f"\nObstacles: {len(obstacle_centers)}")
    for i, center in enumerate(obstacle_centers):
        print(f"  [{i+1}] center={center.squeeze().tolist()}, radius={obstacle_radius}, eps={eps_safe}")

    # Target positions (agents in a line at y=3.5, z=1.0)
    p_target = torch.zeros(n_agent, 3, device=device)
    x_min, x_max = 1.1, 1.9
    x_positions = torch.linspace(x_min, x_max, n_agent, device=device)
    p_target[:, 0] = x_positions
    p_target[:, 1] = 3.5
    p_target[:, 2] = 1.0
    print(f"\nTarget positions: line from x={x_min} to x={x_max}, y=3.5, z=1.0")

    # Cost weights (matching train.py exactly)
    alpha_running = 1
    alpha_terminal = 2e1
    weight_decay = 1e-3
    print(f"\nCost weights: running={alpha_running}, terminal={alpha_terminal}")
    print(f"Weight decay: {weight_decay}")

    # Training parameters (from args, matching train.py)
    num_epochs = args.epochs
    learning_rate = args.lr
    lr_decay_epoch = args.lr_decay
    batch_size = 32
    z0_std = 4e-2
    log_every = 1
    plot_freq = 50

    print(f"\nTraining: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"LR decay at epoch {lr_decay_epoch}")
    print(f"Initial state std: {z0_std}")

    # Network architecture (from args, matching train.py)
    hidden_dim = args.hidden_dim
    n_blocks = args.n_blocks
    input_dim = STATE_DIM + 1  # state + time
    output_dim = CONTROL_DIM

    print(f"\nNetwork: {input_dim} → {hidden_dim}x{n_blocks} → {output_dim}")

    # ========================================================================
    # Initialize network, optimizer, projector
    # ========================================================================
    net = ControlNet(input_dim=input_dim, hidden_dim=hidden_dim,
                     output_dim=output_dim, n_blocks=n_blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    proj = DYSProjector().to(device)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Save directory setup
    save_dir = Path("./models")
    save_dir.mkdir(exist_ok=True)

    # Save configuration
    config_dict = {
        'n_agent': n_agent,
        'STATE_DIM': STATE_DIM,
        'CONTROL_DIM': CONTROL_DIM,
        'T': T,
        'dt': dt,
        'num_steps': num_steps,
        'obstacle_centers': [c.squeeze().tolist() for c in obstacle_centers],
        'obstacle_radius': obstacle_radius,
        'eps_safe': eps_safe,
        'p_target': p_target.tolist(),
        'alpha_running': alpha_running,
        'alpha_terminal_initial': alpha_terminal,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'lr_decay_epoch': lr_decay_epoch,
        'batch_size': batch_size,
        'z0_std': z0_std,
        'hidden_dim': hidden_dim,
        'n_blocks': n_blocks,
        'T_dev_scale': 0.5,
        'tau_scale': 0.1,
        'num_params': num_params,
        'seed': args.seed,
    }
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {save_dir / 'config.json'}")

    # Control scaling (thrust deviation and angular acceleration limits)
    T_dev_scale = 0.5    # max thrust deviation from hover
    tau_scale = 0.1      # max angular acceleration

    def u_fn(z, t):
        """Neural network controller with output scaling."""
        raw = net(z, t).reshape(z.shape[0], n_agent, 4)
        scaled = torch.cat([
            T_dev_scale * torch.tanh(raw[:, :, 0:1]),
            tau_scale * torch.tanh(raw[:, :, 1:4]),
        ], dim=-1)
        return scaled.reshape(z.shape[0], CONTROL_DIM)

    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    loss_history = []
    run_history = []
    term_history = []
    proj_history = []
    n_iters_history = []
    max_res_norm_history = []
    barrier_function_history = []
    grad_norm_history = []

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        optimizer.zero_grad()

        # Sample initial conditions
        z0_sample = sample_initial_condition(batch_size, z0_std)
        assert z0_sample.shape == (batch_size, STATE_DIM)

        # Compute loss
        total_cost, running_cost, terminal_cost, isprojected, \
        n_iters_array, max_res_norm_array, barrier_value_array, traj = compute_loss(
            u_fn, z0_sample, num_steps, f, p_target, obstacle_centers,
            obstacle_radius, eps_safe, alpha_running, alpha_terminal, proj, dt=dt
        )

        # Backward pass
        total_cost.backward()
        optimizer.step()

        # Compute gradient norm
        total_grad_norm = 0.0
        for param in net.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        end_time = time.time()

        # Record history
        loss_history.append(total_cost.item())
        run_history.append(running_cost.item())
        term_history.append(terminal_cost.item())
        proj_history.append(isprojected)
        n_iters_history.append(n_iters_array.max().item())
        max_res_norm_history.append(max_res_norm_array.max().item())
        barrier_function_history.append(barrier_value_array.min().item())
        grad_norm_history.append(total_grad_norm)

        # Save best model
        if total_cost.item() < best_loss:
            best_loss = total_cost.item()
            torch.save(net.state_dict(), save_dir / "best_model.pth")
            if epoch % log_every == 0:
                print(f"  → New best model saved! loss={best_loss:.4e}")

        # Logging
        if epoch % log_every == 0:
            print(f"epoch {epoch:4d} total={total_cost.item():.4e}"
                  f"  L={running_cost.item():.4e}"
                  f"  G={terminal_cost.item():.4e}"
                  f"  proj={int(isprojected)}"
                  f"  res={max_res_norm_array.max().item():.2e}"
                  f"  h={barrier_value_array.min().item():.2e}"
                  f"  iters={int(n_iters_array.mean())}"
                  f"  grad={total_grad_norm:.2e}"
                  f"  t={end_time - start_time:.2f}s")

        # Plot trajectory
        if epoch % plot_freq == 0:
            print("  Plotting trajectory...")
            plot_trajectory(traj.cpu().numpy(),
                          [c.cpu().numpy() for c in obstacle_centers],
                          obstacle_radius, p_target, eps_safe=eps_safe)

        # Increase terminal cost weight
        if epoch % 20 == 0:
            if alpha_terminal <= 2e2:
                alpha_terminal += 5
                print(f"  → Updated alpha_terminal: {alpha_terminal}")

        # Learning rate decay
        if epoch % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"  → Updated learning rate: {param_group['lr']}")

    # ========================================================================
    # Save model and results
    # ========================================================================
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)

    # Save final model
    final_model_path = save_dir / "final_model.pth"
    torch.save(net.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {save_dir / 'best_model.pth'} (loss={best_loss:.4e})")

    # Save training history as CSV
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'total_loss': loss_history,
        'running_cost': run_history,
        'terminal_cost': term_history,
        'projection_triggered': proj_history,
        'max_projection_iters': n_iters_history,
        'max_residual_norm': max_res_norm_history,
        'min_barrier_value': barrier_function_history,
        'gradient_norm': grad_norm_history,
    })
    history_csv_path = save_dir / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to: {history_csv_path}")

    # Plot training curves
    print("\nPlotting training curves...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Loss curves
    axes[0, 0].semilogy(loss_history, label="total", linewidth=2)
    axes[0, 0].semilogy(run_history, label="running", linestyle="--")
    axes[0, 0].semilogy(term_history, label="terminal", linestyle=":")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Projection triggered
    axes[0, 1].plot(proj_history, linewidth=1)
    axes[0, 1].set_title("Projection triggered")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("0/1")
    axes[0, 1].grid(True, alpha=0.3)

    # Barrier function minimum
    axes[0, 2].plot(barrier_function_history, linewidth=1)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', label='Safety boundary')
    axes[0, 2].set_title("Min barrier value (safety)")
    axes[0, 2].set_xlabel("epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Projection iterations
    axes[1, 0].plot(n_iters_history, linewidth=1)
    axes[1, 0].set_title("Max projection iterations")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(True, alpha=0.3)

    # Residual norm
    axes[1, 1].semilogy(max_res_norm_history, linewidth=1)
    axes[1, 1].set_title("Max residual norm")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.3)

    # Gradient norm
    axes[1, 2].semilogy(grad_norm_history, linewidth=1)
    axes[1, 2].set_title("Gradient norm")
    axes[1, 2].set_xlabel("epoch")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = "./models/quadrotor_training_curves.png"
    plt.savefig(curve_path, bbox_inches="tight", dpi=300)
    print(f"Training curves saved to: {curve_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
