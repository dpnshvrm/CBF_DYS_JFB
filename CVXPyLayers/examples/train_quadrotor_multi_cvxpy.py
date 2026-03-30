"""
Example: Train CBF controller for multi-quadrotor system using CVXPyLayers.

This demonstrates:
    - Multi-agent quadrotor dynamics (12-state per agent)
    - HOCBF (Higher-Order CBF, relative degree 2)
    - 3D spherical obstacles
    - CVXPyLayers-based differentiable QP solver
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import pandas as pd
import json
from pathlib import Path

from dynamics import Quadrotor
from barriers import SphericalObstacle
from controllers import CBFQPController, PolicyNetwork
import torch.optim as optim
from tqdm import tqdm


def train_quadrotor(args):
    """Train quadrotor CBF controller using CVXPyLayers."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    torch.manual_seed(args.seed)

    # ========================================================================
    # Configuration
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING: Multi-Quadrotor + 3D Obstacles (CVXPyLayers + HOCBF)")
    print("="*70)

    # Quadrotor parameters (matching train.py setup)
    n_agent = 1
    mass = 0.5
    gravity = 1.0
    T_hover = mass * gravity

    print(f"Device: {device}")
    print(f"Number of agents: {n_agent}")
    print(f"Mass: {mass}, Gravity: {gravity}, Hover thrust: {T_hover}")

    # Create dynamics
    dynamics = Quadrotor(n_agent=n_agent, mass=mass, gravity=gravity)
    print(f"\nDynamics: {dynamics}")
    print(f"State dim: {dynamics.state_dim}, Control dim: {dynamics.control_dim}")
    print(f"Relative degree: {dynamics.relative_degree}")

    # Time parameters
    T = 10.0
    dt = 0.2
    num_steps = int(T / dt)
    print(f"\nTime horizon: T={T}s, dt={dt}s, steps={num_steps}")

    # Obstacles (3D spherical)
    obstacle_configs = [
        {'center': [0.63, 1.0, 1.0], 'radius': 0.35, 'epsilon': 0.15},
        {'center': [1.5, 2.5, 0.8], 'radius': 0.35, 'epsilon': 0.15},
        {'center': [2.37, 1.0, 1.0], 'radius': 0.35, 'epsilon': 0.15},
    ]

    obstacles = []
    for obs_config in obstacle_configs:
        obs = SphericalObstacle(
            center=obs_config['center'],
            radius=obs_config['radius'],
            epsilon=obs_config['epsilon'],
            dynamics=dynamics,
            n_agent=n_agent
        ).to(device)
        obstacles.append(obs)

    print(f"\nObstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"  [{i+1}] {obs}")

    # Target positions (agents in a line at y=3.5, z=1.0)
    # Match train.py: x_min=1.1, x_max=1.9 for 5 agents
    # For 10 agents, adjust range to spread them out more
    x_min, x_max = 0.8, 2.2
    x_positions = torch.linspace(x_min, x_max, n_agent, device=device, dtype=dtype)
    p_target = torch.zeros(n_agent, 3, device=device, dtype=dtype)
    p_target[:, 0] = x_positions
    p_target[:, 1] = 3.5
    p_target[:, 2] = 1.0

    # Target state: [pos, angles=0, vel=0, ang_vel=0]
    target_state = torch.zeros(dynamics.state_dim, device=device, dtype=dtype)
    for i in range(n_agent):
        target_state[12*i:12*i+3] = p_target[i]  # Set target positions

    print(f"\nTarget positions: line from x={x_min} to x={x_max}, y=3.5, z=1.0")

    # CBF parameters (HOCBF)
    cbf_alpha = (5.0, 5.0)  # (alpha1, alpha2)
    print(f"\nCBF alpha: {cbf_alpha}")

    # Create CBF-QP controller
    cbf_controller = CBFQPController(
        dynamics=dynamics,
        obstacles=obstacles,
        alpha=cbf_alpha
    )
    print(f"\nCBF Controller: {cbf_controller}")

    # Cost weights - VERY conservative schedule for CVXPyLayers
    alpha_running = 1.0
    alpha_terminal = 20.0  # Start at 20
    alpha_terminal_final = 30.0  # Lower max (CVXPyLayers is sensitive)
    alpha_sched_step = 5.0  # Increase by 5
    alpha_sched_every = 100  # Much slower: every 100 epochs (was 20)
    weight_decay = 1e-3
    print(f"\nCost weights: running={alpha_running}, terminal={alpha_terminal} (initial)")
    print(f"Alpha terminal schedule: +{alpha_sched_step} every {alpha_sched_every} epochs, max={alpha_terminal_final}")

    # Training parameters - Conservative settings
    num_epochs = args.epochs
    learning_rate = args.lr if args.lr != 0.001 else 1e-4  # Conservative LR
    lr_decay_epoch = args.lr_decay
    batch_size = 32
    z0_std = 4e-2  # Match train.py
    log_every = 1
    plot_freq = 100

    print(f"\nTraining: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"LR decay at epoch {lr_decay_epoch}")

    # Create policy network
    policy = PolicyNetwork(
        state_dim=dynamics.state_dim,
        control_dim=dynamics.control_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.n_blocks,
        activation='silu'
    ).to(device).to(dtype)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy network parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Save directory (inside CVXPyLayers/models/<script_name>/)
    script_name = Path(__file__).stem  # e.g., "train_quadrotor_multi_cvxpy"
    save_dir = Path(__file__).parent.parent / "models" / script_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = {
        'n_agent': n_agent,
        'mass': mass,
        'gravity': gravity,
        'state_dim': dynamics.state_dim,
        'control_dim': dynamics.control_dim,
        'T': T,
        'dt': dt,
        'num_steps': num_steps,
        'obstacles': obstacle_configs,
        'p_target': p_target.tolist(),
        'cbf_alpha': cbf_alpha,
        'alpha_running': alpha_running,
        'alpha_terminal_initial': alpha_terminal,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'lr_decay_epoch': lr_decay_epoch,
        'batch_size': batch_size,
        'z0_std': z0_std,
        'hidden_dim': args.hidden_dim,
        'n_blocks': args.n_blocks,
        'num_params': num_params,
        'seed': args.seed,
        'framework': 'CVXPyLayers',
    }
    with open(save_dir / "config_cvxpy.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {save_dir / 'config_cvxpy.json'}")

    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    loss_history = []
    run_history = []
    term_history = []
    barrier_history = []
    grad_norm_history = []
    alpha_terminal_history = []
    best_loss = float('inf')

    # Mutable alpha_terminal for scheduling
    _alpha_terminal = alpha_terminal

    # Initial state (agents in a line at y=-0.5, z=1.0)
    initial_state = torch.zeros(dynamics.state_dim, device=device, dtype=dtype)
    for i in range(n_agent):
        initial_state[12*i] = x_positions[i]  # x position
        initial_state[12*i+1] = -0.5  # y position
        initial_state[12*i+2] = 1.0  # z position

    # Create directory for trajectory plots
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        optimizer.zero_grad()

        # Sample initial conditions (add noise)
        z0_batch = initial_state.unsqueeze(0).repeat(batch_size, 1)
        # Add noise to xy positions only
        for i in range(n_agent):
            z0_batch[:, 12*i:12*i+2] += z0_std * torch.randn(batch_size, 2, device=device, dtype=dtype)

        # Forward rollout (store trajectory for plotting)
        running_cost = 0.0
        x = z0_batch

        # Store trajectory if we're going to plot this epoch
        if epoch % plot_freq == 0:
            traj = torch.zeros(batch_size, dynamics.state_dim, num_steps + 1, device=device, dtype=dtype)
            traj[:, :, 0] = z0_batch

        for t_step in range(num_steps):
            # Policy outputs desired control
            u_desired = policy(x)

            # Add hover bias
            hover_bias = torch.zeros(dynamics.control_dim, device=device, dtype=dtype)
            hover_bias[::4] = T_hover  # Every 4th element is thrust
            u_desired = u_desired + hover_bias

            # Filter through CBF-QP
            try:
                u_safe = cbf_controller.filter_control(x, u_desired)
            except Exception as e:
                print(f"\nCBF-QP failed at epoch {epoch}, step {t_step}: {e}")
                u_safe = u_desired  # Fallback to desired control

            # Step dynamics
            x = dynamics.step(x, u_safe, dt)

            # Store trajectory
            if epoch % plot_freq == 0:
                traj[:, :, t_step + 1] = x

            # Running cost
            running_cost += dt * 0.5 * (u_safe ** 2).sum(dim=-1).mean()

        # Terminal cost
        position_error = (x - target_state).pow(2).sum(dim=-1).mean()
        terminal_cost = position_error

        # Total cost (use _alpha_terminal for scheduling)
        total_cost = alpha_running * running_cost + _alpha_terminal * terminal_cost

        # Backward
        total_cost.backward()

        # Gradient norm
        grad_norm = 0.0
        for param in policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # Record history
        loss_history.append(total_cost.item())
        run_history.append(running_cost.item())
        term_history.append(terminal_cost.item())
        grad_norm_history.append(grad_norm)
        alpha_terminal_history.append(_alpha_terminal)

        # Compute min barrier value
        min_barrier = float('inf')
        for obs in obstacles:
            h_vals = obs.h(x)  # (batch, n_agent)
            min_barrier = min(min_barrier, h_vals.min().item())
        barrier_history.append(min_barrier)

        # Save best model
        if total_cost.item() < best_loss:
            best_loss = total_cost.item()
            torch.save(policy.state_dict(), save_dir / "best_model_cvxpy.pth")

        # Logging
        if epoch % log_every == 0:
            tqdm.write(f"epoch {epoch:4d} total={total_cost.item():.4e} "
                      f"L={running_cost.item():.4e} G={terminal_cost.item():.4e} "
                      f"alpha_t={_alpha_terminal:.1f} h_min={min_barrier:.2e} grad={grad_norm:.2e}")

        # Plotting
        if epoch % plot_freq == 0:
            tqdm.write(f"  Plotting trajectory at epoch {epoch}...")
            plot_path = plot_dir / f"traj_epoch_{epoch:04d}.png"
            dynamics.plot_trajectory(
                traj=traj,
                obstacles=obstacles,
                p_target=p_target,
                save_path=str(plot_path),
                title=f"Epoch {epoch} - Quadrotor Trajectories"
            )
            tqdm.write(f"  Saved to: {plot_path}")

        # Update terminal cost weight (Conservative schedule from README)
        if epoch % alpha_sched_every == 0 and _alpha_terminal < alpha_terminal_final:
            _alpha_terminal += alpha_sched_step
            tqdm.write(f"  → alpha_terminal: {_alpha_terminal:.1f}")

        # Learning rate decay
        if epoch % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                tqdm.write(f"  → learning rate: {param_group['lr']:.2e}")

    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)

    # Save final model
    torch.save(policy.state_dict(), save_dir / "final_model_cvxpy.pth")
    print(f"Final model saved to: {save_dir / 'final_model_cvxpy.pth'}")
    print(f"Best model saved to: {save_dir / 'best_model_cvxpy.pth'} (loss={best_loss:.4e})")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'total_loss': loss_history,
        'running_cost': run_history,
        'terminal_cost': term_history,
        'alpha_terminal': alpha_terminal_history,
        'min_barrier_value': barrier_history,
        'gradient_norm': grad_norm_history,
    })
    history_csv = save_dir / "training_history_cvxpy.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"Training history saved to: {history_csv}")

    # Generate final trajectory plot with best model
    print("\n" + "="*70)
    print("Generating final trajectory visualization...")
    print("="*70)

    # Rollout with best model (no gradient)
    with torch.no_grad():
        z0_batch = initial_state.unsqueeze(0).repeat(1, 1)  # Single trajectory, no noise
        x = z0_batch
        traj_final = torch.zeros(1, dynamics.state_dim, num_steps + 1, device=device, dtype=dtype)
        traj_final[:, :, 0] = z0_batch

        for t_step in range(num_steps):
            u_desired = policy(x)
            hover_bias = torch.zeros(dynamics.control_dim, device=device, dtype=dtype)
            hover_bias[::4] = T_hover
            u_desired = u_desired + hover_bias

            try:
                u_safe = cbf_controller.filter_control(x, u_desired)
            except Exception as e:
                print(f"CBF-QP failed at step {t_step}: {e}")
                u_safe = u_desired

            x = dynamics.step(x, u_safe, dt)
            traj_final[:, :, t_step + 1] = x

    # Plot final trajectory
    final_plot_path = save_dir / "final_trajectory.png"
    dynamics.plot_trajectory(
        traj=traj_final,
        obstacles=obstacles,
        p_target=p_target,
        save_path=str(final_plot_path),
        title="Final Trajectory - Best Model"
    )
    print(f"Final trajectory plot saved to: {final_plot_path}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=int, default=600)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_quadrotor(args)
