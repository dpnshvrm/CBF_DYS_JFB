"""
Unified training loop for CBF-based control.

Works for any combination of dynamics and obstacles.
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from dynamics import SingleIntegrator, DoubleIntegrator
from barriers import CircularObstacle
from controllers import CBFQPController, PolicyNetwork


class CBFTrainer:
    """
    Trainer for CBF-based safe control policies.

    Handles both single and double integrator dynamics with
    arbitrary number of obstacles.
    """

    def __init__(self, config):
        """
        Initialize trainer.

        Args:
            config: TrainingConfig object
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Create dynamics
        if config.dynamics_type == 'single_integrator':
            self.dynamics = SingleIntegrator(dim=config.position_dim)
        elif config.dynamics_type == 'double_integrator':
            self.dynamics = DoubleIntegrator(dim=config.position_dim)
        else:
            raise ValueError(f"Unknown dynamics type: {config.dynamics_type}")

        # Create obstacles
        self.obstacles = []
        for obs_config in config.obstacles:
            obstacle = CircularObstacle(
                center=obs_config['center'],
                radius=obs_config['radius'],
                epsilon=obs_config.get('epsilon', 0.1),
                dynamics=self.dynamics
            )
            self.obstacles.append(obstacle)

        # Create CBF-QP controller
        self.cbf_controller = CBFQPController(
            dynamics=self.dynamics,
            obstacles=self.obstacles,
            alpha=config.cbf_alpha
        )

        # Create policy network
        self.policy = PolicyNetwork(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation
        ).to(self.device).to(self.dtype)

        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        # Training state
        self.best_loss = float('inf')
        self.epoch = 0

    def sample_initial_states(self, batch_size):
        """
        Sample initial states for training.

        Args:
            batch_size: Number of samples

        Returns:
            z0: Initial states (batch_size, state_dim)
        """
        noise = torch.randn(
            batch_size, self.config.state_dim,
            dtype=self.dtype, device=self.device
        ) * self.config.initial_state_std

        z0 = self.config.initial_state_tensor.unsqueeze(0) + noise
        return z0

    def rollout(self, z0):
        """
        Simulate trajectory with CBF safety filtering.

        Args:
            z0: Initial states (batch_size, state_dim)

        Returns:
            states: State trajectory (batch_size, num_steps+1, state_dim)
            controls: Control trajectory (batch_size, num_steps, control_dim)
        """
        batch_size = z0.shape[0]
        states = [z0]
        controls = []

        z = z0
        for step in range(self.config.num_steps):
            # Policy proposes control
            u_desired = self.policy(z)

            # CBF-QP filters for safety
            try:
                u_safe = self.cbf_controller.filter_control(z, u_desired)
            except Exception as e:
                # QP infeasible - fallback to desired control
                if "infeasible" in str(e).lower() or "unbounded" in str(e).lower():
                    print(f"\n  Warning: CBF-QP failed at step {step}: {e}")
                    print(f"  Falling back to desired control (unsafe!)")
                    u_safe = u_desired
                else:
                    raise e

            # Simulate dynamics
            z = self.dynamics.step(z, u_safe, self.config.dt)

            states.append(z)
            controls.append(u_safe)

        states = torch.stack(states, dim=1)      # (batch, num_steps+1, state_dim)
        controls = torch.stack(controls, dim=1)  # (batch, num_steps, control_dim)

        return states, controls

    def compute_loss(self, states, controls):
        """
        Compute training loss.

        Args:
            states: State trajectory (batch_size, num_steps+1, state_dim)
            controls: Control trajectory (batch_size, num_steps, control_dim)

        Returns:
            total_loss: Weighted sum of running + terminal cost
            running_cost: Control effort
            terminal_cost: Distance to target
        """
        # Running cost: control effort
        running_cost = torch.mean(
            torch.sum(controls**2, dim=-1).sum(dim=-1)
        ) * self.config.dt

        # Terminal cost: distance to target
        final_state = states[:, -1, :]
        target = self.config.target_state_tensor.unsqueeze(0)
        terminal_cost = torch.mean(
            torch.sum((final_state - target)**2, dim=-1)
        )

        # Total loss
        total_loss = (
            self.config.control_penalty * running_cost +
            self.config.terminal_cost_weight * terminal_cost
        )

        return total_loss, running_cost, terminal_cost

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            loss: Total loss
            running_cost: Running cost
            terminal_cost: Terminal cost
        """
        self.policy.train()

        # Sample initial states
        z0 = self.sample_initial_states(self.config.batch_size)

        # Forward pass
        states, controls = self.rollout(z0)
        loss, running_cost, terminal_cost = self.compute_loss(states, controls)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients through long rollouts)
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()

        return loss.item(), running_cost.item(), terminal_cost.item()

    def train(self, verbose=True):
        """
        Full training loop.

        Args:
            verbose: Print progress

        Returns:
            policy: Trained policy network
        """
        if verbose:
            print(f"\n{self.config}")
            print("\nStarting training...")

        pbar = tqdm(range(self.config.num_epochs), disable=not verbose)

        for epoch in pbar:
            self.epoch = epoch

            # Train one epoch
            loss, running, terminal = self.train_epoch()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'run': f'{running:.4f}',
                'term': f'{terminal:.4f}',
                'best': f'{self.best_loss:.4f}'
            })

            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint()

        if verbose:
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")

        return self.policy

    def save_checkpoint(self):
        """Save model checkpoint."""
        metadata = {
            'epoch': self.epoch,
            'loss': self.best_loss,
            'config': {
                'dynamics_type': self.config.dynamics_type,
                'num_obstacles': len(self.obstacles),
                'cbf_alpha': self.config.cbf_alpha,
                'dt': self.config.dt,
                'num_steps': self.config.num_steps
            }
        }
        self.policy.save(self.config.save_path, metadata)

    def load_checkpoint(self, filepath=None):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint (uses config path if None)
        """
        if filepath is None:
            filepath = self.config.save_path

        self.policy, metadata = PolicyNetwork.load(
            filepath,
            device=self.device,
            dtype=self.dtype
        )

        return metadata
