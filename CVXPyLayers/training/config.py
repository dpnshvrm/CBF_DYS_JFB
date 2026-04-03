"""
Unified configuration system for CBF training.

Works for any combination of:
    - Dynamics (single/double integrator)
    - Obstacles (number and configuration)
    - Training parameters
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for CBF-based control training.

    This config works for both single and double integrator systems,
    with arbitrary number of obstacles.
    """

    # ========================================================================
    # Problem Setup
    # ========================================================================
    initial_state: List[float] = field(default_factory=lambda: [0.0, 0.0])
    target_state: List[float] = field(default_factory=lambda: [2.0, 2.0])

    # ========================================================================
    # Dynamics Type
    # ========================================================================
    dynamics_type: str = 'single_integrator'  # 'single_integrator' or 'double_integrator'
    position_dim: int = 2  # dimension of position space

    # ========================================================================
    # Obstacles (list of dicts with 'center', 'radius', 'epsilon')
    # ========================================================================
    obstacles: List[dict] = field(default_factory=lambda: [
        {'center': [1.0, 1.0], 'radius': 0.6, 'epsilon': 0.1}
    ])

    # ========================================================================
    # Time Parameters
    # ========================================================================
    T: float = 2.0          # Time horizon
    dt: float = 0.05        # Time step (0.05 for training, 0.01 for eval)

    # ========================================================================
    # CBF Parameters
    # ========================================================================
    cbf_alpha: float = 10.0  # Class-K parameter
    # For HOCBF (double integrator), can be scalar or tuple (alpha1, alpha2)
    # If scalar, same value used for both

    # ========================================================================
    # Cost Weights
    # ========================================================================
    control_penalty: float = 1.0
    terminal_cost_weight: float = 500.0

    # ========================================================================
    # Network Architecture
    # ========================================================================
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    activation: str = 'relu'  # 'relu' or 'silu'

    # ========================================================================
    # Training Parameters
    # ========================================================================
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    initial_state_std: float = 0.3  # Noise for sampling initial states
    grad_clip_norm: float = 1.0     # Max gradient norm (0 = no clipping)

    # ========================================================================
    # Computation
    # ========================================================================
    device: Optional[str] = None  # Auto-detect if None
    use_double_precision: bool = True

    # ========================================================================
    # Model Saving
    # ========================================================================
    save_path: str = './models/cbf_controller.pth'
    save_best_only: bool = True

    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-detect device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set dtype
        self.dtype = torch.float64 if self.use_double_precision else torch.float32

        # Compute derived quantities
        self.num_steps = int(self.T / self.dt)

        # Determine state and control dimensions based on dynamics type
        if self.dynamics_type == 'single_integrator':
            self.state_dim = self.position_dim
            self.control_dim = self.position_dim
        elif self.dynamics_type == 'double_integrator':
            self.state_dim = 2 * self.position_dim  # [position, velocity]
            self.control_dim = self.position_dim     # acceleration
        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics_type}")

        # Convert lists to tensors
        self.initial_state_tensor = torch.tensor(
            self.initial_state, dtype=self.dtype, device=self.device
        )
        self.target_state_tensor = torch.tensor(
            self.target_state, dtype=self.dtype, device=self.device
        )

        # Ensure correct dimensions
        if len(self.initial_state) != self.state_dim:
            raise ValueError(
                f"Initial state dimension {len(self.initial_state)} "
                f"does not match expected state_dim {self.state_dim}"
            )
        if len(self.target_state) != self.state_dim:
            raise ValueError(
                f"Target state dimension {len(self.target_state)} "
                f"does not match expected state_dim {self.state_dim}"
            )

    def get_num_obstacles(self):
        """Get number of obstacles."""
        return len(self.obstacles)

    def __repr__(self):
        """Pretty print configuration."""
        lines = ["="*70]
        lines.append("CBF Training Configuration")
        lines.append("="*70)
        lines.append(f"Dynamics:      {self.dynamics_type}")
        lines.append(f"Initial state: {self.initial_state}")
        lines.append(f"Target state:  {self.target_state}")
        lines.append(f"Time:          T={self.T}s, dt={self.dt}s, steps={self.num_steps}")
        lines.append(f"Obstacles:     {len(self.obstacles)}")
        for i, obs in enumerate(self.obstacles):
            lines.append(f"  [{i+1}] center={obs['center']}, radius={obs['radius']}")
        lines.append(f"CBF alpha:     {self.cbf_alpha}")
        lines.append(f"Cost weights:  control={self.control_penalty}, terminal={self.terminal_cost_weight}")
        lines.append(f"Network:       {self.state_dim}→{self.hidden_dim}x{self.num_hidden_layers}→{self.control_dim}")
        lines.append(f"Training:      epochs={self.num_epochs}, batch={self.batch_size}, lr={self.learning_rate}")
        lines.append(f"Device:        {self.device}, dtype={self.dtype}")
        lines.append("="*70)
        return "\n".join(lines)


# ============================================================================
# Preset Configurations
# ============================================================================

def single_integrator_two_obstacles():
    """Example config: single integrator with 2 obstacles."""
    return TrainingConfig(
        dynamics_type='single_integrator',
        position_dim=2,
        initial_state=[0.0, 0.0],
        target_state=[4.0, 4.0],
        obstacles=[
            {'center': [1.0, 1.0], 'radius': 0.55, 'epsilon': 0.1},
            {'center': [2.5, 2.5], 'radius': 0.55, 'epsilon': 0.1},
        ],
        T=6.0,
        dt=0.05,
        cbf_alpha=10.0,
        num_epochs=300,
        save_path='./models/single_int_two_obs.pth'
    )


def double_integrator_three_obstacles():
    """Example config: double integrator with 3 obstacles (HOCBF)."""
    return TrainingConfig(
        dynamics_type='double_integrator',
        position_dim=2,
        initial_state=[0.0, 0.0, 0.0, 0.0],  # [px, py, vx, vy]
        target_state=[3.0, 3.0, 0.0, 0.0],   # [px, py, vx, vy]
        obstacles=[
            {'center': [0.4, 1.0], 'radius': 0.4, 'epsilon': 0.1},
            {'center': [2.2, 2.2], 'radius': 0.4, 'epsilon': 0.1},
            {'center': [2.4, 0.6], 'radius': 0.4, 'epsilon': 0.1},
        ],
        T=10.0,
        dt=0.2,
        cbf_alpha=(5.0, 5.0),  # (alpha1, alpha2) for HOCBF
        control_penalty=1.0,
        terminal_cost_weight=1000.0,
        num_epochs=2000,
        batch_size=64,
        save_path='./models/double_int_three_obs.pth'
    )
