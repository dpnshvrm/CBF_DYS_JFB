"""
Single Integrator Dynamics: ẋ = u

State: x ∈ R^n (position)
Control: u ∈ R^n (velocity)

Control-affine form:
    f(x) = 0
    g(x) = I

Relative degree for position barriers: 1
    - h(x) = barrier on position
    - ḣ = ∇h · ẋ = ∇h · u (control appears in first derivative)
    - Standard CBF applies
"""

import torch
from .base import ControlAffineDynamics


class SingleIntegrator(ControlAffineDynamics):
    """
    Single integrator dynamics: ẋ = u

    Simplest control-affine system with relative degree 1.
    """

    def __init__(self, dim=2):
        """
        Initialize single integrator.

        Args:
            dim: Dimension of position space (default: 2 for planar motion)
        """
        super().__init__(
            state_dim=dim,
            control_dim=dim,
            relative_degree=1  # Standard CBF
        )
        self.dim = dim

    def f(self, x):
        """
        Drift dynamics: f(x) = 0

        Args:
            x: State (batch_size, dim)

        Returns:
            f: Zero drift (batch_size, dim)
        """
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.dim, dtype=x.dtype, device=x.device)

    def g(self, x):
        """
        Control matrix: g(x) = I

        Args:
            x: State (batch_size, dim)

        Returns:
            g: Identity matrix (batch_size, dim, dim)
        """
        batch_size = x.shape[0]
        I = torch.eye(self.dim, dtype=x.dtype, device=x.device)
        return I.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, dim, dim)

    def step(self, x, u, dt):
        """
        Euler integration: x_{t+1} = x_t + dt * u

        Args:
            x: Current position (batch_size, dim)
            u: Control velocity (batch_size, dim)
            dt: Time step

        Returns:
            x_next: Next position (batch_size, dim)
        """
        return x + dt * u

    def position(self, x):
        """
        Extract position from state (for single integrator, state = position).

        Args:
            x: State (batch_size, dim)

        Returns:
            p: Position (batch_size, dim)
        """
        return x
