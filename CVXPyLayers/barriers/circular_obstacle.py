"""
Circular obstacle barrier function.

Works for both single and double integrator systems by automatically
adapting to the relative degree of the dynamics.

Barrier function:
    h(p) = ||p - center||² - r² - ε

Safe set: {x : h(p) ≥ 0} (stay outside circle of radius r around center)

For single integrator (relative degree 1):
    - Standard CBF constraint

For double integrator (relative degree 2):
    - HOCBF constraint with auxiliary barrier
"""

import torch
from .base import RelativeDegree1Barrier, RelativeDegree2Barrier


class CircularObstacle:
    """
    Factory for circular obstacle barriers.

    Automatically creates the correct barrier type based on dynamics.
    """

    def __new__(cls, center, radius, epsilon=0.1, dynamics=None):
        """
        Create circular obstacle barrier matching dynamics relative degree.

        Args:
            center: Obstacle center (tensor or list)
            radius: Obstacle radius (including safety margin)
            epsilon: Small constant for numerical stability
            dynamics: Dynamics object (determines relative degree)

        Returns:
            CircularObstacle1 or CircularObstacle2 instance
        """
        if dynamics is None:
            # Default to relative degree 1
            return CircularObstacle1(center, radius, epsilon)

        if dynamics.relative_degree == 1:
            return CircularObstacle1(center, radius, epsilon)
        elif dynamics.relative_degree == 2:
            return CircularObstacle2(center, radius, epsilon)
        else:
            raise ValueError(f"Unsupported relative degree: {dynamics.relative_degree}")


class CircularObstacle1(RelativeDegree1Barrier):
    """
    Circular obstacle for relative degree 1 (single integrator).

    h(x) = ||x - center||² - r² - ε
    ∇h = 2(x - center)

    For single integrator: ẋ = u
        Lf h = 0 (no drift)
        Lg h = ∇h = 2(x - center)

    CBF constraint: 2(x - center) · u ≥ -α h
    """

    def __init__(self, center, radius, epsilon=0.1):
        """
        Initialize circular obstacle barrier.

        Args:
            center: Obstacle center position (tensor or list)
            radius: Total clearance radius (obstacle + safety margin)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon

    def h(self, x, dynamics=None):
        """
        Evaluate barrier function.

        Args:
            x: State (batch_size, state_dim)
            dynamics: Optional (not used for single integrator)

        Returns:
            h: Barrier value (batch_size,)
        """
        # Ensure center is on same device
        center = self.center.to(x.device).to(x.dtype)

        dist_sq = torch.sum((x - center)**2, dim=-1)
        return dist_sq - self.radius**2 - self.epsilon

    def compute_lie_derivatives(self, x, dynamics):
        """
        Compute Lie derivatives for single integrator.

        Args:
            x: State (batch_size, state_dim)
            dynamics: SingleIntegrator object

        Returns:
            Lf_h: Zero (batch_size,)
            Lg_h: 2(x - center) (batch_size, control_dim)
            h_val: Barrier value (batch_size,)
        """
        center = self.center.to(x.device).to(x.dtype)

        # Barrier value
        h_val = self.h(x, dynamics)

        # Gradient: ∇h = 2(x - center)
        grad_h = 2.0 * (x - center)

        # For single integrator: f(x) = 0, g(x) = I
        Lf_h = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        Lg_h = grad_h  # ∇h · I = ∇h

        return Lf_h, Lg_h, h_val

    def __repr__(self):
        return (f"CircularObstacle1(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon})")


class CircularObstacle2(RelativeDegree2Barrier):
    """
    Circular obstacle for relative degree 2 (double integrator).

    h₀(p) = ||p - center||² - r² - ε

    For double integrator x = [p, v], ẍ = a:
        ḣ₀ = 2(p - center) · v
        ḧ₀ = 2||v||² + 2(p - center) · a

    Auxiliary barrier:
        h₁ = ḣ₀ + α₁ h₀

    HOCBF constraint:
        2(p - center) · a ≥ -2||v||² - α₂(ḣ₀ + α₁ h₀)
    """

    def __init__(self, center, radius, epsilon=0.1):
        """
        Initialize circular obstacle barrier for double integrator.

        Args:
            center: Obstacle center position (tensor or list)
            radius: Total clearance radius (obstacle + safety margin)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon

    def h(self, x, dynamics):
        """
        Evaluate barrier function on position.

        Args:
            x: State [p, v] (batch_size, 2*dim)
            dynamics: DoubleIntegrator object

        Returns:
            h: Barrier value (batch_size,)
        """
        # Extract position
        p = dynamics.position(x)

        # Ensure center is on same device
        center = self.center.to(x.device).to(x.dtype)

        dist_sq = torch.sum((p - center)**2, dim=-1)
        return dist_sq - self.radius**2 - self.epsilon

    def compute_hocbf_terms(self, x, dynamics):
        """
        Compute HOCBF terms for double integrator.

        For h₀(p) = ||p - c||² - r² - ε:
            ∇h₀ = 2(p - c)
            ḣ₀ = 2(p - c) · v
            ḧ₀ = 2||v||² + 2(p - c) · a

        Args:
            x: State [p, v] (batch_size, 2*dim)
            dynamics: DoubleIntegrator object

        Returns:
            h0: Barrier value (batch_size,)
            h0_dot: First derivative (batch_size,)
            Lf2_h0: Second Lie derivative without control (batch_size,)
            Lg_Lf_h0: Control gradient (batch_size, control_dim)
        """
        p = dynamics.position(x)
        v = dynamics.velocity(x)

        center = self.center.to(x.device).to(x.dtype)

        # h₀(p)
        dist_sq = torch.sum((p - center)**2, dim=-1)
        h0 = dist_sq - self.radius**2 - self.epsilon

        # Gradient ∇h₀ = 2(p - center)
        grad_h0 = 2.0 * (p - center)

        # ḣ₀ = ∇h₀ · v
        h0_dot = torch.sum(grad_h0 * v, dim=-1)

        # ḧ₀ = 2||v||² + 2(p - center) · a
        #    = Lf²h₀ + Lg Lf h₀ · a

        # Lf²h₀ = 2||v||² (drift contribution to second derivative)
        Lf2_h0 = 2.0 * torch.sum(v**2, dim=-1)

        # Lg Lf h₀ = 2(p - center) (control gradient)
        Lg_Lf_h0 = grad_h0

        return h0, h0_dot, Lf2_h0, Lg_Lf_h0

    def __repr__(self):
        return (f"CircularObstacle2(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon})")
