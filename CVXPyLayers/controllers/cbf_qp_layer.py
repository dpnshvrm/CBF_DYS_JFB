"""
Differentiable CBF-QP layer using CVXPyLayers.

Solves the safety-critical QP:
    minimize:    ||u - u_des||²
    subject to:  A_i @ u ≥ b_i  for all obstacles i

Works for both:
    - Standard CBF (relative degree 1)
    - HOCBF (relative degree 2)

The number of constraints = number of obstacles.
Each constraint ensures safety w.r.t. one obstacle.
"""

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def create_cbf_qp_layer(control_dim, num_obstacles):
    """
    Create differentiable CBF-QP layer with multiple obstacle constraints.

    QP formulation:
        minimize:    ||u - u_desired||²
        subject to:  A_i @ u ≥ b_i  for i = 1, ..., num_obstacles

    The specific form of A_i, b_i depends on relative degree:
        - Relative degree 1: A_i = Lg h_i, b_i = -Lf h_i - α(h_i)
        - Relative degree 2: A_i = Lg Lf h_i, b_i = -Lf² h_i - α₂(h₁_i)

    Args:
        control_dim: Dimension of control input
        num_obstacles: Number of obstacle constraints

    Returns:
        CvxpyLayer: Differentiable QP solver
    """
    # Decision variable
    u = cp.Variable(control_dim)

    # Parameters
    u_desired = cp.Parameter(control_dim)

    # CBF constraint parameters (one per obstacle)
    A_cbf_list = [cp.Parameter(control_dim) for _ in range(num_obstacles)]
    b_cbf_list = [cp.Parameter() for _ in range(num_obstacles)]

    # Objective: minimize deviation from desired control
    # ||u - u_des||² = ||u||² - 2·u_des^T·u + ||u_des||²
    # Drop constant term
    objective = cp.Minimize(cp.sum_squares(u) - 2 * (u_desired @ u))

    # Constraints: one per obstacle
    constraints = [A_cbf_list[i] @ u >= b_cbf_list[i] for i in range(num_obstacles)]

    # Create problem
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem must be DPP (Disciplined Parametrized Programming)"

    # Create differentiable layer
    # Parameters: [u_desired, A_cbf_1, b_cbf_1, A_cbf_2, b_cbf_2, ...]
    parameters = [u_desired] + [param for pair in zip(A_cbf_list, b_cbf_list) for param in pair]

    layer = CvxpyLayer(
        problem,
        parameters=parameters,
        variables=[u]
    )

    return layer


class CBFQPController:
    """
    CBF-QP safety filter for multiple obstacles.

    Takes desired control and projects it to safe control satisfying
    all CBF constraints.
    """

    def __init__(self, dynamics, obstacles, alpha):
        """
        Initialize CBF-QP controller.

        Args:
            dynamics: Dynamics object (provides control_dim, relative_degree)
            obstacles: List of BarrierFunction objects
            alpha: CBF class-K parameter(s)
                  - For relative degree 1: scalar
                  - For relative degree 2: tuple (alpha1, alpha2) or scalar
        """
        self.dynamics = dynamics
        self.obstacles = obstacles
        self.alpha = alpha
        self.num_obstacles = len(obstacles)

        # Create QP layer
        self.qp_layer = create_cbf_qp_layer(dynamics.control_dim, self.num_obstacles)

        # Verify all obstacles have same relative degree as dynamics
        for obs in obstacles:
            if obs.relative_degree != dynamics.relative_degree:
                raise ValueError(
                    f"Obstacle relative degree {obs.relative_degree} "
                    f"does not match dynamics relative degree {dynamics.relative_degree}"
                )

    def filter_control(self, x, u_desired):
        """
        Filter desired control through CBF safety constraints.

        Args:
            x: Current state (batch_size, state_dim)
            u_desired: Desired control from policy (batch_size, control_dim)

        Returns:
            u_safe: Safe control (batch_size, control_dim)
        """
        batch_size = x.shape[0]

        # Compute CBF constraint parameters for each obstacle
        A_cbf_list = []
        b_cbf_list = []

        for obstacle in self.obstacles:
            A_cbf, b_cbf = obstacle.compute_cbf_constraint(x, self.dynamics, self.alpha)
            A_cbf_list.append(A_cbf)
            b_cbf_list.append(b_cbf)

        # Flatten parameters for QP layer
        # [u_desired, A_cbf_1, b_cbf_1, A_cbf_2, b_cbf_2, ...]
        qp_params = [u_desired]
        for A, b in zip(A_cbf_list, b_cbf_list):
            qp_params.append(A)
            qp_params.append(b)

        # Solve QP
        u_safe, = self.qp_layer(*qp_params)

        return u_safe

    def __repr__(self):
        return (f"CBFQPController("
                f"dynamics={self.dynamics.__class__.__name__}, "
                f"num_obstacles={self.num_obstacles}, "
                f"alpha={self.alpha})")
