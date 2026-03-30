"""
Quadrotor dynamics with Euler angles.

State: [x, y, z, ψ, θ, φ, vx, vy, vz, vψ, vθ, vφ]  (12D per agent)
Control: [thrust, τψ, τθ, τφ]  (4D per agent)

Dynamics:
    ṗ = v
    [ψ̇, θ̇, φ̇] = [vψ, vθ, vφ]
    v̇ = (thrust/m) * thrust_direction(ψ, θ, φ) - g*e3
    [v̇ψ, v̇θ, v̇φ] = [τψ, τθ, τφ]

For CBF purposes, we focus on position dynamics:
    ẍ = (thrust/m) * thrust_direction(ψ, θ, φ)_x - 0
    ÿ = (thrust/m) * thrust_direction(ψ, θ, φ)_y - 0
    z̈ = (thrust/m) * thrust_direction(ψ, θ, φ)_z - g

This gives RELATIVE DEGREE 2 for position-based barriers.
"""

import torch
from .base import ControlAffineDynamics


class Quadrotor(ControlAffineDynamics):
    """
    Quadrotor dynamics with Euler angles.

    Multi-agent support: n_agent quadrotors
    Total state: 12 * n_agent
    Total control: 4 * n_agent
    """

    def __init__(self, n_agent=1, mass=0.5, gravity=1.0):
        """
        Initialize quadrotor dynamics.

        Args:
            n_agent: Number of quadrotors
            mass: Mass of each quadrotor
            gravity: Gravity constant
        """
        state_dim = 12 * n_agent
        control_dim = 4 * n_agent
        relative_degree = 2  # Position barriers have relative degree 2

        super().__init__(state_dim, control_dim, relative_degree)

        self.n_agent = n_agent
        self.mass = mass
        self.gravity = gravity
        self.T_hover = mass * gravity

    def thrust_direction(self, angles):
        """
        Compute thrust direction from Euler angles.

        Args:
            angles: (batch, n_agent, 3) with [ψ, θ, φ]

        Returns:
            thrust_dir: (batch, n_agent, 3) unit direction
        """
        psi, theta, phi = angles[..., 0], angles[..., 1], angles[..., 2]

        sps, cps = torch.sin(psi), torch.cos(psi)
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)

        # Thrust direction in world frame
        tx = sps * sph + cps * sth * cph
        ty = -cps * sph + sps * sth * cph
        tz = cth * cph

        return torch.stack([tx, ty, tz], dim=-1)

    def split_state(self, x):
        """
        Split state into components.

        Args:
            x: (batch, 12*n_agent)

        Returns:
            pos: (batch, n_agent, 3) position
            angles: (batch, n_agent, 3) Euler angles
            vel: (batch, n_agent, 3) linear velocity
            ang_vel: (batch, n_agent, 3) angular velocity
        """
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, self.n_agent, 12)
        return (x_reshaped[..., 0:3], x_reshaped[..., 3:6],
                x_reshaped[..., 6:9], x_reshaped[..., 9:12])

    def f(self, x):
        """
        Drift dynamics f(x).

        Args:
            x: (batch, 12*n_agent)

        Returns:
            drift: (batch, 12*n_agent)
        """
        batch = x.shape[0]
        pos, angles, vel, ang_vel = self.split_state(x)

        # Drift dynamics
        dpos = vel
        dangles = ang_vel
        dvel = -self.gravity * torch.tensor([0., 0., 1.],
                                            device=x.device, dtype=x.dtype)
        dang_vel = torch.zeros_like(ang_vel)

        drift = torch.cat([dpos, dangles, dvel, dang_vel], dim=-1)
        return drift.reshape(batch, self.state_dim)

    def g(self, x):
        """
        Control matrix g(x).

        For quadrotor:
        - Thrust affects linear acceleration via thrust direction
        - Angular accelerations directly control angular velocity

        Args:
            x: (batch, 12*n_agent)

        Returns:
            G: (batch, 12*n_agent, 4*n_agent)
        """
        batch = x.shape[0]
        pos, angles, vel, ang_vel = self.split_state(x)

        # Compute thrust direction
        thrust_dir = self.thrust_direction(angles)  # (batch, n_agent, 3)

        # Build control matrix
        G = torch.zeros(batch, self.n_agent, 12, 4, device=x.device, dtype=x.dtype)

        for i in range(self.n_agent):
            # Thrust affects velocity (indices 6:9)
            G[:, i, 6:9, 0] = thrust_dir[:, i, :] / self.mass

            # Angular accelerations affect angular velocities (indices 9:12)
            G[:, i, 9:12, 1:4] = torch.eye(3, device=x.device, dtype=x.dtype)

        # Reshape to (batch, 12*n_agent, 4*n_agent)
        G_full = torch.zeros(batch, 12*self.n_agent, 4*self.n_agent,
                            device=x.device, dtype=x.dtype)
        for i in range(self.n_agent):
            G_full[:, 12*i:12*(i+1), 4*i:4*(i+1)] = G[:, i, :, :]

        return G_full

    def step(self, x, u, dt):
        """
        RK4 integration step.

        Args:
            x: (batch, 12*n_agent)
            u: (batch, 4*n_agent)
            dt: time step

        Returns:
            x_next: (batch, 12*n_agent)
        """
        def dynamics(x_temp, u_temp):
            f_x = self.f(x_temp)
            g_x = self.g(x_temp)
            # g_x @ u: (batch, 12*n_agent, 4*n_agent) @ (batch, 4*n_agent, 1)
            return f_x + (g_x @ u_temp.unsqueeze(-1)).squeeze(-1)

        k1 = dynamics(x, u)
        k2 = dynamics(x + dt/2 * k1, u)
        k3 = dynamics(x + dt/2 * k2, u)
        k4 = dynamics(x + dt * k3, u)

        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def __repr__(self):
        return (f"Quadrotor(n_agent={self.n_agent}, mass={self.mass}, "
                f"gravity={self.gravity}, state_dim={self.state_dim}, "
                f"control_dim={self.control_dim})")
