import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as torch_checkpoint

# DYS Projection 
class DYSProjector(nn.Module):
    """
    Differentiable per-sample projector:
        u*_j = argmin_u  ½‖u − u_nom_j‖²   s.t.  A_j u ≤ b_j

    Lifted variable:
        z = (u, s)
        A u + s = b,   s >= 0

    u is kept free. Only s is projected onto non-negativity.
    """
    def __init__(self, alpha: float = 0.5, sv_thresh: float = 1e-6, grad_mode: str = "jfb"):
        super().__init__()
        self.alpha     = alpha
        self.sv_thresh = sv_thresh
        self.grad_mode = grad_mode

    def _build_matrices(self, A, b):
        batch, m, n = A.shape

        I_m = torch.eye(m, dtype=A.dtype, device=A.device).unsqueeze(0).expand(batch, -1, -1)

        # z = (u, s), constraint: A u + s = b
        A_std = torch.cat([A, I_m], dim=2)   # (batch, m, n + m)

        # Q_u extracts u from z
        M = torch.cat([
            torch.eye(n, dtype=A.dtype, device=A.device),
            torch.zeros(n, m, dtype=A.dtype, device=A.device),
        ], dim=1)                            # (n, n + m)

        AAt      = A_std @ A_std.transpose(-2, -1)
        U, S, _  = torch.linalg.svd(AAt)
        S_inv    = torch.where(S > self.sv_thresh, 1.0 / S, torch.zeros_like(S))
        AAt_pinv = (U * S_inv.unsqueeze(-2)) @ U.transpose(-2, -1)

        P_perp = A_std.transpose(-2, -1) @ AAt_pinv

        return A_std, M, P_perp

    def proj_C1(self, z, n):
        """
        Project onto C1 = {(u, s): s >= 0}.
        u is free, so only s is clamped.
        """
        u = z[:, :n]
        s = torch.clamp(z[:, n:], min=0)
        return torch.cat([u, s], dim=1)

    def proj_C2(self, v, A_std, P_perp, b):
        """
        Project onto C2 = {(u, s): A u + s = b}.
        """
        resid = A_std @ v.unsqueeze(-1) - b
        return v - (P_perp @ resid).squeeze(-1)

    def apply_T(self, z, u_nom, A_std, M, P_perp, b, n):
        x = self.proj_C1(z, n)

        # grad of 1/2 ||Q_u x - u_nom||^2
        grad = (x @ M.T - u_nom) @ M

        v = 2 * x - z - self.alpha * grad
        y = self.proj_C2(v, A_std, P_perp, b)

        return z + y - x

    def forward(self, u_nom, A, b, z0=None,
                max_iter=2000, tol=1e-2, verbose=False, n_grad_iters=1):
        """
        Args:
            u_nom    : (batch, n)
            A        : (batch, m, n)
            b        : (batch, m, 1)
            z0       : (batch, n + m), optional warm start
        Returns:
            u_star : (batch, n)
            z_star : (batch, n + m)
            info   : dict
        """
        batch, m, n = A.shape
        N = n + m

        A_std, M, P_perp = self._build_matrices(A, b)

        z = z0 if z0 is not None else torch.zeros(
            batch, N, dtype=u_nom.dtype, device=u_nom.device
        )

        residuals = []
        converged = False
        k = 0

        if self.grad_mode == "jfb":
            with torch.no_grad():
                for k in range(max_iter - 1):
                    z_new = self.apply_T(z, u_nom, A_std, M, P_perp, b, n)
                    res = (z_new - z).abs().max().item()
                    residuals.append(res)
                    z = z_new

                    if verbose:
                        print(f"  iter {k+1:4d}  |  max residual = {res:.2e}")

                    if res < tol:
                        converged = True
                        break

            for _ in range(n_grad_iters):
                z = self.apply_T(z, u_nom, A_std, M, P_perp, b, n)

        elif self.grad_mode == "ad":
            for k in range(max_iter):
                z_new = self.apply_T(z, u_nom, A_std, M, P_perp, b, n)
                res = (z_new - z).abs().max().item()
                residuals.append(res)
                z = z_new

                if verbose:
                    print(f"  iter {k+1:4d}  |  max residual = {res:.2e}")

                if res < tol:
                    converged = True
                    break

        else:
            raise ValueError(f"Unknown grad_mode: {self.grad_mode}")

        z_star = z
        u_star = z_star[:, :n]

        info = dict(
            iters=k + 1,
            converged=converged,
            final_residual=residuals[-1] if residuals else 0.0,
            residuals=residuals,
        )

        return u_star, z_star, info
        

def euler_step(z, u, ti, h, f):
    return z + h * f(z, u, ti) 
        
def rk4_step(z, u, ti, h, f):
    """One RK4 step with control u held constant over [ti, ti+h]."""
    k1 = f(z, u, ti)
    k2 = f(z + (h/2) * k1,  u, ti + h/2)
    k3 = f(z + (h/2) * k2,  u, ti + h/2)
    k4 = f(z + h * k3, u, ti + h)
    return z + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
#### Neural Network Definition
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ControlNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, n_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks     = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        t_batch = t.view(-1, 1).expand(x.shape[0], -1)   # (batch, 1)
        xt = torch.cat([x, t_batch], dim=-1)              # (batch, input_dim)
        h  = self.input_proj(xt)                          # (batch, hidden_dim)
        h  = self.blocks(h)                               # (batch, hidden_dim)
        return self.output_proj(h)                        # (batch, output_dim)
        

class ControlNetPerAgent(nn.Module):
    def __init__(self, agent_state_dim=12, hidden_dim=128, control_dim=4,
                 n_blocks=3, n_agent=30, n_obs=0, T_total=8.5,
                 pos_dim=3, vel_start=6):
        super().__init__()
        self.n_agent, self.agent_state_dim = n_agent, agent_state_dim
        self.n_obs, self.T_total = n_obs, T_total
        self.pos_dim, self.vel_start = pos_dim, vel_start

        extra_dim = pos_dim * (2 + n_obs)
        self.input_proj = nn.Linear(agent_state_dim + 1 + extra_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ])
            for _ in range(n_blocks)
        ])
        self.act = nn.SiLU()
        self.thrust_proj = nn.Linear(hidden_dim, 1)
        self.torque_proj = nn.Linear(hidden_dim, control_dim - 1)

        for layer in [self.thrust_proj, self.torque_proj]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _block(self, h, block):
        norm, lin1, lin2 = block
        return self.act(h + lin2(self.act(lin1(norm(h)))))

    def forward(self, x, t, p_target=None, obstacle_centers=None):
        B = x.shape[0]
        xa = x.reshape(B * self.n_agent, self.agent_state_dim)
        pos = xa[:, :self.pos_dim]
        vel = xa[:, self.vel_start:self.vel_start + self.pos_dim]

        t_togo = ((self.T_total - t) / self.T_total).reshape(1, 1).expand(B * self.n_agent, 1)
        pt = p_target.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_agent, self.pos_dim)

        feats = [xa, t_togo, pt - pos, vel]
        for c in obstacle_centers:
            feats.append(pos - c.reshape(1, self.pos_dim).expand(B * self.n_agent, self.pos_dim))

        h = self.input_proj(torch.cat(feats, dim=-1))

        for block in self.blocks:
            h = torch_checkpoint.checkpoint(
                lambda y, b=block: self._block(y, b), h, use_reentrant=False
            ) if torch.is_grad_enabled() else self._block(h, block)

        out = torch.cat([self.thrust_proj(h), self.torque_proj(h)], dim=-1)
        return out.reshape(B, self.n_agent * out.shape[-1])