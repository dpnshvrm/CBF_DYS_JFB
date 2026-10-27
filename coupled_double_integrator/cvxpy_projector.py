"""
CVXPY-Layers safety filter for the coupled double-integrator experiment (baseline for DYSProjector).

It solves, for every sample in the batch,

    hard   :  u* = argmin_u  ||u - u_theta||^2                         s.t.  K(z) u <= d(z)
    slack  :  u* = argmin_{u,s>=0}  ||u - u_theta||^2 + lam ||s||^2    s.t.  K(z) u - s <= d(z)

with SCS and implicit differentiation (cvxpylayers), using the same solver settings as the repo's
cvxpy_exp/controllers/cbf_qp_layer.py (SCS, eps = 1e-3, max_iters = 1000, acceleration_lookback = 0).
The slack variant uses the QUADRATIC penalty lam*||s||^2 with lam = 1e4, as both repos' code does
(the paper's eq. (reg_qp) is written with a linear penalty; see the note in CLAUDE.md).

The objective is written as ||u||^2 - 2 u_theta^T u (the repo's form; same minimiser, known to be DPP).

Interface = DYSProjector's:  u, _, info = proj(u_nom, K, dvec, max_iter=None, tol=None)
    u_nom (B, m), K (B, c, m), dvec (B, c, 1);  info = {'iters', 'final_residual', 'failed'}
    final_residual = largest constraint violation max(K u - d, 0) of the returned control.
If SCS fails or returns non-finite values for a batch, the nominal control is returned unchanged for that
batch (it is then NOT safe), proj.n_fail is incremented and info['failed'] is True.
"""
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CvxpyProjector(torch.nn.Module):
    def __init__(self, n_control, n_pair, slack=False, slack_penalty=1e4, eps=1e-3, max_iters=1000):
        super().__init__()
        self.slack = bool(slack)
        self.n_fail = 0
        self.solver_args = {"solve_method": "SCS", "eps": float(eps), "max_iters": int(max_iters),
                            "acceleration_lookback": 0}
        u = cp.Variable(n_control)
        u_nom = cp.Parameter(n_control)
        K = cp.Parameter((n_pair, n_control))
        dvec = cp.Parameter(n_pair)
        if self.slack:
            s = cp.Variable(n_pair, nonneg=True)
            objective = cp.Minimize(cp.sum_squares(u) - 2 * (u_nom @ u) + float(slack_penalty) * cp.sum_squares(s))
            constraints = [K @ u - s <= dvec]
        else:
            objective = cp.Minimize(cp.sum_squares(u) - 2 * (u_nom @ u))
            constraints = [K @ u <= dvec]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "problem must be DPP for cvxpylayers"
        self.layer = CvxpyLayer(problem, parameters=[u_nom, K, dvec], variables=[u])

    def forward(self, u_nom, K, dvec, max_iter=None, tol=None):
        # max_iter / tol are accepted for interface compatibility with DYSProjector and ignored:
        # the solver settings are fixed at construction (paper: max_iters 1000, eps 1e-3).
        dev, dtype = u_nom.device, u_nom.dtype
        failed = False
        try:
            u, = self.layer(u_nom.cpu(), K.detach().cpu(), dvec.squeeze(-1).detach().cpu(),
                            solver_args=self.solver_args)
            u = u.to(device=dev, dtype=dtype)
            if not bool(torch.isfinite(u).all()):
                failed = True
        except Exception as exc:  # cvxpy.SolverError, diffcp errors, ...
            failed = True
            if self.n_fail < 5:
                print(f"[CvxpyProjector] solve failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        if failed:
            self.n_fail += 1
            u = u_nom
        with torch.no_grad():
            viol = (K @ u.unsqueeze(-1) - dvec).clamp(min=0).max()
        return u, None, {"iters": 0, "final_residual": float(viol), "failed": failed}
