"""
DC3-style safety filter for the coupled double-integrator experiment (a second baseline next to CVXPY Layers).

It follows the *inequality correction* of DC3 (Donti, Rolnick, Kolter, ICLR 2021; arXiv 2104.12225), specialised to
our constraint K u <= d.  Writing the constraint with a slack s = d - K u as the completed variable (DC3's
"completion", explicit here), the squared violation is  || ReLU(K u - d) ||^2  and DC3's gradient correction is

    step_k = lr * grad_u || ReLU(K u_k - d) ||^2  +  momentum * step_{k-1}  ,   grad = 2 K^T ReLU(K u_k - d)
    u_{k+1} = u_k - step_k                                                    (k = 0 .. steps-1, u_0 = u_nom)

exactly as in DC3's repo (method.py, grad_steps: new_step = lr*grad + momentum*old_step).
The correction is unrolled and differentiated through by autograd (DC3 does the same, which is why it keeps `steps` small).
It is NOT a projection: it returns a point with smaller violation, not the closest feasible point.

Interface = DYSProjector's:  u, _, info = proj(u_nom, K, dvec, max_iter=None, tol=None)
    u_nom (B, m), K (B, c, m), dvec (B, c, 1);  info = {'iters', 'final_residual', 'failed'}
    iters = number of correction steps; final_residual = largest remaining violation max(K u - d, 0).
max_iter / tol are accepted and ignored (DC3 uses a fixed number of steps).
If the correction produces non-finite values, the nominal control is returned for that call (counted in n_fail).
"""
import torch


class DC3Projector(torch.nn.Module):
    def __init__(self, steps=10, lr=0.01, momentum=0.5):
        super().__init__()
        self.steps = int(steps)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.n_fail = 0

    def forward(self, u_nom, K, dvec, max_iter=None, tol=None):
        Kd = K.detach() if not K.requires_grad else K
        u = u_nom
        old_step = torch.zeros_like(u_nom)
        for _ in range(self.steps):
            viol = torch.relu(Kd @ u.unsqueeze(-1) - dvec)                 # (B, c, 1)
            grad = 2.0 * (Kd.transpose(-2, -1) @ viol).squeeze(-1)          # (B, m)
            step = self.lr * grad + self.momentum * old_step
            u = u - step
            old_step = step
        failed = not bool(torch.isfinite(u).all())
        if failed:
            self.n_fail += 1
            u = u_nom
        with torch.no_grad():
            resid = torch.relu(K @ u.unsqueeze(-1) - dvec).max()
        return u, None, {"iters": self.steps, "final_residual": float(resid), "failed": failed}
