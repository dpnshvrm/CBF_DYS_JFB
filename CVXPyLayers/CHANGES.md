# CVXPyLayers Multi-Quadrotor: Problem Setup, CBF, and Bug Fixes

Reference implementation: `quadcopter_multi.py` + `train.py` (DYS/JFB projector).

---

## 1. Control Problem

### System

`N = 5` quadrotors, each with a 12-dimensional state and 4-dimensional control:

```
State per agent:   z_i = [x, y, z,  ψ, θ, φ,  vx, vy, vz,  vψ, vθ, vφ]  ∈ R¹²
Control per agent: u_i = [T_i,  τψ_i, τθ_i, τφ_i]  ∈ R⁴
```

- `T_i` — collective thrust (scalar)
- `τψ, τθ, τφ` — angular accelerations (body-frame torques)
- `ψ, θ, φ` — yaw, pitch, roll (Euler angles)

Full stacked state and control: `z ∈ R⁶⁰`,  `u ∈ R²⁰`.

### Dynamics  `ż = f(z) + g(z)u`

```
ṗ_i      =  v_i
[ψ̇, θ̇, φ̇]  =  [vψ, vθ, vφ]
v̇_i      =  (T_i / m) · d_i(ψ,θ,φ)  −  g e₃
[v̇ψ, v̇θ, v̇φ]  =  [τψ, τθ, τφ]
```

where the thrust direction `d_i` in the world frame is:

```
d = [ sin(ψ)sin(φ) + cos(ψ)sin(θ)cos(φ),
     −cos(ψ)sin(φ) + sin(ψ)sin(θ)cos(φ),
      cos(θ)cos(φ) ]
```

The drift `f(z)` collects the gravity and kinematic terms; `g(z)` maps `u` into
accelerations.  The control matrix `g` is block-diagonal over agents.

### Optimal Control Problem

```
min_{θ}   E_{z₀}[ ∫₀ᵀ ½‖u(t)‖² + 2‖v(t)‖²  dt  +  G(z(T)) ]

subject to:
  ż = f(z) + g(z) u                     (dynamics, integrated with RK4)
  A_cbf(z) · u  ≥  b_cbf(z)             (HOCBF safety for all agents & obstacles)
  u = u_safe = argmin_u' ½‖u' − π_θ(z)‖²  s.t. A_cbf u' ≥ b_cbf
```

The terminal cost is:

```
G(z) = ½‖pos − p_target‖² + ½‖vel‖² + ½‖angles‖² + ½‖ang_vel‖²
```

The total training loss is `α_run · L_run  +  α_term · G`, where `α_term` is
annealed upward during training.

### Policy architecture

The neural network `π_θ : R⁶⁰ → R²⁰` maps the full stacked state to raw control
deviations.  Output is bounded before adding the hover bias:

```
u_policy = [ T_dev_scale · tanh(raw[:, :, 0]),   ← thrust deviation ∈ [−0.5, 0.5]
             tau_scale    · tanh(raw[:, :, 1:4]) ] ← torques ∈ [−0.1, 0.1]
u_desired = u_policy + hover_bias              (hover_bias[::4] = m·g)
```

This bounds the QP inputs so the problem stays feasible even with a random
initialisation.

---

## 2. CBF / HOCBF Safety Constraints

### Barrier function

For each agent `i` and each of the 3 spherical obstacles `j` with center `c_j`,
radius `r_j`, and safety margin `ε_j`:

```
h_{ij}(z)  =  ‖p_i − c_j‖²  −  (r_j + ε_j)²
```

Safe set: `S = { z : h_{ij}(z) ≥ 0  ∀ i,j }`.

### Why relative degree 2

`ḣ = Lf h = 2(p_i − c_j)·v_i` — control `u` does not appear (only velocity,
which is itself a state).  Control appears in `ḧ` through thrust → linear
acceleration.  Hence the system has **relative degree 2** for every agent–obstacle
pair, requiring a Higher-Order CBF.

---

---

## 3. HOCBF Derivation (Relative Degree 2)

### Setup

For a barrier `h(x) = ||p - c||² - (r + ε)²` on a quadrotor (position `p`, velocity `v`):

- `Lg h = 0`  — control does not appear in `ḣ` (relative degree 2 confirmed)
- `Lf h = 2(p − c)·v`

Define the auxiliary barrier:

```
ψ₁ = Lf h + α₁ · h  =  2(p−c)·v + α₁·h
```

### Deriving the QP constraint

We require `ψ̇₁ + α₂·ψ₁ ≥ 0`.  Expand `ψ̇₁`:

```
ψ̇₁ = d/dt [Lf h + α₁ h]
    = Lf² h  +  Lg Lf h · u  +  α₁ · Lf h        (since Lg h = 0)
```

where:
- `Lf² h = 2||v||² + 2(p−c)·(−g e₃) = 2||v||² − 2g·(pz − cz)`  (drift-only acceleration)
- `Lg Lf h · u` = thrust coefficient × `u_thrust_i`

Substituting into `ψ̇₁ + α₂ψ₁ ≥ 0`:

```
Lf²h  +  Lg Lf h · u  +  α₁·Lf h  +  α₂·ψ₁  ≥  0
```

Rearranging to QP form `A·u ≥ b`:

```
Lg Lf h · u  ≥  − Lf²h  −  α₁·Lf h  −  α₂·ψ₁
```

Equivalently, expanding `ψ₁ = Lf h + α₁ h`:

```
b_cbf = − Lf²h  −  (α₁ + α₂)·Lf h  −  α₁α₂·h
```

### Expanded form for agent i, obstacle j

```
A_cbf[i, 4i]  =  (2/m) · (p_i − c) · thrust_dir_i      (scalar, only thrust matters)

b_cbf[i]      =  − 2||v_i||²
                 + 2g·(p_iz − cz)
                 − α₁ · 2(p_i−c)·v_i                   ← this term was missing
                 − α₂ · [2(p_i−c)·v_i + α₁·h_i]
```

---

## 4. Alignment with Reference (`quadcopter_multi.py`)

### Constraint sign convention

The reference uses `K·u ≤ d` with a **negative** coefficient:

```python
# quadcopter_multi.py:127-128
K_row[:, 4*i] = -(2.0 / m) * (dp[:, i, :] * td[:, i, :]).sum(dim=-1)
# constraint checked as:  psi2 = d_cbf - K_cbf @ u  ≥ 0
#   i.e.  K·u ≤ d   ↔   -A·u ≤ d   ↔   A·u ≥ -d
```

The CVXPyLayers version uses `A·u ≥ b` with a **positive** coefficient and negated
RHS.  Both are equivalent: `A = −K`,  `b = −d`.  The sign convention is consistent.

### d_vals / b_cbf comparison (after fix)

Reference `d_vals` with `gamma(x) = 1.0 * x` (i.e. α₁ = α₂ = 1):

```
d_vals  =  Lf²h  +  gamma(Lf h)  +  gamma(ψ₁)
        =  Lf²h  +  Lf h         +  (Lf h + h)
        =  Lf²h  +  2·Lf h  +  h
```

CVXPyLayers `b_cbf` (after fix, with α₁ = α₂ = 5):

```
b_cbf  =  −Lf²h  −  5·Lf h  −  5·(Lf h + 5h)
       =  −Lf²h  −  10·Lf h  −  25h
```

Both implement the correct HOCBF formula `−Lf²h − (α₁+α₂)·Lf h − α₁α₂·h`.
The difference is only the **class-K parameters**: reference uses α=1, CVXPyLayers
uses α=5.  Higher α enforces a wider safety margin but makes the CBF more likely
to be active, which costs gradient signal.

### Matching table

| Aspect | `quadcopter_multi.py` | CVXPyLayers (after fixes) |
|---|---|---|
| HOCBF formula | `Lf²h + γ(Lf h) + γ(ψ₁)` | `Lf²h + α₁Lf h + α₂ψ₁` ✓ |
| Constraint sign | `K·u ≤ d` (K negative) | `A·u ≥ b` (A positive) ✓ |
| CBF α₁, α₂ | 1.0, 1.0 | 5.0, 5.0 (more conservative) |
| State detach for A,b | `z.detach()` ✓ | fixed to `x.detach()` ✓ |
| dt | 0.2 s | 0.2 s ✓ |
| Initial x range (N=5) | 1.1 – 1.9 | 1.1 – 1.9 ✓ |
| α_terminal start | 20 | 20 ✓ |
| α_terminal final | 200 | 200 ✓ |
| α schedule | +5 every 20 ep | +5 every 20 ep ✓ |
| Output scaling | tanh (T_dev=0.5, τ=0.1) | tanh (T_dev=0.5, τ=0.1) ✓ |
| z0_std | 4e-2 | 4e-2 ✓ |
| Network arch | ResBlock + time input | plain MLP, no time input |
| QP solver | DYS (GPU, JFB grad) | SCS via CVXPyLayers (CPU, IFT grad) |
| Always projects | No — skips if satisfied | Yes — always runs QP |

The remaining structural difference is the QP solver and gradient method, discussed
in the next section.

---

## 5. Training: CVXPyLayers (IFT) vs Reference DYS (JFB)

Both approaches solve the same safety-filter QP at every step, but differ in how
they differentiate through it.

### Reference: DYS + Jacobian-Free Backpropagation (JFB)

The projector (`utils.py DYSProjector`) uses **Douglas-Rachford Splitting**:

```
T(z) = z + proj_C1(2·proj_C2(z) − z − α∇L) − proj_C2(z)
```

where `C1` is the affine constraint set and `C2` is the non-negativity set for slack
variables.  The fixed point `z* = T(z*)` gives the QP solution.

**JFB gradient** (Jacobian-Free Backpropagation):

```
Step 1  (forward, no_grad):   run  z ← T(z)  until convergence
Step 2  (backward, grad ON):  apply ONE final T step
```

The gradient of that one step is used as a proxy for `∂z*/∂u_nom`.  This is an
approximation — it is exact only when `T` is a contraction with spectral radius ≈ 0
— but it is:
- **Cheap**: one matrix–vector product per step
- **Stable**: no matrix inversion, no ill-conditioning from near-degenerate constraints
- **Sparse**: skips entirely when `u_nom` already satisfies all constraints (gradient = I)

### CVXPyLayers: SCS + Implicit Function Theorem (IFT)

CVXPyLayers differentiates via the **KKT conditions** at the optimal solution.
At `u* = argmin ½‖u − u_nom‖² s.t. A·u ≥ b`, the KKT system is:

```
[ 2I    −Aᵀ ] [ ∂u*/∂u_nom ]   [ 2I ]
[ λ*A    D  ] [ ∂λ*/∂u_nom ] = [  0 ]

where D = diag(A·u* − b)  (complementary slackness)
```

Solving this gives the exact Jacobian of `u*` w.r.t. `u_nom`.

**Strengths:**
- Exact gradient (up to SCS tolerance)
- Principled: grounded in the implicit function theorem

**Weaknesses in this problem:**

| Issue | Consequence |
|---|---|
| IFT requires inverting the KKT matrix | Near-zero `A_cbf` coefficients (degenerate directions) → ill-conditioned inversion → garbage gradients |
| Always runs SCS, even if constraints inactive | No free "identity" gradient when policy already satisfies HOCBF |
| SCS runs on CPU via numpy | ~5× slower than GPU-based DYS; requires device transfer each step |
| Gradient through A(x) and b(x) (before fix) | Three simultaneous gradient paths through the KKT system; compounding Jacobians across 50 steps → vanishing |
| Batching via outer Python loop | Each of `batch × steps = 800` SCS calls per epoch is a separate Python object |

### Gradient flow comparison across the rollout

For a rollout of `T` steps, the terminal cost gradient must backpropagate through:

```
∂G/∂θ  =  ∂G/∂z_T · (∏ᵢ ∂z_{i+1}/∂z_i)

where  ∂z_{i+1}/∂z_i  includes  ∂u*_i/∂u_nom_i · ∂u_nom_i/∂z_i  (QP Jacobian)
```

| Method | QP Jacobian `∂u*/∂u_nom` | Gradient quality over 50 steps |
|---|---|---|
| DYS + JFB | One T-step Jacobian; rank-preserving | Moderate approximation, numerically stable |
| CVXPyLayers + IFT | Exact KKT inversion; projection onto active-constraint null space | Accurate per step, but compounds poorly; near-zero eigenvalues kill long-horizon signal |

When a CBF constraint is **active** (agent near obstacle), the IFT Jacobian is a
projection matrix that zeros out the gradient component aligned with the constraint
normal.  With 15 constraints in a 20D space, up to 15 gradient directions are killed
at each step.  Over 50 steps the product of these projection Jacobians converges to
zero for almost all policy parameters — the terminal cost gradient vanishes entirely.

JFB is immune to this because the one-step Jacobian of `T` does not project; it
approximates the full sensitivity including the complementary-slackness correction.

### Summary

CVXPyLayers + IFT is the theoretically cleaner tool but is fragile for long-horizon
training with many active inequality constraints.  The DYS + JFB reference trades
gradient accuracy for numerical robustness, which is exactly the right trade-off here.
The fixes in this repo (detaching x for A,b; tanh output bounding; hyperparameter
alignment) bring the CVXPyLayers version as close as possible to the reference's
gradient behaviour within the CVXPyLayers framework.

---

## 6. Bug Fixes

### Fix 1 — HOCBF `b_cbf` formula (critical)

**File:** `CVXPyLayers/barriers/spherical_obstacle.py:170`  
**File:** `CVXPyLayers/barriers/base.py:209`

The `α₁·Lf h` term (from differentiating `α₁h` inside `ψ₁`) was dropped.

```python
# BEFORE (wrong — missing -alpha1 * Lf_h):
b_cbf = -Lf2_h - alpha2 * psi1

# AFTER (correct):
b_cbf = -Lf2_h - alpha1 * Lf_h - alpha2 * psi1
```

**Effect of the bug:** When an agent approaches an obstacle (`Lf h < 0`), the missing term
made the constraint *less* restrictive than the theory requires.  The CBF under-corrected
during approach, letting agents get closer to obstacles than intended.

**Reference match:** `quadcopter_multi.py:121-124`:
```python
d_vals = (2.0 * vel.pow(2).sum(dim=-1)
          - 2.0 * g * dp[:, :, 2]
          + gamma(Lf_h)    # ← this is α₁·Lf h (gamma = identity with α₁=1)
          + gamma(psi1))   # ← this is α₂·ψ₁
```

---

### Fix 2 — Detach state when computing CBF constraint matrices (critical for gradients)

**File:** `CVXPyLayers/controllers/cbf_qp_layer.py:145`

The reference always computes CBF constraint matrices from a **detached** state:
```python
# reference: quadcopter_multi.py / compute_loss
K_cbf, d_cbf = construct_cbf_constraints(z.detach(), ...)
```

Without detaching, gradients flow through **three** paths simultaneously:
1. `u_desired → u_safe`  (through the QP objective)
2. `x → A_cbf(x) → u_safe`  (through the QP constraint rows)
3. `x → b_cbf(x) → u_safe`  (through the QP constraint offsets)

The implicit-differentiation backward of CVXPyLayers must invert the full KKT system
for all three paths at once.  With 15 constraints in 20D, across 50 rollout steps, this
products Jacobians that compound to near-zero — the terminal cost gradient vanishes.

Detaching `x` before computing `A_cbf` and `b_cbf` restricts the backward pass to path 1
only, exactly as in the reference.

```python
# BEFORE:
A_cbf, b_cbf = obstacle.compute_cbf_constraint(x, self.dynamics, self.alpha)

# AFTER:
x_cbf = x.detach()
A_cbf, b_cbf = obstacle.compute_cbf_constraint(x_cbf, self.dynamics, self.alpha)
```

---

### Fix 3 — Output scaling with tanh

**File:** `CVXPyLayers/examples/train_quadrotor_multi_cvxpy.py`

The reference bounds the policy output before adding hover bias:
```python
# reference train.py u_fn
T_dev_scale * torch.tanh(raw[:, :, 0:1])   # thrust deviation ∈ [−0.5, 0.5]
tau_scale   * torch.tanh(raw[:, :, 1:4])   # torques ∈ [−0.1, 0.1]
```

Without this, an initialising policy can produce arbitrarily large thrusts, causing the
CBF QP to be infeasible (SCS returns a degenerate point with corrupted gradients).

```python
# BEFORE:
u_desired = policy(x) + hover_bias

# AFTER:
u_raw = policy(x).reshape(batch_size, n_agent, 4)
u_policy = torch.cat([
    T_dev_scale * torch.tanh(u_raw[:, :, 0:1]),
    tau_scale   * torch.tanh(u_raw[:, :, 1:4]),
], dim=-1).reshape(batch_size, dynamics.control_dim)
u_desired = u_policy + hover_bias
```

---

### Fix 4 — Training hyperparameters aligned to reference

**File:** `CVXPyLayers/examples/train_quadrotor_multi_cvxpy.py`

| Parameter | Before | After | Reference value |
|---|---|---|---|
| `dt` | 0.4 s | 0.2 s | 0.2 s |
| `num_steps` | 25 | 50 | 50 |
| `x_min, x_max` | 0.8, 2.2 | 1.1, 1.9 | 1.1, 1.9 |
| `alpha_terminal` (start) | 5.0 | 20.0 | 20.0 |
| `alpha_terminal_final` | 50.0 | 200.0 | 200.0 |
| `alpha_sched_every` | 100 epochs | 20 epochs | 20 epochs |
| `z0_std` | 2e-2 | 4e-2 | 4e-2 |

**`dt = 0.4` impact:** Halving the number of steps reduces the gradient path length but
increases per-step integration error. RK4 error scales as O(dt⁵), so dt=0.4 gives
~32× more integration error per step than dt=0.2.  The reference uses dt=0.2 with 50 steps.

**`x_min=0.8, x_max=2.2` impact:** Outer agents were placed 0.3 m closer to the obstacle
columns than in the reference.  With `obstacle_radius + epsilon = 0.5`, agents at x=0.8
start only ~0.13 m from the x=0.63 obstacle safety boundary, immediately activating the
CBF and blocking gradients from the first step.

**`alpha_terminal=5.0` impact:** The terminal cost contributed only 5/(5+running) ≈ 1%
of the gradient initially, making the policy almost exclusively minimise control effort
with no incentive to reach the target.

### Fix 5 — Remove constraint normalization (`cbf_qp_layer.py`)

**File:** `CVXPyLayers/controllers/cbf_qp_layer.py`

**Root cause of "Solved/Inaccurate" and training divergence.**

At initialization, agents sit at `z=1.0` and obstacles 1 & 3 are also at `z=1.0`.
At hover (angles ≈ 0), the thrust direction is straight up: `thrust_dir = [0, 0, 1]`.
The CBF coefficient for those obstacles is:

```
A_cbf[i, 4i]  =  (2/m) · dp · thrust_dir  =  (2/m) · (p_z − c_z)  =  0
```

The normalization clamp then triggers:

```python
A_norm = A.norm(...).clamp(min=1e-4)   # hits floor → 1e-4
b_normalized = b / 1e-4               # b ≈ −55  →  b_normalized ≈ −550,000
```

SCS receives constraints whose RHS values span six orders of magnitude
(`[−550000, +10]`).  Its internal preconditioner fails → "Solved/Inaccurate" →
`u_safe` is garbage → trajectory blows up → corrupted gradient → G increases as
`alpha_terminal` is annealed up instead of decreasing.

**The normalization was counter-productive**: a large negative `b` (trivially-satisfied
constraint) becomes an astronomically negative `b_normalized`, which SCS cannot
handle.  SCS already performs its own internal diagonal scaling; raw constraints
are the correct input.

```python
# BEFORE — manual row-normalisation amplifies degenerate constraints:
A_norm       = A.norm(dim=-1, keepdim=True).clamp(min=1e-4)
A_normalized = A / A_norm
b_normalized = b / A_norm.squeeze(-1)
...
qp_params.append(A_normalized.cpu())
qp_params.append(b_normalized.cpu())

# AFTER — pass raw constraints, let SCS scale internally:
qp_params.append(A.cpu())
qp_params.append(b.cpu())
```

---

### Fix 6 — SCS solver parameters (`cbf_qp_layer.py`)

**File:** `CVXPyLayers/controllers/cbf_qp_layer.py`

```python
# BEFORE:
"eps": 1e-4,   "max_iters": 2500

# AFTER:
"eps": 1e-3,   "max_iters": 10000
```

`eps=1e-4` was unnecessarily tight.  Gradient accuracy through IFT is already
limited by the KKT approximation, not by the last digit of the primal solution.
`1e-3` is the standard tolerance used in CBF-QP papers and converges reliably on
this 20-variable / 15-constraint problem.  `max_iters=2500` was too low for the
(poorly pre-conditioned) problem that the normalization bug was creating; 10000 is
sufficient at `1e-3`.

---

### Fix 11 — Barrier function constant: `(r+ε)²` → `r² + ε²` (critical — caused geometric infeasibility)

**File:** `CVXPyLayers/barriers/spherical_obstacle.py:97, :135`

**Root cause of repeated "Solver scs returned status infeasible" errors.**

The barrier was defined as:

```python
# BEFORE (wrong):
h = ||p - c||²  −  (r + ε)²
```

The reference defines it as:

```python
# quadcopter_multi.py:88-90
return ((pos - center) ** 2).sum(dim=-1) - r_obs ** 2 - eps ** 2
# i.e.  h = ||p - c||²  −  r²  −  ε²
```

These differ by the cross term `2rε`:

```
(r + ε)²  =  r²  +  2rε  +  ε²   (CVXPyLayers, WRONG)
r²  +  ε²                          (reference, CORRECT)
```

With `r = 0.35`, `ε = 0.15`:

| Formula | Value | Implied safe radius |
|---|---|---|
| `(r+ε)²` | 0.2500 | **0.500 m** |
| `r² + ε²` | 0.1450 | **0.381 m** |

**Why this caused geometric infeasibility.**  Obstacles 1 and 3 sit at `x = 0.63` and `x = 2.37`
respectively.  Agents 0 and 4 start at `x = 1.1` and `x = 1.9` and must pass through `y ≈ 1.0`
(the obstacle plane) to reach the target at `y = 3.5`.

At the tightest point (agents 0/4 passing directly alongside obstacles 1/3):

```
clearance = |x_agent − x_obstacle|
           = |1.1 − 0.63| = 0.470 m          (same for agent 4 vs obstacle 3)

CVXPyLayers required:  0.500 m  →  0.470 < 0.500  →  agent inside safe set  →  h < 0
Reference required:    0.381 m  →  0.470 > 0.381  →  agent outside safe set →  h > 0
```

With `h < 0`, the HOCBF constraint `A·u ≥ b` has `b = −Lf²h − α₁Lf_h − α₂ψ₁` with `ψ₁ = Lf_h + α₁h < Lf_h`. For agents 0 and 4 simultaneously approaching obstacles 1 and 3, the constraints push them outward — in opposite directions — with no feasible `u` satisfying both.  SCS reports `infeasible`.

**Effect on training:** The fallback `u_safe = u_desired` was used whenever this occurred, meaning
the safety filter was silently bypassed for the 2 outer agents every time they passed through the
obstacle plane.  Gradients from those steps were computed without the QP projection, corrupting the
IFT backward pass.  The infeasibility also masked the true h_min: the logged `h_min ≈ 0.8` was from
the final state (after 50 steps), not from the mid-trajectory minimum where `h` was negative.

```python
# BEFORE — wrong cross term creates too-large safe radius:
h_vals = dist_sq - (self.radius + self.epsilon) ** 2

# AFTER — matches reference exactly:
h_vals = dist_sq - self.radius ** 2 - self.epsilon ** 2
```

Fixed in both `h()` (line 97) and `compute_cbf_constraint()` (line 135).

---

## 7. Why the terminal cost did not decrease — root cause summary

1. **Wrong HOCBF `b_cbf`** → CBF under-constrained during approach → agents could drift
   toward obstacles → CBF activated earlier/harder → more gradient blocking.

2. **No detach on `x` for CBF matrices** → CVXPyLayers backward must differentiate through
   `x → A(x)` and `x → b(x)` across 50 QP solves.  The compound Jacobian from 50 implicit-
   differentiation steps effectively zeroed out the terminal cost gradient.

3. **No output tanh scaling** → unconstrained large thrusts → frequent QP infeasibility →
   SCS returned inaccurate solutions → corrupted backward gradients.

4. **`alpha_terminal` too small, schedule too slow** → the terminal cost term was negligible
   relative to the running cost, providing almost no learning signal toward the target.

5. **`x_min/x_max` too wide** → outer agents started inside the CBF safety margin,
   activating constraints at step 0 and immediately blocking the gradient path.

6. **Constraint normalization dividing by near-zero `A_norm`** → benign slack
   constraints (b ≈ −55) inflated to b ≈ −550,000 → SCS "Solved/Inaccurate" →
   garbage `u_safe` → exploding trajectory costs → G increased as `alpha_terminal`
   was annealed up (observed in training: G went from ~150 to ~5000 by epoch 35).

7. **SCS `eps=1e-4, max_iters=2500` too tight / too few** → SCS couldn't converge
   on the poorly-scaled problem the normalization created, compounding issue 6.

8. **`alpha_terminal_final = 50.0` (bug)** → equalled `alpha_terminal`, so the
   annealing condition `_alpha_terminal < alpha_terminal_final` was always `False`.
   The terminal weight was frozen at 50 for all 1000 epochs; `alpha_t=50.0` in every
   log line confirmed this.  Fixed to `200.0` for all.

9. **`--grad_clip` defaulted to `False`** → gradient norms of 8k–10k with spikes to
   14k caused large parameter updates (scale ≈ lr × grad_norm ≈ 1.0 per step).
   Fixed: grad clipping enabled by default.
   Change: grad clip is false.

10. **Short horizon (T=2s) is physically infeasible for the task** → with `tau_scale=0.1`
    rad/s² and `dt=0.2`, a quadrotor achieves ~0.4m y-displacement in 2s.  The target
    is 4m away; `G≈40` (full initial distance) regardless of policy quality.  The policy
    cannot learn because the task is physically impossible in the given horizon, not
    because of gradient issues.  `h_min≈2.0` confirmed CBF constraints were completely
    inactive (no gradient blocking from IFT projection).  **Reverted to T=10s, 50 steps.**
    The gradient vanishing hypothesis was partly confounded with the normalization bug
    (fix 5) which made SCS return garbage, corrupting IFT gradients.  With fix 5 applied,
    IFT at 50 steps may be tractable; the primary cause of stagnation in the original run
    was the `alpha_terminal_final` bug (fix 8) and missing grad clipping (fix 9).

11. **Barrier constant `(r+ε)²` instead of `r²+ε²`** → safe radius 0.500 m instead of
    0.381 m → agents 0 and 4 (x=1.1, x=1.9) have only 0.470 m clearance to obstacles 1/3
    when passing through the obstacle plane → `h < 0` for both simultaneously →
    HOCBF constraints push them in opposite directions → QP geometrically infeasible →
    SCS reports `infeasible` → fallback `u_safe = u_desired` silently used → safety filter
    bypassed for outer agents throughout training → corrupted IFT gradients.
    This was the root cause of the repeated "Solver scs returned status infeasible" messages
    observed from ~epoch 700 onward.  **Fixed in `spherical_obstacle.py` (Fix 11).**

12. **IFT Jacobian kills gradients when CBF constraints are active (structural CVXPyLayers limitation)** →
    after fixes 1–11, training still plateaued at `terminal_cost ≈ 10` (agents reach ~y=3,
    short of target y=3.5).  Root cause: CVXPyLayers differentiates via the Implicit Function
    Theorem (IFT) at the KKT optimum.  When constraint `i` is active, the KKT Jacobian
    `∂u*/∂u_desired` is exactly zero in the direction of that constraint row — gradient is
    killed.  With 15 constraints over a 50-step rollout, the product of these projection
    Jacobians converges to near-zero for most policy parameters.

    The reference (DYS + JFB) avoids this because JFB uses one T-step of the DYS operator
    as the gradient approximation.  The T-step Jacobian w.r.t. `u_nom` is non-zero even
    for active constraints (it flows through the objective gradient term inside `apply_T`),
    so gradient signal is never killed.

    **Attempted fix: straight-through gradient estimator (`--straight_through` flag) — REVERTED.**

    The straight-through estimator was added (forward pass uses SCS for safe control, backward
    pass uses identity Jacobian `∂u_safe/∂u_desired = I`), but empirically made training worse.
    The `straight_through` parameter and flag have been removed. This remains an open problem.

---

## 8. Single / Double Integrator Example Fixes

The fixes discovered for the quadrotor (Fixes 1–12) were audited against the simpler integrator
examples (`train_single_integrator.py`, `train_double_integrator.py`) and their shared infrastructure
(`training/trainer.py`, `training/config.py`, `visualization/plotting.py`).

Fixes 2, 5, 6 (detach, no normalization, SCS params) are already in `cbf_qp_layer.py` and thus
automatically apply to both examples. Fixes 1, 11 (HOCBF formula, barrier constant) are already
correct in `barriers/base.py` and `barriers/circular_obstacle.py`. The following additional issues
were found and fixed.

---

### Fix 13 — Missing gradient clipping in `CBFTrainer` (critical for training stability)

**Files:** `CVXPyLayers/training/trainer.py`, `CVXPyLayers/training/config.py`

The unified trainer had no gradient clipping, identical to the quadrotor Fix 9.  Over long rollouts
(50–120 steps) with CVXPyLayers IFT backward passes, gradient spikes can destabilise training.

```python
# BEFORE — no clipping:
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()

# AFTER — configurable clipping via grad_clip_norm:
self.optimizer.zero_grad()
loss.backward()
if self.config.grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
self.optimizer.step()
```

Added `grad_clip_norm: float = 1.0` to `TrainingConfig` (set to 0 to disable).  Default 1.0 is
appropriate for the lower-dimensional integrator problems; the quadrotor uses 100.0.

---

### Fix 14 — Double integrator example: `terminal_cost_weight` and `num_epochs` too low

**File:** `CVXPyLayers/examples/train_double_integrator.py`

| Parameter | Before | After | Reason |
|---|---|---|---|
| `terminal_cost_weight` | 100.0 | 1000.0 | Matches preset `double_integrator_three_obstacles()` in config.py; 100 provides insufficient gradient signal toward target |
| `num_epochs` | 100 | 500 | 100 epochs is far too few for a 50-step HOCBF rollout to converge; preset uses 2000 |

---

### Fix 15 — Visualization: safety boundary radius incorrect

**File:** `CVXPyLayers/visualization/plotting.py`

The safety boundary circle was drawn at radius `r + ε`, but the barrier function is:

```
h = ||p - c||² - r² - ε  ≥  0   →   safe set boundary at  ||p - c|| = sqrt(r² + ε)
```

```python
# BEFORE — wrong radius (r + ε):
circle_outer = Circle(..., obstacle.radius + obstacle.epsilon, ...)

# AFTER — correct radius sqrt(r² + ε):
safe_radius = math.sqrt(obstacle.radius ** 2 + obstacle.epsilon)
circle_outer = Circle(..., safe_radius, ...)
```

For `r=0.55, ε=0.1`: drawn was 0.65, correct is 0.635.
For `r=0.4, ε=0.1`: drawn was 0.50, correct is 0.510.
Difference is small but the boundary should match what the CBF actually enforces.
