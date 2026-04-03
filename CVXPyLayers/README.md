# CVXPyLayers Multi-Quadrotor CBF Training

Reference implementation to compare against: `../quadcopter_multi.py` + `../train.py` (DYS/JFB).  
Bug-fix log and derivations: `CHANGES.md`.

---

## What CVXPyLayers is doing here

At every rollout step, the policy outputs a **desired control** `u_desired`.  A
safety filter then solves a small QP to find the closest safe control:

```
u_safe = argmin_u  ½‖u − u_desired‖²
         s.t.      A_cbf · u ≥ b_cbf      (one row per agent per obstacle)
```

**CVXPyLayers turns this QP into a differentiable layer.**  In the forward pass it
calls the SCS solver (via `diffcp`) to get `u_safe`.  In the backward pass it uses
the implicit function theorem (IFT) on the KKT conditions to compute
`∂u_safe / ∂u_desired`, so gradients flow from the trajectory loss back through
the safety filter into the policy network.

The full computation graph per timestep:

```
policy(x)  →  u_desired
                  ↓
   A_cbf(x.detach()), b_cbf(x.detach())   ← CBF constraint matrices (no grad)
                  ↓
         CVXPyLayers QP (SCS solver)
                  ↓
              u_safe
                  ↓
         dynamics.step(x, u_safe, dt)      ← RK4 integration
                  ↓
              x_next
```

Gradients flow backward only through `u_desired → u_safe` (IFT).  The constraint
matrices are computed from a detached state so they are treated as constants.

---

## File map

### `examples/train_quadrotor_multi_cvxpy.py` — training entry point

| Section | What it does |
|---|---|
| Lines 44–86 | Build `Quadrotor` dynamics and `SphericalObstacle` list |
| Lines 88–103 | Define `p_target` and `target_state` |
| Lines 105–115 | Build `CBFQPController` (creates the CVXPyLayers QP layer) |
| Lines 118–136 | Set cost weights, alpha schedule, training hyperparams |
| Lines 143–153 | Build `PolicyNetwork` |
| Lines 222–300 | **Training loop**: rollout → costs → backward → step |
| Lines 242–258 | Per-step: `policy(x)` → tanh scale → hover bias → `filter_control` → `dynamics.step` |
| Lines 265–282 | Running cost (control effort + velocity) and terminal cost (position + attitude) |
| Lines 343–344 | Alpha schedule: increment `alpha_terminal` every `alpha_sched_every` epochs |
| Lines 348–351 | LR decay |

---

### `controllers/cbf_qp_layer.py` — the CVXPyLayers QP

**`create_cbf_qp_layer(control_dim, num_constraints)`**

Creates the CVXPY problem once at startup:
```
minimize  ‖u‖² − 2·u_desired^T·u
s.t.      A_i · u ≥ b_i   for i = 1…num_constraints
```
Wraps it in a `CvxpyLayer` so it is callable as a batched differentiable function.
Parameters are `[u_desired, A_1, b_1, A_2, b_2, …]` — all passed at call time.

**`CBFQPController.filter_control(x, u_desired)`**

Called at every rollout step.  Does:
1. Detach `x` → compute `A_cbf`, `b_cbf` for every obstacle (no gradient through constraints)
2. Flatten per-agent rows into a list of 15 constraint pairs (3 obs × 5 agents)
3. Move all tensors to CPU (SCS runs via numpy)
4. Call `self.qp_layer(u_desired, A_1, b_1, …)` → returns `u_safe`
5. Move `u_safe` back to original device

**SCS solver settings** (inside `filter_control`):
```python
"eps": 1e-3       # primal/dual tolerance — 1e-3 is reliable; tighter causes "Solved/Inaccurate"
"max_iters": 10000 # plenty for 20-var/15-constraint QP at eps=1e-3
"acceleration_lookback": 10  # Anderson acceleration window for SCS 2.x
```

---

### `barriers/spherical_obstacle.py` — HOCBF constraint computation

**`SphericalObstacle.h(x)`**

Barrier value for all agents at once:
```
h(x)[batch, agent] = ‖p_i − c‖² − (r + ε)²
```
Returns `(batch, n_agent)`.  Positive = safe, negative = inside obstacle.

**`SphericalObstacle.compute_cbf_constraint(x, dynamics, alpha)`**

Computes `A_cbf` and `b_cbf` for the HOCBF QP constraint.

Steps inside:
1. Extract `dp = p_i − c`,  `v_i` from state `x`
2. `h = ‖dp‖² − (r+ε)²`
3. `Lf h = 2·dp·v`  (first Lie derivative, drift only)
4. `ψ₁ = Lf h + α₁·h`  (auxiliary barrier)
5. `Lf²h = 2‖v‖² − 2g·dp_z`  (second Lie derivative, drift only)
6. `thrust_dir = dynamics.thrust_direction(angles)`
7. `A_cbf[:, i, 4*i] = (2/m)·dp_i·thrust_dir_i`  (only thrust affects position)
8. `b_cbf[:, i] = −Lf²h − α₁·Lf h − α₂·ψ₁`

Returns `A_cbf: (batch, n_agent, 4*n_agent)`,  `b_cbf: (batch, n_agent)`.

---

### `dynamics/quadrotor.py` — quadrotor physics

**`Quadrotor.f(x)`** — drift vector field (no control):
```
[ṗ, ȧngles, −g·e₃, 0]   per agent
```

**`Quadrotor.g(x)`** — control matrix:
- Thrust `u[4i]` enters `v̇_i` as `(thrust_dir / m)`
- Torques `u[4i+1:4i+4]` enter `ang_vel_i` directly via identity

**`Quadrotor.step(x, u, dt)`** — RK4 integration.

**`Quadrotor.thrust_direction(angles)`** — converts `(ψ, θ, φ)` Euler angles to
unit thrust vector in world frame.

---

### `controllers/policy_network.py` — neural policy

Plain MLP: `state_dim → hidden_dim × n_layers → control_dim`.

Output is raw deviations.  The training script applies tanh scaling before adding
hover bias:
```python
u_desired = [ T_dev_scale·tanh(raw[thrust]),  tau_scale·tanh(raw[torques]) ] + hover_bias
```

---

## Knobs and what they do

### Safety / HOCBF

| Parameter | Location | Effect |
|---|---|---|
| `cbf_alpha = (α₁, α₂)` | `train_quadrotor_multi_cvxpy.py:106` | Controls how aggressively the CBF enforces safety. Higher → larger safety margin, but CBF activates more often and blocks more gradient directions. Reference uses `(1,1)`; current is `(5,5)`. |
| `obstacle_radius` | `train_quadrotor_multi_cvxpy.py:69` | Physical obstacle radius (m). |
| `epsilon` | `train_quadrotor_multi_cvxpy.py:69` | Safety margin added to radius. Safe set boundary is at `r + ε`. |

**How `α₁, α₂` affect training:** The QP constraint RHS is
`b = −Lf²h − (α₁+α₂)·Lf h − α₁α₂·h`.  Large α₁α₂ makes the constraint activate
at larger distances (more conservative), which means the CBF QP is active more
often. When active, IFT projects out gradient components aligned with the active
constraint normals — with 15 constraints in 20D, up to 15 of 20 gradient directions
can be zeroed per step.

---

### Cost weights and schedule

| Parameter | Location | Effect |
|---|---|---|
| `alpha_running` | line 119 | Weight on `∫ ½‖u‖² + 2‖v‖² dt`. Too high → policy minimises effort but ignores target. |
| `alpha_terminal` (start) | line 120 | Initial weight on terminal position/attitude error. Too low → policy ignores target early in training. |
| `alpha_terminal_final` | line 121 | Cap for the schedule. |
| `alpha_sched_step` | line 122 | How much to increment `alpha_terminal` each schedule step. |
| `alpha_sched_every` | line 123 | Epochs between increments. Smaller → faster annealing. |

**Rule of thumb:** `alpha_terminal` should dominate early so the target gradient is
not swamped by the running cost.  Reference starts at 20 and increments by 5 every
20 epochs.  If G is not decreasing, check that `alpha_terminal / alpha_running` is
large enough relative to the cost magnitudes.

---

### Training dynamics

| Parameter | Location | Effect |
|---|---|---|
| `dt` | line 62 | Integration timestep. Smaller → more rollout steps, longer gradient path, more accurate physics. `dt=0.2` (50 steps) matches reference. |
| `T` | line 61 | Total horizon (s). |
| `batch_size` | line 132 | Samples per gradient estimate. Larger → less noisy gradient, more memory. |
| `z0_std` | line 133 | Noise on initial `xy` positions. Too small → policy overfits to one trajectory. Reference uses `4e-2`. |
| `learning_rate` | line 130 | Adam LR. `1e-4` is safe; can try `5e-4` if G is not moving. |
| `lr_decay_epoch` | line 131 | Halve LR every N epochs. |
| `grad_clip_norm` | line 136 | Max gradient norm before optimizer step. With `grad_norm ≈ 2e4`, clip at `100` reduces effective signal by 200×. Try `1000` if G is stuck. |
| `weight_decay` | line 154 | L2 regularization on policy weights. |

---

### Policy output scaling

| Parameter | Location | Effect |
|---|---|---|
| `T_dev_scale = 0.5` | line 141 | Max thrust deviation from hover: `u_thrust ∈ [T_hover − 0.5, T_hover + 0.5]`. Bounds the QP input so SCS stays feasible. |
| `tau_scale = 0.1` | line 142 | Max torque magnitude. Larger → more aggressive attitude changes, but can cause numerical issues. |

These bound the control BEFORE the CBF filter.  If `u_desired` is huge, the QP
feasibility set may be empty and SCS returns "Solved/Inaccurate".

---

### SCS solver

| Parameter | Location | Effect |
|---|---|---|
| `eps` | `cbf_qp_layer.py:139` | Primal/dual feasibility tolerance. `1e-3` is reliable. Tighter (e.g. `1e-4`) frequently triggers "Solved/Inaccurate" when constraint matrix is near-degenerate. |
| `max_iters` | `cbf_qp_layer.py:140` | Max SCS iterations. `10000` is safe at `eps=1e-3`. |
| `acceleration_lookback` | `cbf_qp_layer.py:141` | Anderson acceleration window. Leave at `10`. |

---

### Initial conditions

| Parameter | Location | Effect |
|---|---|---|
| `x_min, x_max` | line 91 | x-range for agent positions. Must match reference `(1.1, 1.9)` for 5 agents to keep agents away from the x=0.63 and x=2.37 obstacle columns at startup. |
| Initial y | line 214 | Fixed at `−0.5`. Agents start 4 m below targets. |
| Initial z | line 215 | Fixed at `1.0`. Same z as obstacles 1 & 3 — means `A_cbf ≈ 0` at hover (thrust perpendicular to obstacle direction). |

---

## Diagnostic guide

| Symptom | Likely cause | Fix |
|---|---|---|
| `Solved/Inaccurate` warning | Near-zero `A_cbf` coefficient → normalization inflates `b`. Or `u_desired` too large. | Remove constraint normalization (done). Check tanh bounds. |
| G increases as `alpha_terminal` rises | SCS returning garbage → corrupted gradients amplified by larger weight | Fix SCS stability (eps, normalization) |
| G flat / not decreasing | Gradient vanishing through 50 QP steps; or `grad_clip_norm` too small | Increase `grad_clip_norm`; reduce `cbf_alpha` to reduce active-constraint blocking |
| G decreasing but `h_min < 0` | CBF not enforcing safety — check HOCBF formula (`b_cbf` missing `α₁·Lf h`) | Fix `b_cbf` in `spherical_obstacle.py` |
| Training diverges (costs → 1e6+) | Unclamped policy output → infeasible QP | Add tanh output scaling |
| Very slow per-epoch time (>15s) | SCS called for all 50 steps × batch × 16 QPs, all on CPU | Expected; reference DYS runs on GPU |
