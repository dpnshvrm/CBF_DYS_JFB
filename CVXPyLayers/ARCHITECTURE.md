# CVXPyLayers CBF Framework Architecture

**A Deep Dive into Differentiable Control Barrier Functions**

---

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Framework Components](#framework-components)
4. [How the Three Examples Work](#how-the-three-examples-work)
5. [The Training Loop](#the-training-loop)
6. [Mathematical Background](#mathematical-background)
7. [Data Flow](#data-flow)

---

## Overview

### What is CVXPyLayers?

CVXPyLayers is a library that makes **convex optimization layers differentiable**. This means you can:
1. Solve a convex optimization problem (like a QP)
2. **Backpropagate through the solution** to train a neural network

### What Are We Building?

We're training neural network **policies** that:
- Output desired control actions `u_desired`
- Are filtered through a **CBF-QP safety layer**
- Produce safe controls `u_safe` that avoid obstacles
- The whole pipeline is end-to-end differentiable!

### The Magic

```
Policy NN → u_desired → [CBF-QP via CVXPyLayers] → u_safe → Dynamics → Next state
            ↑__________ Gradients flow back through QP! ___________________↓
```

---

## Core Concepts

### 1. Control Barrier Functions (CBF)

**Purpose:** Guarantee safety by constraining control to keep system in safe set.

**Barrier Function `h(x)`:**
- `h(x) > 0` ⇒ System is SAFE
- `h(x) = 0` ⇒ Boundary of safe set
- `h(x) < 0` ⇒ UNSAFE (collision!)

**Example - Circular Obstacle:**
```python
h(x) = ||p - c||² - (r + ε)²
```
Where:
- `p` = robot position
- `c` = obstacle center
- `r` = obstacle radius
- `ε` = safety margin

### 2. Relative Degree

**Critical concept** that determines CBF constraint form:

**Relative Degree 1** (e.g., Single Integrator `ẋ = u`):
- Control appears in first derivative
- **Standard CBF constraint:** `Lg h · u ≥ -Lf h - α·h`

**Relative Degree 2** (e.g., Double Integrator `ẍ = a`, Quadrotor):
- Control appears in second derivative
- **HOCBF (Higher-Order CBF):**
  - Define `ψ₁ = Lf h + α₁·h`
  - Constraint: `Lg Lf h · u ≥ -Lf² h - α₂·ψ₁`

### 3. The CBF-QP

At each timestep, solve:
```
minimize:    ||u - u_desired||²
subject to:  A_i · u ≥ b_i    for all obstacles i
```

Where `A_i, b_i` encode the CBF constraints.

### 4. CVXPyLayers Makes It Differentiable

**Without CVXPyLayers:**
- Solve QP → Get `u_safe`
- Can't compute `∂u_safe/∂u_desired` → Can't train policy!

**With CVXPyLayers:**
- Wraps QP solver (OSQP, ECOS, SCS)
- Automatically computes gradients via **implicit differentiation**
- Policy network receives gradients through the QP!

---

## Framework Components

### Component 1: Dynamics (`dynamics/`)

**What it does:** Defines how the system evolves.

**Base Class:** `ControlAffineDynamics`
```python
ẋ = f(x) + g(x)·u
```

**Key Methods:**
- `f(x)`: Drift dynamics (what happens with no control)
- `g(x)`: Control matrix (how control affects state)
- `step(x, u, dt)`: Integrate forward one timestep

#### Why split `f` and `g` separately?

The reference implementation (`quadcopter_multi.py`) fuses everything into one function:
```python
def f(z, u, t):           # reference: computes ẋ = f(x) + g(x)u in one shot
    dv = (u_thrust/m) * td - g * e3   # drift + control fused together
    ...
```

We split them because **`CBFQPController` must build the constraint matrix `A_cbf`
separately from the RHS `b_cbf`**.  Specifically, `A_cbf = Lg Lf h` — the Lie
derivative of `Lf h` along the *control* directions — which requires evaluating
`g(x)` (the thrust direction) independently of `f(x)`:

```python
# In SphericalObstacle.compute_cbf_constraint():
thrust_dir = dynamics.thrust_direction(angles)      # ← from g(x)
A_cbf[:, i, 4*i] = (2/m) * (dp * thrust_dir).sum()  # Lg Lf h

Lf2_h = 2*||v||² - 2g*dp_z                          # ← from f(x) only
b_cbf = -Lf2_h - alpha1*Lf_h - alpha2*psi1          # pure drift terms
```

The reference avoids this by deriving `K_cbf` and `d_cbf` manually inline in
`construct_cbf_constraints()` rather than going through a generic `g(x)` matrix.
Our control-affine split is more general and reusable across different systems.

**Three Implementations:**

| Class | System | State | Control | Relative Degree |
|-------|--------|-------|---------|-----------------|
| `SingleIntegrator` | `ẋ = u` | `[x, y]` | `[ux, uy]` | 1 |
| `DoubleIntegrator` | `ẍ = a` | `[x, y, vx, vy]` | `[ax, ay]` | 2 |
| `Quadrotor` | 6-DOF with Euler angles | `[pos, angles, vel, ang_vel]` × N | `[thrust, torques]` × N | 2 |

**Shared Pattern:**
```python
class MyDynamics(ControlAffineDynamics):
    def __init__(self, ...):
        super().__init__(state_dim, control_dim, relative_degree)

    def f(self, x):  # Drift
        return ...

    def g(self, x):  # Control matrix
        return ...

    def step(self, x, u, dt):  # Integration (RK4)
        return ...
```

---

### Component 2: Barriers (`barriers/`)

**What it does:** Defines safety constraints for obstacles.

**Base Class:** `BarrierFunction`

**Key Method:**
```python
def compute_cbf_constraint(self, x, dynamics, alpha):
    """
    Compute A_cbf, b_cbf such that:
        A_cbf · u ≥ b_cbf  ⇒  CBF constraint satisfied

    Returns:
        A_cbf: (batch, control_dim) - gradient
        b_cbf: (batch,) - RHS value
    """
```

**Four Implementations:**

| Class | Dimension | Relative Degree | Agents | Used In |
|-------|-----------|-----------------|--------|---------|
| `CircularObstacle1` | 2D (x, y) | 1 | 1 | Single Integrator |
| `CircularObstacle2` | 2D (x, y) | 2 (HOCBF) | 1 | Double Integrator |
| `MultiAgentCircularObstacle2` | 2D (x, y) | 2 (HOCBF) | N | Multi-agent Double Integrator |
| `SphericalObstacle` | 3D (x, y, z) | 2 (HOCBF) | N | Quadrotor |

`CircularObstacle` is a **factory** — it returns the right class automatically based on
`dynamics.relative_degree` and `n_agent`.

**Example - CircularObstacle for Single Integrator (RD=1):**
```python
def compute_cbf_constraint(self, x, dynamics, alpha):
    # h(x) = ||p - c||² - r²
    p = x[:, 0:2]  # Extract position
    dp = p - self.center  # (batch, 2)
    h = (dp**2).sum(dim=-1) - self.radius**2  # (batch,)

    # Lf h = ∇h · f(x) = 2(p-c) · velocity
    # Lg h = ∇h · g(x) = 2(p-c) · I = 2(p-c)

    A_cbf = 2 * dp  # (batch, 2)
    b_cbf = -2 * (dp * velocity).sum(dim=-1) - alpha * h  # (batch,)

    return A_cbf, b_cbf
```

**Example - MultiAgentCircularObstacle2 for N double integrators (RD=2, HOCBF):**
```python
# n_agent=2, spatial_dim=2
# State layout: [px1,py1,px2,py2, vx1,vy1,vx2,vy2]   (DoubleIntegrator(dim=4))
# Control layout: [ax1,ay1, ax2,ay2]

def compute_cbf_constraint(self, x, dynamics, alpha):
    alpha1, alpha2 = alpha

    # x: (batch, 8)  →  pos: (batch, 2, 2),  vel: (batch, 2, 2)
    pos = x[:, :N*sdim].reshape(batch, N, sdim)
    vel = x[:, N*sdim:].reshape(batch, N, sdim)

    dp   = pos - center                            # (batch, N, 2)
    h0   = (dp**2).sum(-1) - r² - ε               # (batch, N)
    Lf_h = 2*(dp*vel).sum(-1)                     # (batch, N)
    psi1 = Lf_h + alpha1*h0                       # (batch, N)
    Lf2_h = 2*(vel**2).sum(-1)                    # (batch, N)  no drift

    # Block-diagonal A: agent i's constraint only involves agent i's control
    A_cbf = zeros(batch, N, N*sdim)
    for i in range(N):
        A_cbf[:, i, i*sdim:(i+1)*sdim] = 2*dp[:, i, :]   # (batch, 2)

    b_cbf = -(Lf2_h + alpha1*Lf_h + alpha2*psi1)  # (batch, N)
    return A_cbf, b_cbf
```

`A_cbf` for 2 agents looks like:
```
agent 0: [2(p0-c)_x, 2(p0-c)_y,  0,           0          ]
agent 1: [0,          0,           2(p1-c)_x,  2(p1-c)_y  ]
```
Each agent only controls its own acceleration, so the matrix is block-diagonal.
`CBFQPController` detects the `(batch, N, control_dim)` shape and unpacks into
`N` separate QP constraints automatically.

**Usage:**
```python
dynamics = DoubleIntegrator(dim=n_agent * 2)   # e.g. dim=4 for 2 agents in 2D
obstacles = [
    CircularObstacle(center=[1.0, 1.5], radius=0.4, epsilon=0.1,
                     dynamics=dynamics, n_agent=n_agent),
]
# Factory returns MultiAgentCircularObstacle2 automatically
```

---

**Example - SphericalObstacle for Quadrotor (RD=2, HOCBF):**
```python
def compute_cbf_constraint(self, x, dynamics, alpha):
    alpha1, alpha2 = alpha

    # Extract positions/velocities for all agents
    positions = x.reshape(batch, n_agent, 12)[:, :, 0:3]  # (batch, n_agent, 3)
    velocities = x.reshape(batch, n_agent, 12)[:, :, 6:9]

    # h(x) = ||p - c||² - r²
    dp = positions - center  # (batch, n_agent, 3)
    h = (dp**2).sum(dim=-1) - r**2  # (batch, n_agent)

    # Lf h = 2(p-c)·v
    Lf_h = 2 * (dp * velocities).sum(dim=-1)  # (batch, n_agent)

    # ψ₁ = Lf h + α₁·h
    psi1 = Lf_h + alpha1 * h

    # Lf² h = 2||v||² + 2(p-c)·a_drift
    Lf2_h = 2 * (velocities**2).sum(dim=-1) - 2 * gravity * dp[:, :, 2]

    # Lg Lf h: how control affects Lf h
    # For thrust control: 2(p-c) · (thrust/m * thrust_direction)
    thrust_dir = dynamics.thrust_direction(angles)  # (batch, n_agent, 3)

    A_cbf = torch.zeros(batch, n_agent, 4*n_agent)
    for i in range(n_agent):
        # Only thrust (index 4*i) affects position
        A_cbf[:, i, 4*i] = (2/m) * (dp[:, i] * thrust_dir[:, i]).sum(dim=-1)

    b_cbf = -Lf2_h - alpha2 * psi1  # (batch, n_agent)

    return A_cbf, b_cbf
```

---

### Component 3: Controllers (`controllers/`)

**What it does:** Combines everything into the CBF-QP layer.

#### 3a. `CBFQPController`

**The Heart of the System!**

```python
class CBFQPController:
    def __init__(self, dynamics, obstacles, alpha):
        self.dynamics = dynamics
        self.obstacles = obstacles
        self.alpha = alpha

        # CREATE THE DIFFERENTIABLE QP LAYER!
        self.qp_layer = create_cbf_qp_layer(
            control_dim=dynamics.control_dim,
            num_obstacles=len(obstacles)
        )

    def filter_control(self, x, u_desired):
        """
        Filter desired control through CBF safety constraints.

        Args:
            x: State (batch, state_dim)
            u_desired: Desired control from policy (batch, control_dim)

        Returns:
            u_safe: Safe control (batch, control_dim)
        """
        # 1. Compute CBF constraint parameters from each obstacle
        A_cbf_list = []
        b_cbf_list = []
        for obstacle in self.obstacles:
            A, b = obstacle.compute_cbf_constraint(x, self.dynamics, self.alpha)
            A_cbf_list.append(A)
            b_cbf_list.append(b)

        # 2. Pack parameters for QP
        qp_params = [u_desired] + [A for A in A_cbf_list] + [b for b in b_cbf_list]

        # 3. SOLVE QP (differentiably!)
        u_safe, = self.qp_layer(*qp_params)

        return u_safe
```

#### 3b. `create_cbf_qp_layer()`

**This creates the CVXPy problem:**

```python
def create_cbf_qp_layer(control_dim, num_obstacles):
    # Decision variable
    u = cp.Variable(control_dim)

    # Parameters (will be filled at runtime)
    u_desired = cp.Parameter(control_dim)
    A_cbf_list = [cp.Parameter(control_dim) for _ in range(num_obstacles)]
    b_cbf_list = [cp.Parameter() for _ in range(num_obstacles)]

    # Objective: minimize deviation from desired control
    objective = cp.Minimize(cp.sum_squares(u - u_desired))

    # Constraints: one per obstacle
    constraints = [A_cbf_list[i] @ u >= b_cbf_list[i]
                   for i in range(num_obstacles)]

    # Create problem
    problem = cp.Problem(objective, constraints)

    # WRAP IN CVXPYLAYERS!
    parameters = [u_desired] + A_cbf_list + b_cbf_list
    layer = CvxpyLayer(problem, parameters=parameters, variables=[u])

    return layer
```

**What `CvxpyLayer` does:**
1. Takes parameters as PyTorch tensors
2. Solves convex problem using OSQP/ECOS/SCS
3. Returns solution as PyTorch tensor **with gradients**!

#### 3c. `PolicyNetwork`

**Simple feedforward network:**

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim, ...):
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            *[ResBlock(hidden_dim) for _ in range(num_layers)],
            nn.Linear(hidden_dim, control_dim)
        )

    def forward(self, x):
        return self.layers(x)
```

---

### Component 4: Training (`training/`)

#### 4a. `TrainingConfig`

**Unified config for all examples:**

```python
@dataclass
class TrainingConfig:
    # Problem setup
    dynamics_type: str = 'single_integrator'  # or 'double_integrator'
    initial_state: List[float] = [0.0, 0.0]
    target_state: List[float] = [3.0, 3.0]

    # Obstacles
    obstacles: List[dict] = [
        {'center': [1.0, 1.0], 'radius': 0.6, 'epsilon': 0.1}
    ]

    # Time
    T: float = 6.0
    dt: float = 0.05

    # CBF
    cbf_alpha: float = 10.0  # or tuple (alpha1, alpha2) for HOCBF

    # Costs
    control_penalty: float = 1.0
    terminal_cost_weight: float = 500.0

    # Training
    num_epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
```

#### 4b. `CBFTrainer`

**Main training class:**

```python
class CBFTrainer:
    def __init__(self, config):
        # 1. Create dynamics
        if config.dynamics_type == 'single_integrator':
            self.dynamics = SingleIntegrator(dim=2)
        elif config.dynamics_type == 'double_integrator':
            self.dynamics = DoubleIntegrator(dim=2)

        # 2. Create obstacles
        self.obstacles = [
            CircularObstacle(
                center=obs['center'],
                radius=obs['radius'],
                epsilon=obs['epsilon'],
                dynamics=self.dynamics
            )
            for obs in config.obstacles
        ]

        # 3. Create CBF-QP controller (THE KEY COMPONENT!)
        self.cbf_controller = CBFQPController(
            dynamics=self.dynamics,
            obstacles=self.obstacles,
            alpha=config.cbf_alpha
        )

        # 4. Create policy network
        self.policy = PolicyNetwork(...)

        # 5. Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

    def train(self, verbose=True):
        for epoch in range(self.config.num_epochs):
            # Sample initial states
            x0_batch = self.sample_initial_states(batch_size)

            # Forward rollout (see next section)
            total_cost = self.rollout_trajectory(x0_batch)

            # Backward
            self.optimizer.zero_grad()
            total_cost.backward()
            self.optimizer.step()
```

---

## How the Three Examples Work

### Common Structure

All three examples follow this pattern:

```python
# 1. Setup
config = TrainingConfig(
    dynamics_type='...',
    obstacles=[...],
    ...
)

# 2. Create trainer
trainer = CBFTrainer(config)

# 3. Train
policy = trainer.train(verbose=True)

# 4. Visualize
plot_trajectories(trainer, num_trajectories=10)
```

### Example 1: Single Integrator

**File:** `examples/train_single_integrator.py`

**Dynamics:** `ẋ = u` (velocity is control)

**Obstacles:** 2 circular obstacles in 2D

**CBF:** Relative degree 1 (standard CBF)

**Config:**
```python
TrainingConfig(
    dynamics_type='single_integrator',
    position_dim=2,
    initial_state=[0.0, 0.0],  # Start at origin
    target_state=[4.0, 4.0],   # Go to (4, 4)
    obstacles=[
        {'center': [1.0, 1.0], 'radius': 0.55, 'epsilon': 0.1},
        {'center': [2.5, 2.5], 'radius': 0.55, 'epsilon': 0.1},
    ],
    T=6.0,
    dt=0.05,
    cbf_alpha=10.0,  # Standard CBF parameter
)
```

**Key Math:**
- State: `x = [px, py]` (2D)
- Control: `u = [ux, uy]` (2D)
- Barrier: `h(x) = ||p - c||² - r²`
- CBF constraint: `2(p-c) · u ≥ -α·h`

### Example 2: Double Integrator

**File:** `examples/train_double_integrator.py`

**Dynamics:** `ẍ = a` (acceleration is control)

**Obstacles:** 3 circular obstacles in 2D

**CBF:** Relative degree 2 (HOCBF)

**Config:**
```python
TrainingConfig(
    dynamics_type='double_integrator',
    position_dim=2,
    initial_state=[0.0, 0.0, 0.0, 0.0],  # [px, py, vx, vy]
    target_state=[3.0, 3.0, 0.0, 0.0],   # Stop at (3, 3)
    obstacles=[
        {'center': [0.4, 1.0], 'radius': 0.4, 'epsilon': 0.1},
        {'center': [2.2, 2.2], 'radius': 0.4, 'epsilon': 0.1},
        {'center': [2.4, 0.6], 'radius': 0.4, 'epsilon': 0.1},
    ],
    T=10.0,
    dt=0.2,
    cbf_alpha=(5.0, 5.0),  # (α₁, α₂) for HOCBF
)
```

**Key Math:**
- State: `x = [px, py, vx, vy]` (4D)
- Control: `u = [ax, ay]` (2D)
- Barrier: `h(x) = ||p - c||² - r²`
- HOCBF:
  - `ψ₁ = Lf h + α₁·h = 2(p-c)·v + α₁·h`
  - Constraint: `2(p-c) · a ≥ -2||v||² - α₂·ψ₁`

### Example 3: Quadrotor (Multi-Agent)

**File:** `examples/train_quadrotor_multi_cvxpy.py`

**Dynamics:** 6-DOF quadrotor with Euler angles

**Obstacles:** 3 spherical obstacles in 3D

**CBF:** Relative degree 2 (HOCBF, focus on position)

**Config:**
```python
dynamics = Quadrotor(n_agent=5, mass=0.5, gravity=1.0)
obstacles = [SphericalObstacle(...), ...]   # 3 spherical obstacles

config = TrainingConfig(
    dynamics=dynamics,           # pass object, not type string
    obstacles_list=obstacles,
    output_transform=make_output_transform(...),   # tanh + hover bias
    cost_fn=make_cost_fn(...),                     # multi-component cost
    sample_fn=make_sample_fn(...),                 # per-agent xy noise
    plot_fn=make_plot_fn(...),
    cbf_alpha=(5.0, 5.0),        # HOCBF (α₁, α₂)
    ...
)
trainer = CBFTrainer(config)
trainer.train()
```

**Key Math:**
- State per agent: `[x, y, z, ψ, θ, φ, vx, vy, vz, vψ, vθ, vφ]` (12D)
- Control per agent: `[thrust, τψ, τθ, τφ]` (4D)
- Total: 5 agents → 60D state, 20D control
- Barrier per agent per obstacle: `h = ‖p_i − c‖² − r² − ε²`
- HOCBF gives 15 total constraints (3 obstacles × 5 agents)

**Special Considerations:**
- Multi-agent: each agent has its own constraint per obstacle
- 3D spherical obstacles
- Non-linear dynamics (thrust direction depends on Euler angles)
- Only thrust affects position; torques don't appear in CBF (`Lg h = 0` for torques)

---

## The Training Loop

### What Happens During One Epoch

```python
for epoch in range(num_epochs):
    # 1. SAMPLE INITIAL STATES
    x0_batch = sample_initial_states(batch_size=32)  # (32, state_dim)

    # 2. FORWARD ROLLOUT
    running_cost = 0.0
    x = x0_batch  # Current state

    for t in range(num_steps):
        # a) Policy outputs desired control
        u_desired = policy(x)  # (32, control_dim)

        # b) CBF-QP filters to safe control
        u_safe = cbf_controller.filter_control(x, u_desired)  # (32, control_dim)
        #      ^^^^^^^^^ CVXPyLayers magic happens here!

        # c) Simulate dynamics
        x = dynamics.step(x, u_safe, dt)  # (32, state_dim)

        # d) Accumulate running cost
        running_cost += dt * 0.5 * ||u_safe||²

    # 3. COMPUTE TERMINAL COST
    terminal_cost = ||x - x_target||²

    # 4. TOTAL COST
    total_cost = α_running * running_cost + α_terminal * terminal_cost

    # 5. BACKWARD (gradients flow through everything!)
    optimizer.zero_grad()
    total_cost.backward()
    #           ^^^^^^^^^ Gradients flow:
    #           - Through dynamics integration
    #           - Through CBF-QP (via CVXPyLayers!)
    #           - Back to policy network

    # 6. UPDATE POLICY
    optimizer.step()
```

### Gradient Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     FORWARD PASS                            │
└─────────────────────────────────────────────────────────────┘
    x₀ → Policy → u_des → CBF-QP → u_safe → Dynamics → x₁ → ...
                   NN      CVXPy    Layer     RK4

┌─────────────────────────────────────────────────────────────┐
│                     BACKWARD PASS                           │
└─────────────────────────────────────────────────────────────┘
    ∂L/∂θ ← ∂L/∂u_des ← ∂u_safe/∂u_des ← ∂L/∂u_safe ← ∂L/∂x₁ ← ...
    policy    policy      CVXPyLayers!    dynamics     dynamics
    params    output      implicit diff   gradients    gradients
```

**Key:** `∂u_safe/∂u_des` comes from **implicit differentiation** of the QP solution!

---

## Mathematical Background

### Higher-Order Control Barrier Functions (HOCBF)

#### Why Standard CBF Is Not Enough

For a system `ẋ = f(x) + g(x)u`, the standard CBF condition is:

```
ḣ + α(h) ≥ 0   ⟺   Lf h + Lg h · u ≥ -α(h)
```

This requires that control `u` appears in `ḣ` — i.e., `Lg h ≠ 0`. This holds when
the barrier depends on the state components that control directly drives.

**Problem:** For position barriers like `h(p) = ‖p - c‖² - r²`:
- Single integrator `ẋ = u`: control is velocity, so `Lg h = 2(p-c) ≠ 0` — OK!
- Double integrator `ẍ = a`: control is acceleration, velocity is intermediate.
  `ḣ = 2(p-c)·v` and `Lg h = 0` because `u` does not appear in `ḣ` at all.
  Standard CBF fails: you can't constrain `u` from this equation.

The **relative degree** of `h` w.r.t. the system is the number of times you must
differentiate `h` before control `u` appears. Position barriers with double integrators
and quadrotors have relative degree **2**.

---

#### Step-by-Step HOCBF Derivation (Relative Degree 2)

**Setup:** Control-affine system `ẋ = f(x) + g(x)u`, barrier `h₀(x)` with `Lg h₀ = 0`.

---

**Step 1 — Original barrier and safe set**

```
h₀(x) ≥ 0   defines the safe set S₀
```

**Step 2 — First time derivative (control not present)**

```
ḣ₀ = ∇h₀ · ẋ = ∇h₀ · f(x) + ∇h₀ · g(x) · u
             = Lf h₀ + 0          (since Lg h₀ = 0)
             = Lf h₀
```

Since `u` doesn't appear, we cannot impose the CBF condition yet.

**Step 3 — Introduce auxiliary barrier ψ₁**

We want h₀ to stay non-negative. A sufficient condition is that the set

```
S₁ = { x : ψ₁(x) ≥ 0 }    where    ψ₁ = Lf h₀ + α₁ · h₀
```

is forward invariant. This is the HOCBF "cascade": if `ψ₁ ≥ 0` and `h₀ ≥ 0`
initially, then `h₀` stays non-negative.

**Intuition:** `ψ₁ ≥ 0` means "even if `h₀` is small, it is decreasing slowly enough
(proportional to how small it is) that the safe set boundary won't be crossed."

**Step 4 — Second time derivative (control appears here)**

Differentiate `ψ₁`:

```
ψ̇₁ = d/dt [Lf h₀ + α₁ h₀]
     = d/dt [Lf h₀] + α₁ ḣ₀
     = Lf(Lf h₀) + Lg(Lf h₀) · u + α₁ Lf h₀
     = Lf² h₀   +  Lg Lf h₀ · u  +  α₁ Lf h₀
```

Now `u` appears! `Lg Lf h₀ ≠ 0` is the condition for relative degree exactly 2.

**Step 5 — Apply CBF condition to ψ₁**

Require `ψ̇₁ + α₂ ψ₁ ≥ 0` (standard CBF on the auxiliary barrier):

```
Lf² h₀  +  Lg Lf h₀ · u  +  α₁ Lf h₀  +  α₂ ψ₁  ≥  0
```

**Step 6 — Rearrange into QP form**

Move everything without `u` to the right:

```
( Lg Lf h₀ ) · u  ≥  -(Lf² h₀  +  α₁ Lf h₀  +  α₂ ψ₁)
```

This is `A_cbf · u ≥ b_cbf` with:

```
A_cbf  =  Lg Lf h₀
b_cbf  =  -(Lf² h₀  +  α₁ Lf h₀  +  α₂ ψ₁)
       =  -(Lf² h₀  +  (α₁+α₂) Lf h₀  +  α₁α₂ h₀)
```

The second form (expanding `ψ₁ = Lf h₀ + α₁ h₀`) makes the polynomial structure
visible: the characteristic polynomial `(s + α₁)(s + α₂)` has roots `-α₁, -α₂`,
giving exponential convergence toward the safe set boundary.

---

#### Concrete Example: Double Integrator + Circular Obstacle

**System:** `ẋ = [v, a]` with `f(x) = [v, 0]`, `g(x) = [0, I]`

**Barrier:** `h₀(p) = ‖p - c‖² - r² - ε`

| Quantity | Formula | Code (`circular_obstacle.py`) |
|----------|---------|-------------------------------|
| `h₀` | `‖p - c‖² - r² - ε` | `dist_sq - radius² - epsilon` |
| `∇h₀` | `2(p - c)` | `grad_h0 = 2*(p - center)` |
| `Lf h₀` | `2(p-c) · v` | `h0_dot = sum(grad_h0 * v)` |
| `ψ₁` | `Lf h₀ + α₁ h₀` | `h1 = h0_dot + alpha1 * h0` |
| `Lf² h₀` | `2‖v‖²` | `Lf2_h0 = 2*sum(v**2)` |
| `Lg Lf h₀` | `2(p - c)` | `Lg_Lf_h0 = grad_h0` |
| `A_cbf` | `2(p - c)` | `A_cbf = Lg_Lf_h0` |
| `b_cbf` | `-(2‖v‖² + α₁ Lf h₀ + α₂ ψ₁)` | `b_cbf = -Lf2_h0 - alpha1*h0_dot - alpha2*h1` |

**Why `Lf² h₀ = 2‖v‖²`?**

```
Lf² h₀ = d/dt [2(p-c)·v]  along f(x) = [v, 0]
        = 2ṗ · v + 2(p-c) · v̇_drift
        = 2v · v  +  2(p-c) · 0          (double integrator has no drift acceleration)
        = 2‖v‖²
```

**Why `Lg Lf h₀ = 2(p-c)`?**

```
Lf h₀ = 2(p-c) · v        (depends on v)
∂(Lf h₀)/∂v = 2(p-c)      (gradient w.r.t. velocity component)
g(x) = [0; I]              (control enters as acceleration → affects velocity)
Lg Lf h₀ = ∂(Lf h₀)/∂v · I = 2(p-c)
```

**Geometric meaning:** `2(p-c)` is the outward normal to the obstacle boundary.
The constraint says: "your acceleration must have a sufficient outward component
(relative to velocity and how close you are)."

---

#### Concrete Example: Quadrotor + Spherical Obstacle

**System:** 6-DOF quadrotor, state per agent `[p, η, v, ω]` (12D), Euler angles `η = [ψ,θ,φ]`

**Control:** `u = [T, τ_ψ, τ_θ, τ_φ]` — thrust + three torques

**Drift acceleration** (what `f(x)` contributes to `v̇`): gravity `−g e₃`

**Barrier:** `h₀(p) = ‖p - c‖² - r² - ε²`

| Quantity | Formula | Code (`spherical_obstacle.py`) |
|----------|---------|-------------------------------|
| `h₀` | `‖p-c‖² - r² - ε²` | `(dp**2).sum() - r² - ε²` |
| `Lf h₀` | `2(p-c) · v` | `2*(dp*velocities).sum(-1)` |
| `ψ₁` | `Lf h₀ + α₁ h₀` | `psi1 = Lf_h + alpha1*h_vals` |
| `Lf² h₀` | `2‖v‖² - 2g (p_z - c_z)` | `2*vel_sq - 2*gravity*dp[:,:,2]` |
| `Lg Lf h₀` (thrust) | `(2/m)(p-c) · thrust_dir` | `(2/m)*(dp[:,i]*thrust_dir[:,i]).sum(-1)` |
| `Lg Lf h₀` (torques) | `0` | not set (zeros by default) |
| `A_cbf` | sparse, entry at `4*i` only | `A_cbf[:, i, 4*i] = ...` |
| `b_cbf` | `-(Lf²h + α₁ Lf h + α₂ ψ₁)` | `-Lf2_h - alpha1*Lf_h - alpha2*psi1` |

**Why `Lf² h₀ = 2‖v‖² - 2g(p_z - c_z)`?**

```
Lf² h₀ = d/dt [2(p-c)·v]  along drift f(x):
        = 2ṗ · v + 2(p-c) · v̇_drift
        = 2v · v  +  2(p-c) · (-g e₃)     (gravity is the drift acceleration)
        = 2‖v‖²  -  2g (p_z - c_z)
```

**Why only thrust enters `Lg Lf h₀`?**

```
Lf h₀ = 2(p-c) · v        (depends on v)
Control u = [T, τ_ψ, τ_θ, τ_φ]:
  - Thrust T → v̇ += (T/m) * d(η)    (d = body z-axis, rotation-dependent)
  - Torques τ → ω̇ only              (angular acceleration, not v)

Lg Lf h₀ · u = 2(p-c) · ∂v̇/∂u · u = 2(p-c) · (T/m) d(η)
→ scalar coeff of T: (2/m)(p-c) · d(η)
→ coeff of torques:  0
```

The thrust direction `d(η)` is computed by `dynamics.thrust_direction(angles)`,
which returns the body z-axis rotated into the world frame via Euler angles.

**Sparsity in multi-agent setting:**

For `n` agents and one obstacle, `A_cbf` is shape `(batch, n_agent, 4*n_agent)`.
Agent `i`'s constraint row has only one non-zero entry — at column `4*i` (its thrust):

```
A_cbf[:, i, :] = [0, 0, ..., (2/m)(p_i-c)·d_i, 0, ..., 0]
                               ^^^^^^^^^^^^^^^^^
                               index 4*i (agent i's thrust)
```

Torques don't appear because angular dynamics don't affect linear position
through `Lf h₀`. This sparsity is why the QP stays tractable at 30 constraints
(3 obstacles × 10 agents).

---

#### Safety Guarantee Summary

Two conditions together guarantee `h₀(x) ≥ 0` for all time:

1. **Initialization:** `h₀(x₀) ≥ 0` and `ψ₁(x₀) ≥ 0`
2. **HOCBF constraint satisfied at every timestep:** `A_cbf · u ≥ b_cbf`

If these hold, then by the cascade argument:
- `ψ₁ ≥ 0` is forward invariant (from the QP constraint via Nagumo's theorem)
- `ψ₁ ≥ 0` + `h₀(x₀) ≥ 0` ⟹ `h₀(x(t)) ≥ 0` for all `t ≥ 0`

The QP in `CBFQPController` enforces condition 2 at every timestep,
so safety is guaranteed as long as the system starts in `S₀ ∩ S₁`.

---

#### Class-K Functions and Why Linear α Is Fine

The theory requires `α₁, α₂` to be class-K functions (continuous, zero at 0, strictly
increasing). We use linear `α₁(h) = α₁ · h` and `α₂(ψ₁) = α₂ · ψ₁`, which satisfies
this. Larger `α₁, α₂` → less conservative (allows faster approaches to boundary)
but may require larger control effort. In practice `α₁ = α₂ = 5.0` works well.

---

### Implicit Differentiation of QP

**The QP:**
```
u* = argmin ||u - u_des||²
     s.t.    A·u ≥ b
```

**KKT Conditions (necessary and sufficient for convex QP):**
```
u* - u_des + Aᵀλ = 0          (stationarity)
A·u* ≥ b                      (primal feasibility)
λ ≥ 0                         (dual feasibility)
λ ⊙ (A·u* - b) = 0            (complementarity)
```

**Implicit Function Theorem:**
Differentiate KKT conditions w.r.t. `u_des`:
```
∂u*/∂u_des = (I + AᵀM⁻¹A)⁻¹
```
where `M` is the matrix of active constraints.

**CVXPyLayers computes this automatically!**

### Why This Works for Training

1. **Policy learns what's feasible:**
   - If `u_des` violates constraints → QP projects to boundary
   - Policy receives gradient pointing toward feasible region
   - Over time, policy learns to stay away from constraints

2. **End-to-end optimization:**
   - Minimize total cost (control + terminal)
   - Policy learns trade-off between effort and goal-reaching
   - While respecting safety!

3. **No reward shaping needed:**
   - Don't need to manually penalize constraint violations
   - Hard constraints → QP ensures safety
   - Just optimize the cost!

---

## Data Flow

### Shared Across All Examples

**1. Training Config → Components:**
```
TrainingConfig
    ├→ Dynamics (SingleIntegrator / DoubleIntegrator)
    ├→ Obstacles (CircularObstacle × N)
    ├→ CBFQPController
    │   ├→ create_cbf_qp_layer() → CvxpyLayer
    │   └→ filter_control()
    ├→ PolicyNetwork
    └→ Optimizer
```

**2. Training Loop Data Flow:**
```
Epoch Loop:
  ├─ Sample x₀ (batch)
  │
  ├─ Trajectory Rollout:
  │   └─ for t in timesteps:
  │       ├─ x → Policy → u_des
  │       ├─ x + u_des → CBF-QP → u_safe
  │       └─ x + u_safe → Dynamics → x_next
  │
  ├─ Compute Cost:
  │   ├─ Running: Σ ||u_safe||²
  │   └─ Terminal: ||x_final - x_target||²
  │
  └─ Backward & Update:
      ├─ total_cost.backward()
      └─ optimizer.step()
```

### Quadrotor-Specific Additions

**1. Multi-Agent State:**
```
State: [agent₁ state, agent₂ state, ..., agent₁₀ state]
       ↓             ↓                    ↓
       12D          12D                  12D
       = [pos, angles, vel, ang_vel]
```

**2. Multi-Agent Control:**
```
Control: [agent₁ ctrl, agent₂ ctrl, ..., agent₁₀ ctrl]
         ↓            ↓                   ↓
         4D           4D                  4D
         = [thrust, τψ, τθ, τφ]
```

**3. Constraint Multiplication:**
```
3 obstacles × 10 agents = 30 CBF constraints per timestep!

For each (agent_i, obstacle_j):
  - Compute h_ij(x) = ||p_i - c_j||² - r²
  - Compute HOCBF constraint: A_ij · u ≥ b_ij
```

**4. Sparse Control Matrix:**
```
Only agent i's thrust affects agent i's position:

A_ij for (agent_i, obstacle_j):
  [0, 0, ..., (gradient for agent_i thrust), 0, ..., 0]
                    ↑
                 index 4*i
```

---

## Summary

### What's Shared Across Examples

| Component | Single Int | Double Int | Quadrotor |
|-----------|------------|------------|-----------|
| `ControlAffineDynamics` base | ✓ | ✓ | ✓ |
| `BarrierFunction` base | ✓ | ✓ | ✓ |
| `CBFQPController` | ✓ | ✓ | ✓ |
| `create_cbf_qp_layer()` | ✓ | ✓ | ✓ |
| `PolicyNetwork` | ✓ | ✓ | ✓ |
| `CBFTrainer` | ✓ | ✓ | ✓ (hooks) |
| `TrainingConfig` | ✓ | ✓ | ✓ (hooks) |

All three examples now use `CBFTrainer` / `TrainingConfig` with the same pattern:
build dynamics + obstacles → pass hooks for system-specific logic → `trainer.train()`.
The quadrotor uses `output_transform`, `cost_fn`, `sample_fn`, and `plot_fn` hooks.

### Key Differences

| Feature | Single Int | Double Int | Multi-Agent Double Int | Quadrotor |
|---------|------------|------------|------------------------|-----------|
| **State Dim** | 2 | 4 | 4×N | 120 (12×10) |
| **Control Dim** | 2 | 2 | 2×N | 40 (4×10) |
| **Relative Degree** | 1 | 2 | 2 | 2 |
| **CBF Type** | Standard | HOCBF | HOCBF | HOCBF |
| **Obstacle Class** | `CircularObstacle1` | `CircularObstacle2` | `MultiAgentCircularObstacle2` | `SphericalObstacle` |
| **Obstacle Dim** | 2D | 2D | 2D | 3D |
| **Num Constraints** | n_obs | n_obs | n_obs × N | n_obs × N |
| **Multi-Agent** | No | No | Yes | Yes |
| **Dynamics** | Linear | Linear | Linear | Non-linear |
| **Dynamics Class** | `SingleIntegrator(dim=2)` | `DoubleIntegrator(dim=2)` | `DoubleIntegrator(dim=N*2)` | `Quadrotor(n_agent=N)` |

### The Pipeline

```
┌──────────────┐
│ User Config  │
└──────┬───────┘
       ↓
┌──────────────────────────────────────────────────┐
│              Framework Setup                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Dynamics │  │ Barriers │  │ CBF-QP Layer  │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
└──────────────────────┬───────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────┐
│              Training Loop                       │
│  ┌─────────────────────────────────────────┐    │
│  │  x → Policy → CBF-QP → Dynamics → Cost  │    │
│  │   ↑________________↓___________________↓ │    │
│  │        Backprop through CVXPyLayers      │    │
│  └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────┐
│               Trained Policy                     │
│  - Outputs safe controls                         │
│  - Avoids obstacles                              │
│  - Reaches target                                │
└──────────────────────────────────────────────────┘
```

---

## Further Reading

- **CVXPyLayers Paper:** [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)
- **CBF Tutorial:** [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
- **HOCBF:** [Safety Barrier Certificates for Collisions-Free Multirobot Systems](https://arxiv.org/abs/1812.05950)
- **Implicit Differentiation:** [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443)

---

**Questions?** See the code! Every component is documented with:
- Docstrings explaining purpose
- Type hints for inputs/outputs
- Comments on mathematical formulations
