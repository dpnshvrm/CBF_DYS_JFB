# CVXPyLayers Training Notes & Best Practices

## Critical Lessons Learned

This document tracks important parameters, issues, and solutions discovered during quadrotor CBF training with CVXPyLayers.

**Last Updated**: March 30, 2026

---

## Table of Contents

1. [Critical Solver Settings](#critical-solver-settings)
2. [Training Hyperparameters](#training-hyperparameters)
3. [What Works vs What Doesn't](#what-works-vs-what-doesnt)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Success Indicators](#success-indicators)

---

## Critical Solver Settings

### ⚠️ MOST IMPORTANT: Use SCS Solver!

**Location**: `CVXPyLayers/controllers/cbf_qp_layer.py`

```python
solver_args = {
    "solve_method": "SCS",  # ❌ NOT "CLARABEL" - it doesn't support differentiation!
    "eps": 1e-4,
    "max_iters": 2500,
    "acceleration_lookback": 10,
}
```

**Why SCS?**
- CVXPyLayers differentiation **ONLY works with SCS solver**
- CLARABEL/ECOS/others: No backpropagation → garbage gradients → training fails
- This was the #1 root cause of training failures!

**Evidence of Wrong Solver:**
- Gradients explode to 1e+18, 1e+25, 1e+68
- Terminal cost stuck at 3000-4000 (doesn't decrease)
- Loss oscillates wildly

---

## Training Hyperparameters

### Current Best Settings (1 Agent, Quadrotor)

**Location**: `train_quadrotor_multi_cvxpy.py` lines 117-135

| Parameter | Value | Why? | Old Value (Failed) |
|-----------|-------|------|-------------------|
| `alpha_terminal` (initial) | **5.0** | Start low, policy learns obstacles first | 100.0 (too aggressive) |
| `alpha_terminal_final` | **50.0** | Conservative max for CVXPyLayers | 100.0 (unstable) |
| `alpha_sched_step` | **5.0** | Gentle increase | 5.0 |
| `alpha_sched_every` | **100** | Slow schedule (every 100 epochs) | 20 (too fast) |
| `learning_rate` | **1e-4** | Balance: not stuck, not exploding | 1e-3 (exploded), 1e-5 (stuck) |
| `batch_size` | **16** | Smaller = more stable gradients | 64 (unstable) |
| `z0_std` | **2e-2** | Less noise = easier to learn initially | 4e-2 (too noisy) |
| `dt` | **0.4** | Larger timestep = faster training | 0.2 (slow) |
| `num_steps` | **25** | Fewer steps (from dt=0.4, T=10) | 50 (slow) |

### Alpha Terminal Schedule

```python
# Conservative schedule for CVXPyLayers stability
Epoch    0-100:  α_terminal = 5.0   (learn dynamics + obstacles)
Epoch  100-200:  α_terminal = 10.0  (start caring about target)
Epoch  200-300:  α_terminal = 15.0
Epoch  300-400:  α_terminal = 20.0
...
Epoch  900+:     α_terminal = 50.0  (cap)
```

**Why Start Low?**
- High α_terminal early → policy ignores obstacles → crashes → infeasible QPs
- Low α_terminal early → policy learns safe behavior first → gradual target focus

---

## What Works vs What Doesn't

### ✅ What WORKS

| Setting | Value | Result |
|---------|-------|--------|
| Solver | **SCS** | Differentiable, training converges |
| α_terminal schedule | **5 → 50** (slow) | Stable, gradual improvement |
| Learning rate | **1e-4** | No explosions, steady progress |
| Batch size | **16** | Stable gradients |
| Starting simple | **1 agent first** | Debugging easier |
| **A: Output init** | **uniform(-0.01, 0.01)** | **Small initial u_desired** |
| **B: Grad clipping** | **100.0** | **REQUIRED for 5+ agents** |
| **C: Normalize QP** | **‖A‖ = 1** | **Prevents unbounded QP** |

### ❌ What DOESN'T Work

| Setting | Value | Symptom |
|---------|-------|---------|
| Solver | CLARABEL | Gradients useless, G stuck at 3000+ |
| α_terminal start | 100.0 | Policy ignores obstacles, infeasible QPs |
| α_terminal schedule | Every 20 epochs | Too aggressive, training unstable |
| Learning rate | 1e-3 | Explosions (grad=1e+68), NaN/Inf |
| Learning rate | 1e-5 | Stuck (no learning after 500 epochs) |
| Batch size | 64 | Unstable gradients |
| Fixed α_terminal | 100.0 (no schedule) | Policy ignores safety early, poor convergence |
| **No gradient clipping** | **5+ agents** | **grad=1e+54, complete breakdown** |

---

## Common Issues & Solutions

### Issue 1: Gradients Exploding (1e+18, 1e+68)

**Symptoms:**
```
epoch  21 total=7.9691e+28 L=1.8250e+27 G=7.7866e+26 grad=9.52e+68
```

**Root Cause:** Wrong solver (CLARABEL) or learning rate too high

**Solution:**
1. ✅ Verify `solver_args["solve_method"] = "SCS"` in `cbf_qp_layer.py`
2. ✅ Lower learning rate to `1e-4` or `1e-5`

---

### Issue 2: Training Stuck (Terminal Cost Not Decreasing)

**Symptoms:**
```
epoch  506 total=4.7332e+04 L=6.1409e+02 G=1.5573e+03  (stuck for 400+ epochs)
epoch  900 total=9.7000e+04 L=1.5700e+01 G=3.2380e+03  (no improvement)
```

**Root Cause:** Learning rate too low OR wrong solver

**Solution:**
1. If `lr = 1e-5`: Increase to `1e-4`
2. Verify SCS solver is active
3. Check if α_terminal is increasing (should see schedule updates every 100 epochs)

---

### Issue 3: QP Infeasible / Unbounded

**Symptoms:**
```
CBF-QP failed at epoch 118, step 46: Solver scs returned status infeasible
```

**Root Cause:** Constraints too tight, policy outputs infeasible controls

**Solutions:**
1. ✅ Exception handling added to fallback to `u_desired` (line 248-252 in train script)
2. If happening frequently (>10%):
   - Lower α_terminal
   - Increase CBF alpha parameters (makes constraints looser)
   - Check initial conditions aren't too close to obstacles

---

### Issue 4: Barrier Values Going Negative (Collisions!)

**Symptoms:**
```
epoch  970 h_min=-0.03274  ⚠️ COLLISION!
```

**Root Cause:** Policy learned to cut through obstacles

**Solutions:**
1. Lower learning rate (policy changing too fast)
2. Start α_terminal lower (focus on safety first)
3. Increase obstacle epsilon (larger safety margin)
4. Check if QP is failing frequently (see Issue 3)

---

### Issue 5: Multi-Agent Gradient Explosions + QP Unbounded/Infeasible

**Symptoms:**
```
epoch 1 grad=1.46e+34  🔥
epoch 3 grad=1.64e+54  🔥🔥🔥 (more atoms than in the universe!)
CBF-QP unbounded/infeasible at step 24
h_min=1.47e+02  (agent far from obstacles → large b_cbf → degenerate QP)
```

**Root Causes:**
1. **Gradient explosions:** SCS solver gradient noise scales with problem size
   - 5 agents = 60-state, 20-control, 15 constraints
   - CVXPyLayers' SCS differentiation not designed for large multi-agent
2. **QP degeneracy:** When h is large, b_cbf can be O(1e6), making QP numerically unsolvable
   - Large h → large ψ₁ → large b → constraint requires impossibly large thrust
   - Random init policy outputs O(100) → velocities explode → Lf²h dominates

**Solution (Three-Part Fix):**

✅ **A: Small output layer initialization** - Prevents large initial u_desired

**Location**: `CVXPyLayers/controllers/policy_network.py` lines 64-67

```python
# In PolicyNetwork.__init__, after building layers:
nn.init.uniform_(output_layer.weight, -0.01, 0.01)
nn.init.zeros_(output_layer.bias)
# → Policy starts near-zero, so u_desired ≈ hover_bias (physically reasonable)
```

✅ **B: Gradient clipping** - Prevents policy divergence during training

**Location**: `train_quadrotor_multi_cvxpy.py` lines 136, 297

```python
# Add parameter (line 136)
grad_clip_norm = 100.0  # CRITICAL for multi-agent (5+) stability

# Apply after backward, before optimizer step (line 297)
torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
```

✅ **C: Normalize QP constraints** - Prevents numerical degeneracy

**Location**: `CVXPyLayers/controllers/cbf_qp_layer.py` lines 171-181

```python
# Normalize constraints: â·u ≥ b̂ where ‖â‖ = 1
for A, b in zip(A_cbf_list, b_cbf_list):
    A_norm = A.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    A_normalized = A / A_norm       # Unit-norm row
    b_normalized = b / A_norm.squeeze(-1)  # Scale b consistently
```

**Why This Works:**
- **A:** Initial outputs ~0.01 instead of ~10 → no runaway velocities
- **B:** Grad=1e+54 clipped to 100 → update = 0.01 instead of 1e+50 ✅
- **C:** b̂ is the required component of u in constraint direction (physically bounded)
  - No hard-coded clipping value needed
  - Scales automatically with problem size
  - More principled than fixed `b.clamp(-50, 50)`

**Why Not `tanh` Output Bounds?**
- Too restrictive: limits policy capacity permanently
- A+B+C achieve same initial stability without limiting learned range

---

## Success Indicators

### ✅ Training is Working If:

| Metric | Good Range | Great Range |
|--------|-----------|-------------|
| **Terminal Cost (G)** | Decreasing | < 100 by epoch 500 |
| **Running Cost (L)** | 8-15 | 10-14 (stable) |
| **Total Loss** | Decreasing | < 1000 by epoch 500 |
| **Gradient Norm** | 1e4 - 1e6 | 1e4 - 1e5 (or <1e8 if 5+ agents) |
| **Min Barrier (h)** | > 0.1 | > 5.0 |
| **QP Failures** | < 5% | < 1% |

### ❌ Training is Failing If:

| Symptom | Indicates |
|---------|-----------|
| G > 3000 after 200 epochs | Not learning to reach target |
| G stuck/oscillating | Wrong solver or lr too low |
| Grad > 1e8 | Wrong solver or lr too high |
| h < 0 (negative barriers) | Collisions! Policy unsafe |
| QP fails > 20% | Constraints too tight |
| Loss = NaN/Inf | Numerical explosion |

---

## Parameter Evolution Log

### Version 1 (FAILED)
- Solver: CLARABEL ❌
- α_terminal: 20 → 200 (every 20 epochs)
- lr: 1e-3
- Result: G stuck at 3200, gradients 1e6

### Version 2 (FAILED)
- Solver: SCS ✅
- α_terminal: 100 (fixed)
- lr: 1e-3
- Result: Gradients exploded to 1e+68

### Version 3 (FAILED)
- Solver: SCS ✅
- α_terminal: 20 → 200
- lr: 1e-5
- Result: G stuck at 1500, too conservative

### Version 4 (Works for 1-3 agents)
- Solver: SCS ✅
- α_terminal: 5 → 50 (every 100 epochs)
- lr: 1e-4
- batch_size: 16
- z0_std: 2e-2
- Result: Stable for 1-3 agents

### Version 5 (CURRENT - Multi-Agent)
- Solver: SCS ✅
- α_terminal: 5 → 50 (every 100 epochs)
- lr: 1e-4
- batch_size: 16
- z0_std: 2e-2
- **grad_clip_norm: 100.0** ✅ (NEW!)
- Result: Handles 5+ agents, prevents 1e+54 gradient explosions

---

## Terminal Cost Formulation

**CRITICAL**: Must match dynamics structure!

**Location**: `train_quadrotor_multi_cvxpy.py` lines 263-276

```python
# CORRECT formulation (matches train.py)
x_reshaped = x.reshape(batch_size, n_agent, 12)
pos = x_reshaped[:, :, 0:3]      # Position
angles = x_reshaped[:, :, 3:6]   # Euler angles
vel = x_reshaped[:, :, 6:9]      # Linear velocity
ang_vel = x_reshaped[:, :, 9:12] # Angular velocity

terminal_cost = (
    0.5 * ((pos - p_target.unsqueeze(0)) ** 2).sum(dim=-1).sum(dim=-1) +
    0.5 * vel.pow(2).sum(dim=-1).sum(dim=-1) +
    0.5 * angles.pow(2).sum(dim=-1).sum(dim=-1) +
    0.5 * ang_vel.pow(2).sum(dim=-1).sum(dim=-1)
).mean()
```

**Why this matters:**
- Penalizes: position error, final velocity, attitude error, angular rates
- Matches the physics of hovering at target
- Wrong formulation → policy learns wrong objective

---

## Quick Reference: Files to Check

When debugging training issues:

1. **Solver Settings**: `CVXPyLayers/controllers/cbf_qp_layer.py` line 133-140
2. **Training Params**: `CVXPyLayers/examples/train_quadrotor_multi_cvxpy.py` line 117-135
3. **Terminal Cost**: `CVXPyLayers/examples/train_quadrotor_multi_cvxpy.py` line 263-276
4. **Loss Function**: `CVXPyLayers/examples/train_quadrotor_multi_cvxpy.py` line 260-279
5. **Config Output**: `CVXPyLayers/models/train_quadrotor_multi_cvxpy/config_cvxpy.json`
6. **Training Log**: `CVXPyLayers/models/train_quadrotor_multi_cvxpy/training_history_cvxpy.csv`

---

## Recommended Next Steps

### For 1 Agent (Testing)
1. ✅ Verify SCS solver
2. ✅ Set α_terminal = 5 → 50
3. ✅ Set lr = 1e-4
4. Run 200 epochs
5. Check if G < 1000 by epoch 100

### For Multiple Agents (5+)
- **MANDATORY: Add gradient clipping** (`grad_clip_norm = 100.0`)
- **MANDATORY: Small output init** (`uniform(-0.01, 0.01)`)
- **MANDATORY: Normalize QP constraints** (‖A‖ = 1)
- Start with 2-3 agents, then scale to 5+
- More conservative lr (1e-4, possibly lower)
- Longer schedule (every 200 epochs)
- Expect gradients >1e8 for 5+ agents (clipping handles this)
- Training slower (~20-30s/epoch vs ~10s for 1 agent) due to CPU bottleneck

---

## Notes on Visualization

**Location**: `CVXPyLayers/dynamics/quadrotor.py` lines 187-283

- Plots saved every `plot_freq` epochs (default: 100)
- Saved to: `models/train_quadrotor_multi_cvxpy/plots/traj_epoch_XXXX.png`
- Shows: 3D view + bird's-eye view + obstacles + targets
- Use to debug: trajectories cutting through obstacles, not reaching target, etc.

---

## Contact / Questions

If training still fails after following this guide:
1. Check config.json matches these parameters
2. Verify SCS solver in cbf_qp_layer.py
3. Verify all three fixes (A: output init, B: grad clipping, C: QP normalization)
4. Look at first 50 epochs in CSV - is G decreasing at all?
5. Try even more conservative settings (lr=5e-5, slower schedule)

**Key Insight**: CVXPyLayers requires careful tuning. When in doubt, use more conservative settings.

---

**End of Training Notes**
