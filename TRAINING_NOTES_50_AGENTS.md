# Training Notes: 50-Agent Quadrotor System

**Date**: March 22, 2026
**System**: Multi-agent quadrotor navigation with CBF constraints
**Environment**: Palmetto Cluster (H200 GPU)

---

## Problem Setup

### Configuration
- **Agents**: 50 quadrotors
- **State dimension**: 600 (12 per agent: position, angles, velocities, angular velocities)
- **Control dimension**: 200 (4 per agent: thrust + 3 torques)
- **Horizon**: 12 seconds (120 timesteps, dt=0.1s)
- **Obstacles**: 3 spheres creating navigation challenge

### Target Formations

#### Circle Formation (Easier)
```python
TARGET_TYPE = 'circle'
TARGET_CENTER = [1.5, 3.0]
TARGET_RADIUS = 0.5
```
- All agents end in circular formation
- Similar path lengths
- Good for initial learning

#### Horizontal Line Formation (Harder)
```python
TARGET_TYPE = 'horizontal_line'
TARGET_LINE_X_MIN = 0.3  # or 0.8 for easier version
TARGET_LINE_X_MAX = 2.7  # or 2.2 for easier version
TARGET_LINE_Y = 3.0
```
- 50 agents spread across 2.4m line
- Spacing: 0.048m (48mm) between agents
- Forces different navigation strategies:
  - Left agents go around left obstacle
  - Right agents go around right obstacle
  - Center agents navigate through gaps
- **Much harder** - requires coordination

### Obstacle Configuration

#### Current Setup (After tuning)
```python
obstacle_cfg = [
    [1.0, 1.2, 1.0],   # left obstacle
    [2.0, 1.2, 1.0],   # right obstacle
    [1.5, 2.0, 1.0],   # top-center obstacle
]
obstacle_radius = 0.28
eps_safe = 0.10  # Safety margin
```

**Geometry**:
- Start: Circle at (1.5, 0.0), radius 0.5
- Swarm spans: x ∈ [1.0, 2.0]
- Gap between left/right obstacles: ~1.0m (minus 2×0.38m = 0.24m navigable)
- Obstacles form inverted triangle pattern

#### Design Considerations
- **Too close**: DYS solver fails (max iterations, high residuals)
- **Too far**: Problem becomes trivial
- **Current**: Challenging but feasible with proper tuning

---

## Training Challenges & Solutions

### Challenge 1: DYS Solver Hitting Max Iterations

**Symptom**:
```
iters=4999 res=0.3-0.6
```
Every timestep hits maximum iterations, high residuals

**Root Cause**:
- CBF constraints too tight
- Problem infeasible or nearly infeasible
- Solver can't find valid control that satisfies all constraints

**Solutions**:
1. **Relax tolerance** (quadrotor_multi_50.py line 212):
   ```python
   proj(u_nom, K_cbf, d_cbf, max_iter=5000, tol=1e-2)  # was 5e-3
   ```

2. **Widen obstacles** (see obstacle config above)

3. **More control authority**:
   ```python
   T_dev_scale = 0.5   # ±50% thrust deviation from hover
   tau_scale = 0.12    # ±0.12 Nm torque
   ```

### Challenge 2: Terminal Cost Not Converging

**Symptom**:
```
term=70-100, fluctuating heavily
```

**Root Cause**:
- `alpha_terminal = 120` too high too early
- Network forced to achieve perfect alignment before learning navigation
- Gradient saturation

**Solution - Alpha Terminal Scheduling**:

#### Recommended Schedule
```python
# Start low, gradually increase
alpha_terminal = 10.0  # Starting value
alpha_sched_every = 200  # Increase every 200 epochs
alpha_sched_step = 5.0   # Increment by 5
alpha_sched_max = 60.0   # Stop at 60 (NOT 120!)
```

#### Implementation Pattern
```python
_alpha = alpha_terminal  # Mutable copy

# In training loop:
if epoch % alpha_sched_every == 0 and _alpha < alpha_sched_max:
    _alpha += alpha_sched_step
    print(f'  alpha_terminal -> {_alpha:.1f}')

# Use _alpha in loss computation
total_cost, ... = compute_loss(..., alpha_terminal=_alpha, ...)
```

#### Rationale
- **Epoch 0-200**: alpha=10 → Learn basic navigation, avoid obstacles
- **Epoch 200-400**: alpha=15 → Start caring about target positions
- **Epoch 400-600**: alpha=20 → Improve alignment
- **Epoch 600+**: alpha increases → Progressive refinement
- **Cap at 60**: Accept ~15-25 terminal cost as SUCCESS for 50 agents

### Challenge 3: Gradient Clipping

**Symptom**:
```
grad=1.00e+01 (always exactly 10.0)
```

**Root Cause**:
- Hitting gradient clip limit every step
- Network can't learn effectively

**Solution**:
```python
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20.0)  # was 10.0
```

Or remove clipping entirely once stable.

---

## Network Architecture

### Current Configuration
```python
hidden_dim = 512    # Up from 256 for 50 agents
n_blocks = 5        # Up from 4
T_dev_scale = 0.5
tau_scale = 0.12
```

**Network size**: ~2.6M parameters

**Rationale**:
- 50 agents requires 4× more capacity than 10 agents
- Need to model complex coordination patterns
- Horizontal line: different strategies for different agents

### Alternative Architectures (Not Implemented)
- **Attention-based**: Model agent-agent interactions explicitly
- **Graph Neural Network**: Represent swarm as graph
- **Hierarchical**: Group agents into subswarms

*Current feed-forward network is simpler but may be sufficient*

---

## Hyperparameter Recommendations

### Conservative (Start Here)
```python
learning_rate = 1e-4
batch_size = 32
alpha_running = 1.0
alpha_terminal = 10.0  # Start low!
alpha_sched_step = 5.0
alpha_sched_max = 60.0
n_epochs = 5000
```

**Expected outcome**: Terminal cost 15-30 by epoch 3000

### Aggressive (If Conservative Works)
```python
learning_rate = 2e-4
batch_size = 64
alpha_terminal = 20.0  # Start higher
alpha_sched_step = 10.0
alpha_sched_max = 100.0
```

**Risk**: May diverge or get stuck

### For Circle Target (Easier)
```python
alpha_terminal = 20.0
alpha_sched_max = 120.0  # Can go higher
# Expect terminal cost < 10 achievable
```

---

## Success Metrics

### What "Good" Looks Like

| Metric | Circle Target | Horizontal Line | Notes |
|--------|---------------|-----------------|-------|
| Terminal cost | 5-15 | 15-30 | Lower is better |
| DYS iterations | < 2000 | < 3000 | Max is 4999 |
| Residual | < 0.1 | < 0.3 | Solver convergence |
| Barrier min | > 0.05 | > 0.03 | Safety margin |
| Training time | ~3-4 hrs | ~5-6 hrs | On H200 |

### Red Flags
- `iters=4999` for > 50% of steps → Problem too hard
- `term > 100` after epoch 500 → alpha_terminal too high
- `grad=10.0` constantly → Increase clip limit
- `h_min < 0` → Constraint violation (bad!)

---

## Curriculum Learning Strategy

### Phase 1: Learn Navigation (Epochs 0-1000)
```python
TARGET_TYPE = 'circle'
alpha_terminal = 10.0 → 40.0
```
**Goal**: Learn to navigate around obstacles, basic formation

### Phase 2: Introduce Line (Epochs 1000-2000)
```python
TARGET_TYPE = 'horizontal_line'
TARGET_LINE_X_MIN = 0.8  # Shorter line
TARGET_LINE_X_MAX = 2.2
alpha_terminal = 20.0 → 50.0
```
**Goal**: Learn distributed targets, different paths

### Phase 3: Full Challenge (Epochs 2000+)
```python
TARGET_LINE_X_MIN = 0.3  # Full line
TARGET_LINE_X_MAX = 2.7
alpha_terminal = 30.0 → 60.0
```
**Goal**: Refine coordination, achieve tight formation

**Implementation**: Requires modifying training script to switch configs

---

## Visualization Improvements

### 3D View Enhancements
```python
# Fading trajectory (light → dark shows direction)
for k in range(nt - 1):
    alpha = 0.15 + 0.85 * k / nt
    ax3d.plot(xs[k:k+2], ys[k:k+2], zs[k:k+2],
              color=colors[a % 10], alpha=alpha)

# High-quality obstacle rendering
u = np.linspace(0, 2 * np.pi, 40)  # More points = smoother
v = np.linspace(0, np.pi, 40)
plot_surface(..., shade=True, antialiased=True)

# Proper aspect ratio
ax3d.set_box_aspect([1, 1, 0.5])  # X,Y equal; Z compressed
```

### For Horizontal Line
```python
# Draw connecting line to show formation
if TARGET_TYPE == 'horizontal_line':
    ax.plot(tgt[:, 0], tgt[:, 1], 'g--', linewidth=2,
            alpha=0.5, label='Target line')
```

---

## Known Issues & Workarounds

### Issue: PyTorch CUDA Library Loading
**Error**: `libtorch_global_deps.so: cannot open shared object file`

**Fix**: Load CUDA module in SLURM script
```bash
module load cuda/12.3.0  # Match PyTorch version
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH}"
```

### Issue: Flat Obstacle Spheres in 3D
**Cause**: Wrong aspect ratio

**Fix**:
```python
ax3d.set_box_aspect([1, 1, 0.5])
```

### Issue: Training Stalls After Epoch 1000
**Likely cause**: Learning rate too low

**Fix**: Add LR scheduler or restart with lower alpha_terminal

---

## File Structure

```
CBF_DYS_JFB/
├── quadrotor_multi_50.py          # Dynamics, CBF, target generation
├── train_quadrotor_50.py          # Training script
├── test_visualization_50.py       # Quick viz test (no training)
├── slurm_palmetto/
│   └── train_quadrotor_50.sh      # SLURM submission script
└── results_quadrotor_50/          # Outputs
    ├── placement_50.png           # Initial configuration
    ├── traj_xy_epoch_*.png        # Trajectory snapshots
    ├── training_curves_50.png     # Loss curves
    └── quadrotor_control_net_50_*.pth  # Checkpoints
```

---

## Quick Reference: Training on Palmetto

### Submit Job
```bash
cd /home/dverma/CBF_DYS_JFB
sbatch slurm_palmetto/train_quadrotor_50.sh
```

### Monitor Progress
```bash
# Check queue
squeue -u dverma

# Watch training log
tail -f slurm_logs/quad50-*.out

# Check latest metrics
tail -20 slurm_logs/quad50-*.out | grep "ep.*total"
```

### Download Results
```bash
# On local machine
scp dverma@login.palmetto.clemson.edu:/home/dverma/CBF_DYS_JFB/results_quadrotor_50/*.png ./
scp dverma@login.palmetto.clemson.edu:/home/dverma/CBF_DYS_JFB/results_quadrotor_50/*.pth ./
```

---

## Future Improvements

### Short-term
- [ ] Implement alpha_terminal scheduling
- [ ] Relax DYS tolerance to 1e-2
- [ ] Test circle target first (easier)
- [ ] Increase gradient clip to 20.0

### Medium-term
- [ ] Curriculum learning: circle → short line → full line
- [ ] Adaptive scheduling based on loss plateau
- [ ] Better initial conditions (learned policy)
- [ ] Analyze which agents fail most (edge vs center)

### Long-term
- [ ] Attention-based architecture
- [ ] Multi-scale approach (groups of 10 → full 50)
- [ ] Obstacle avoidance prediction (anticipate conflicts)
- [ ] Real-time replanning (dynamic obstacles)

---

## Contact & References

**Repository**: `/home/dverma/CBF_DYS_JFB` on Palmetto
**Base code**: 10-agent version in `train_quadrotor.py`, `quadrotor_multi.py`
**Key papers**:
- Control Barrier Functions (CBF)
- Differentiable Optimization (DYS/JFB backprop)
- Multi-agent coordination

---

## Changelog

### 2026-03-22
- Initial 50-agent setup
- Horizontal line target implementation
- Obstacle tuning (3 spheres, inverted triangle)
- Network scaling (256→512 hidden, 4→5 blocks)
- Identified need for alpha_terminal scheduling
- Documented challenges and solutions

---

**Remember**: Terminal cost 15-30 is SUCCESS for 50 agents in tight formation!
