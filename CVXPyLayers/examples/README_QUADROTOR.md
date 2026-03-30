# Quadrotor Training Examples

## Two Implementations

### 1. `train_quadrotor_multi.py` (DYS-based, original)
- **Solver:** Custom DYSProjector (PyTorch fixed-point iteration)
- **Source:** Imports from parent repo (`quadcopter_multi.py`, `utils.py`)
- **GPU:** Required (custom PyTorch solver)
- **Status:** Works but doesn't use CVXPyLayers framework

### 2. `train_quadrotor_multi_cvxpy.py` (CVXPyLayers-based) ✨ NEW
- **Solver:** CVXPyLayers (OSQP/ECOS with backprop)
- **Source:** Uses CVXPyLayers modules (`dynamics.Quadrotor`, `barriers.SphericalObstacle`)
- **GPU:** Optional (CPU solvers, but GPU speeds up network training)
- **Status:** Properly integrated with CVXPyLayers framework

## New CVXPyLayers Modules Created

### Dynamics
- `CVXPyLayers/dynamics/quadrotor.py`
  - `Quadrotor` class with multi-agent support
  - State: [pos, angles, vel, ang_vel] × n_agent
  - Control: [thrust, angular_accel] × n_agent
  - Relative degree 2 for position-based barriers

### Barriers
- `CVXPyLayers/barriers/spherical_obstacle.py`
  - `SphericalObstacle` class for 3D obstacles
  - HOCBF constraints (relative degree 2)
  - Multi-agent support (one constraint per agent per obstacle)

## Usage

### CVXPyLayers version (recommended):
```bash
cd CVXPyLayers/examples
python train_quadrotor_multi_cvxpy.py --epochs 1000 --lr 0.001
```

### DYS version (original):
```bash
cd CVXPyLayers/examples
python train_quadrotor_multi.py --epochs 1000 --lr 0.001
```

## Outputs

Both save to `models/`:
- **CVXPy version:** `*_cvxpy.{pth,csv,json}`
- **DYS version:** Regular filenames

## Key Differences

| Feature | DYS Version | CVXPyLayers Version |
|---------|------------|---------------------|
| QP Solver | Custom DYS | CVXPyLayers (OSQP/ECOS) |
| Framework | Standalone | Integrated with CVXPyLayers |
| Visualization | 3D plots via quadcopter_multi | Not yet implemented |
| GPU Requirement | Required | Optional |
| Modularity | Monolithic | Modular (dynamics, barriers, controllers) |

## TODO

For CVXPyLayers version:
- [ ] Add 3D trajectory visualization
- [ ] Test convergence vs DYS version
- [ ] Benchmark performance
- [ ] Add to SLURM scripts
