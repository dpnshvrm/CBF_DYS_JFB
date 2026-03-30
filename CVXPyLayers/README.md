# CVXPyLayers CBF Framework

**Differentiable Control Barrier Functions using CVXPyLayers**

A modular framework for training neural network policies with safety guarantees through Control Barrier Functions (CBF).

---

## 📚 Documentation

**→ [ARCHITECTURE.md](ARCHITECTURE.md)** - **READ THIS FIRST!**
Comprehensive guide explaining:
- How CVXPyLayers works
- What each component does
- Mathematical background
- Data flow through the system
- Differences between examples

---

## 🚀 Quick Start

### Example 1: Single Integrator (Easiest)
```bash
cd examples
python train_single_integrator.py
```
- 2D navigation with velocity control
- 2 circular obstacles
- Standard CBF (relative degree 1)

### Example 2: Double Integrator
```bash
cd examples
python train_double_integrator.py
```
- 2D navigation with acceleration control
- 3 circular obstacles
- HOCBF (relative degree 2)

### Example 3: Multi-Agent Quadrotor (Advanced)
```bash
cd examples
python train_quadrotor_multi_cvxpy.py --epochs 1000
```
- 10 quadrotors in 3D
- 3 spherical obstacles
- HOCBF with multi-agent constraints

---

## 🏗️ Framework Structure

```
CVXPyLayers/
├── dynamics/              # System dynamics
│   ├── single_integrator.py
│   ├── double_integrator.py
│   └── quadrotor.py
│
├── barriers/              # Safety barriers
│   ├── circular_obstacle.py   (2D)
│   └── spherical_obstacle.py  (3D)
│
├── controllers/           # CBF-QP layer
│   ├── cbf_qp_layer.py        (CVXPyLayers wrapper)
│   └── policy_network.py
│
├── training/              # Training utilities
│   ├── config.py
│   └── trainer.py
│
├── visualization/         # Plotting
│   └── plotting.py
│
└── examples/              # Runnable examples
    ├── train_single_integrator.py
    ├── train_double_integrator.py
    └── train_quadrotor_multi_cvxpy.py
```

---

## 🧠 Key Concepts

### Control Barrier Functions (CBF)
Enforce safety by constraining control:
```
h(x) > 0  ⇒  System is SAFE
h(x) ≤ 0  ⇒  UNSAFE (collision)
```

### CVXPyLayers Magic
Solves a safety-constrained QP at each timestep:
```
minimize:    ||u - u_desired||²
subject to:  CBF constraints
```
**And backpropagates through the solution!**

### The Pipeline
```
State → Policy NN → u_desired → [CBF-QP] → u_safe → Dynamics → Next State
         ↑_____________________ Gradients flow back! ____________________↓
```

---

## 📊 Outputs

Each training run saves:
- **Models:** `models/best_model.pth`, `models/final_model.pth`
- **History:** `models/training_history.csv` (all metrics per epoch)
- **Config:** `models/config.json` (full configuration)
- **Plots:** Trajectory visualizations

---

## 🖥️ Running on Palmetto

```bash
# Submit to cluster
sbatch CVXPyLayers/slurm_palmetto/train_quadrotor_multi.sh

# Monitor
squeue -u $USER
tail -f slurm_logs/cvxpy_quad-*.out
```

See `slurm_palmetto/README.md` for details.

---

## 🔬 Research Use

### Citation
If you use this framework, please cite:
- CVXPyLayers: [Agrawal et al., NeurIPS 2019](https://arxiv.org/abs/1910.12430)
- Control Barrier Functions: [Ames et al., 2019](https://arxiv.org/abs/1903.11199)

### Extensions
- Multi-agent coordination
- Dynamic obstacles
- High-dimensional systems
- Custom dynamics/barriers

---

## 🐛 Troubleshooting

### Import Errors
Make sure you're in the right directory:
```bash
cd CVXPyLayers/examples
python train_single_integrator.py
```

### CVXPy Solver Errors
Install required solvers:
```bash
pip install cvxpy osqp ecos
```

### Memory Issues
Reduce batch size in config:
```python
batch_size = 16  # Instead of 32
```

---

## 📖 Learn More

1. **Start here:** [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Quadrotor details:** [examples/README_QUADROTOR.md](examples/README_QUADROTOR.md)
3. **Training on cluster:** [slurm_palmetto/README.md](slurm_palmetto/README.md)

---

## 🤝 Contributing

To add a new dynamics/barrier:
1. Inherit from base class (`ControlAffineDynamics` or `BarrierFunction`)
2. Implement required methods
3. Add to `__init__.py`
4. Create example script

See existing implementations for templates!

---

**Questions?** Read [ARCHITECTURE.md](ARCHITECTURE.md) - it explains everything in detail!
