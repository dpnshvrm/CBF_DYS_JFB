# Palmetto SLURM Scripts for CVXPyLayers Training

## Quick Start

```bash
# From repository root
sbatch CVXPyLayers/slurm_palmetto/train_quadrotor_multi.sh
```

## Scripts

### `train_quadrotor_multi.sh`
Trains multi-quadrotor CBF controller using **CVXPyLayers** framework (10 agents, 3D obstacles, HOCBF)

**Default settings:**
- 1000 epochs
- Learning rate: 1e-4 (conservative, decays at epoch 600)
- Network: 64 hidden dim, 3 ResBlocks
- **GPU: H200** (neural network on GPU, CVXPy solvers on CPU)
- Partition: nextlab200
- CPUs: 4
- Memory: 32GB
- Time limit: 24 hours

**Why GPU?**
- Neural network training/inference runs on GPU (faster)
- CVXPy QP solvers run on CPU (OSQP/ECOS)
- Best of both worlds: fast NN + reliable solvers

**Modify settings:**
Edit the script and change the arguments in the final `python` command.

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f slurm_logs/cvxpy_quad-<JOB_ID>.out
```

View errors:
```bash
tail -f slurm_logs/cvxpy_quad-<JOB_ID>.err
```

## Output Files

### Models
- **Best model:** `models/best_model_cvxpy.pth` (lowest loss during training)
- **Final model:** `models/final_model_cvxpy.pth` (model at epoch N)

### Training Data
- **Training history:** `models/training_history_cvxpy.csv` (all metrics per epoch including alpha_terminal schedule)
- **Configuration:** `models/config_cvxpy.json` (all hyperparameters)

### Logs
- **SLURM logs:** `slurm_logs/cvxpy_quad-<JOB_ID>.{out,err}`

**Note:** Files are suffixed with `_cvxpy` to distinguish from DYS-based training runs.
