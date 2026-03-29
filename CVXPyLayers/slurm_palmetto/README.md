# Palmetto SLURM Scripts for CVXPyLayers Training

## Quick Start

```bash
# From repository root
sbatch CVXPyLayers/slurm_palmetto/train_quadrotor_multi.sh
```

## Scripts

### `train_quadrotor_multi.sh`
Trains multi-quadrotor CBF controller (5 agents, 3D obstacles, HOCBF)

**Default settings:**
- 1000 epochs
- Learning rate: 0.001 (decays at epoch 600)
- Network: 64 hidden dim, 3 ResBlocks
- GPU: H200
- Memory: 64GB
- Time limit: 24 hours

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
- **Best model:** `models/best_model.pth` (lowest loss during training)
- **Final model:** `models/final_model.pth` (model at epoch N)

### Training Data
- **Training history:** `models/training_history.csv` (all metrics per epoch)
- **Configuration:** `models/config.json` (all hyperparameters)

### Visualizations
- **Training curves:** `models/quadrotor_training_curves.png` (6-panel figure)
- **Trajectories:** `trajs.png` (updated every 50 epochs, in repo root)

### Logs
- **SLURM logs:** `slurm_logs/cvxpy_quad-<JOB_ID>.{out,err}`
