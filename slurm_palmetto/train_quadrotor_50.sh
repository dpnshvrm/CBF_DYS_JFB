#!/bin/bash
#SBATCH --job-name=quad50
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Exit on error
set -eo pipefail

# Check if running under SLURM
if [ -z "${SLURM_JOB_ID}" ]; then
    echo "ERROR: This script must be submitted via 'sbatch', not run directly!"
    echo "Usage: sbatch slurm_palmetto/train_quadrotor_50.sh"
    exit 1
fi

source /etc/profile 2>/dev/null || true

# Load CUDA and other necessary modules
module load cuda/12.1.1
module load cudnn/8.9.6
module load anaconda3/2023.09-0

ENV_PATH="${ENV_PATH:-/scratch/dverma/cbf-quadrotor-torch}"
REPO_DIR="${REPO_DIR:-/home/dverma/CBF_DYS_JFB}"

source activate "${ENV_PATH}"

# Add conda lib paths
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

# Add CUDNN paths from both conda and module
CUDNN_PATH="${ENV_PATH}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "${REPO_DIR}/slurm_logs"
cd "${REPO_DIR}"

echo "Running on $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Env : ${ENV_PATH}"
echo "CUDA module: $(module list 2>&1 | grep cuda || echo 'none')"
echo ""

nvidia-smi || echo "nvidia-smi not available"
echo ""

echo "Loaded modules:"
module list 2>&1
echo ""

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

echo "Checking PyTorch installation..."
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo ""
echo "Starting training..."
python train_quadrotor_50.py
