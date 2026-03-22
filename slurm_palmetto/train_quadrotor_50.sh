#!/bin/bash
#SBATCH --job-name=quad50
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -eo pipefail

if [ -z "${SLURM_JOB_ID}" ]; then
    echo "ERROR: Submit via sbatch, not directly."
    exit 1
fi

source /etc/profile 2>/dev/null || true

module load cuda/12.3.0
module load anaconda3/2023.09-0

ENV_PATH="${ENV_PATH:-/scratch/dverma/cbf-quadrotor-torch}"
REPO_DIR="${REPO_DIR:-/home/dverma/CBF_DYS_JFB}"

conda activate "${ENV_PATH}"

export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

CUDNN_PATH="${ENV_PATH}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
    export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "${REPO_DIR}/slurm_logs"
cd "${REPO_DIR}"

echo "Host: $(hostname)"
echo "CUDA: $(module list 2>&1 | grep cuda || echo 'none')"
nvidia-smi || echo "nvidia-smi not available"
module list 2>&1
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

echo "Starting training..."
python train_quadrotor_50.py