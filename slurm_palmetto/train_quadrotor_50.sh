#!/bin/bash
#SBATCH --job-name=quad50
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

source /etc/profile
module load anaconda3/2023.09-0

ENV_PATH="${ENV_PATH:-/scratch/dverma/cbf-quadrotor-torch}"
REPO_DIR="${REPO_DIR:-/home/dverma/CBF_DYS_JFB}"

source activate "${ENV_PATH}"

CUDNN_PATH="${ENV_PATH}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH:-}"
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "${REPO_DIR}/slurm_logs"
cd "${REPO_DIR}"

echo "Running on $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Env : ${ENV_PATH}"
nvidia-smi || true

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python train_quadrotor_50.py
