#!/bin/bash
#SBATCH --job-name=prope-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=interactive_singlenode
#SBATCH --account=nvr_torontoai_videogen
#SBATCH --output=slurm-logs/prope-parity-%j.out
#SBATCH --error=slurm-logs/prope-parity-%j.err
#
# Run the PRoPE numerical-parity test that compares our port at
# integrations/artifixer/artifixer/network/prope.py against the dreamfix
# reference at model_training/net/prope.py.
#
# The artifixer container has torch installed at the system Python level
# and the dreamfix repo is mounted via /lustre, so we don't need uv sync.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
DREAMFIX_REPO_ROOT="${DREAMFIX_REPO_ROOT:-/lustre/fsw/portfolios/nvr/users/rdelutio/code/dreamfix}"
CONTAINER_IMAGE="${ARTIFIXER_CONTAINER_IMAGE:-/lustre/fsw/portfolios/nvr/users/hturki/containers/artifixer-cuda12.sqsh}"

COMMAND="set -euo pipefail; \
cd ${REPO_DIR} && \
export PYTHONPATH=${REPO_DIR}/integrations/artifixer:${REPO_DIR}/flashdreams:${DREAMFIX_REPO_ROOT}:\${PYTHONPATH:-} && \
export DREAMFIX_REPO_ROOT='${DREAMFIX_REPO_ROOT}' && \
python3 -m pip install --user pytest loguru einops boto3 tyro 2>&1 | tail -3 && \
python3 -m pytest -v integrations/artifixer/tests/test_prope_parity.py integrations/artifixer/tests/test_patches.py integrations/artifixer/tests/test_latent_mix.py integrations/artifixer/tests/test_smoke.py::test_compute_kv_neighbor_and_cache_init"

echo "[slurm] command: ${COMMAND}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:/home,/lustre:/lustre" \
     --container-workdir="${REPO_DIR}" \
     bash -c "${COMMAND}"
