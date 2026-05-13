#!/bin/bash
#SBATCH --job-name=fd-artifixer-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=interactive_singlenode
#SBATCH --account=nvr_torontoai_videogen
#SBATCH --output=slurm-logs/fd-artifixer-smoke-%j.out
#SBATCH --error=slurm-logs/fd-artifixer-smoke-%j.err
#
# Run the artifixer plugin smoke tests inside the FlashDreams container.
#
# Phase 1: smoke tests are dataclass / entry-point validation only; they
# need ``flashdreams`` importable + pytest. The flashdreams container ships
# with ``uv`` but not a pre-baked venv, so first invocation runs ``uv sync``
# (slow). Subsequent runs in the same workspace reuse ``.venv/``.
#
# If ``uv sync`` is too slow for iteration, run the static checks in
# ``scripts/static_check_artifixer.sh`` instead (no env, no GPU, ~1s).
#
# Requires:
#   - ``/lustre/fsw/portfolios/nvr/users/rdelutio/containers/flashdreams-base-v0.3.sqsh``
#     created via ``scripts/import_flashdreams_sqsh.sh`` (one-time)
#   - GitHub PAT for ghcr.io (only for first sqsh import)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
if [ -z "$REPO_DIR" ]; then
    echo "Could not determine REPO_DIR; set REPO_DIR or submit from inside the repo." >&2
    exit 1
fi

CONTAINER_IMAGE="${FLASHDREAMS_CONTAINER_IMAGE:-/lustre/fsw/portfolios/nvr/users/rdelutio/containers/flashdreams-base-v0.3.sqsh}"

COMMAND="set -euo pipefail; \
cd ${REPO_DIR} && \
uv sync --extra dev --extra runners --group lint && \
uv run --extra dev pytest -q integrations/artifixer/tests"

echo "[slurm] command: ${COMMAND}"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="$HOME:/home,/lustre:/lustre" \
     --container-workdir="${REPO_DIR}" \
     bash -c "${COMMAND}"
