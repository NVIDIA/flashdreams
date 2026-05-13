#!/bin/bash
#SBATCH --job-name=fd-import
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=interactive_singlenode
#SBATCH --account=nvr_torontoai_videogen
#SBATCH --output=slurm-logs/fd-import-%j.out
#SBATCH --error=slurm-logs/fd-import-%j.err
#
# One-shot: pull the FlashDreams container from ghcr.io into a sqsh file
# under /lustre so later slurm jobs can reuse it via
# ``--container-image=/lustre/.../flashdreams-base-v0.3.sqsh`` instead of
# re-pulling from the registry on every worker.

set -euo pipefail

OUTPUT_SQSH="${1:-/lustre/fsw/portfolios/nvr/users/rdelutio/containers/flashdreams-base-v0.3.sqsh}"
IMAGE_URI="${2:-ghcr.io#nvidia/flashdreams:base-v0.3-20260424-55bd566}"

mkdir -p "$(dirname "$OUTPUT_SQSH")"

if [ -f "$OUTPUT_SQSH" ]; then
    echo "[import] $OUTPUT_SQSH already exists; refusing to overwrite. Delete or pass a new path." >&2
    exit 0
fi

# Use a per-job ENROOT_TEMP / RUNTIME directory under the worker's /raid to
# avoid clashes with concurrent jobs that share /raid/containers.
JOB_RAID_DIR="/raid/containers/tmp/${USER}-${SLURM_JOB_ID:-import}"
mkdir -p "${JOB_RAID_DIR}"
export ENROOT_TEMP_PATH="${JOB_RAID_DIR}/tmp"
export ENROOT_RUNTIME_PATH="${JOB_RAID_DIR}/runtime"
export ENROOT_DATA_PATH="${JOB_RAID_DIR}/data"
mkdir -p "${ENROOT_TEMP_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_DATA_PATH}"

echo "[import] target sqsh: $OUTPUT_SQSH"
echo "[import] image uri:   docker://${IMAGE_URI}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    bash -c "enroot import --output '${OUTPUT_SQSH}' 'docker://${IMAGE_URI}'"

echo "[import] done. Size:"
ls -lh "${OUTPUT_SQSH}"
