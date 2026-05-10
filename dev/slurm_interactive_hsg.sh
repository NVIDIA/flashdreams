#!/usr/bin/env bash
# HSG-compatible interactive Slurm launcher for FlashDreams.
#
# Usage:
#   bash dev/slurm_interactive_hsg.sh [NUM_GPUS] [OPTIONS]
#
# Examples:
#   # One full GB200 node, required by HSG GPU policy.
#   bash dev/slurm_interactive_hsg.sh
#
#   # Two full GB200 nodes.
#   bash dev/slurm_interactive_hsg.sh 4 --nodes 2
#
#   # Fallback when the prebuilt container registry is not readable.
#   bash dev/slurm_interactive_hsg.sh --no-container

set -euo pipefail

NUM_GPUS="${SLURM_GPUS_PER_NODE:-4}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-nvr_torontoai_videogen}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-interactive}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_CPUS_PER_GPU="${SLURM_CPUS_PER_GPU:-36}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_COMMENT="${SLURM_COMMENT:-fact_off}"
NO_CONTAINER="${FLASHDREAMS_NO_CONTAINER:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)          SLURM_ACCOUNT="$2";           shift 2 ;;
        --account=*)        SLURM_ACCOUNT="${1#*=}";      shift   ;;
        --partition)        SLURM_PARTITION="$2";         shift 2 ;;
        --partition=*)      SLURM_PARTITION="${1#*=}";    shift   ;;
        --qos)              SLURM_QOS="$2";               shift 2 ;;
        --qos=*)            SLURM_QOS="${1#*=}";          shift   ;;
        --nodes)            SLURM_NODES="$2";             shift 2 ;;
        --nodes=*)          SLURM_NODES="${1#*=}";        shift   ;;
        --cpus-per-gpu)     SLURM_CPUS_PER_GPU="$2";      shift 2 ;;
        --cpus-per-gpu=*)   SLURM_CPUS_PER_GPU="${1#*=}"; shift   ;;
        --time)             SLURM_TIME="$2";              shift 2 ;;
        --time=*)           SLURM_TIME="${1#*=}";         shift   ;;
        --comment)          SLURM_COMMENT="$2";           shift 2 ;;
        --comment=*)        SLURM_COMMENT="${1#*=}";      shift   ;;
        --no-container)     NO_CONTAINER=1;               shift   ;;
        -h|--help)          awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        --*)                echo "Unknown option: $1" >&2; exit 2 ;;
        *)                  NUM_GPUS="$1";                shift   ;;
    esac
done

if [[ "${NUM_GPUS}" != "4" ]]; then
    echo "HSG GB200 GPU jobs require whole nodes; use 4 GPUs per node." >&2
    exit 2
fi

REPO_ROOT="${FLASHDREAMS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FLASHDREAMS_TEST_IMAGE:-docker://ghcr.io#nvidia/flashdreams:base-v0.3-20260424-55bd566}"
CACHE_ROOT="${FLASHDREAMS_RUNTIME_CACHE_ROOT:-${REPO_ROOT}/.cache/runtime}"

UV_CACHE_HOST="${FLASHDREAMS_UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
HF_CACHE_HOST="${FLASHDREAMS_HF_CACHE_DIR:-${CACHE_ROOT}/huggingface}"
FLASHDREAMS_CACHE_HOST="${FLASHDREAMS_CACHE_DIR:-${CACHE_ROOT}/flashdreams}"
TRITON_CACHE_HOST="${FLASHDREAMS_TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
JOB_HOME_HOST="${FLASHDREAMS_JOB_HOME:-${CACHE_ROOT}/home}"
ENROOT_CONFIG_HOST="${FLASHDREAMS_ENROOT_CONFIG_DIR:-${CACHE_ROOT}/enroot}"

mkdir -p \
    "${UV_CACHE_HOST}" \
    "${HF_CACHE_HOST}" \
    "${FLASHDREAMS_CACHE_HOST}" \
    "${TRITON_CACHE_HOST}" \
    "${JOB_HOME_HOST}" \
    "${ENROOT_CONFIG_HOST}"

SRUN_ARGS=(
    srun
    -A "${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --qos="${SLURM_QOS}"
    --nodes="${SLURM_NODES}"
    --gpus-per-node="${NUM_GPUS}"
    --cpus-per-gpu="${SLURM_CPUS_PER_GPU}"
    --time="${SLURM_TIME}"
    --exclusive
    --comment="${SLURM_COMMENT}"
)

if [[ "${NO_CONTAINER}" == "1" ]]; then
    "${SRUN_ARGS[@]}" \
        --pty \
        --chdir="${REPO_ROOT}" \
        --export=ALL,HOME="${JOB_HOME_HOST}",UV_CACHE_DIR="${UV_CACHE_HOST}",HF_HOME="${HF_CACHE_HOST}",TRITON_CACHE_DIR="${TRITON_CACHE_HOST}",FLASHDREAMS_CACHE_DIR="${FLASHDREAMS_CACHE_HOST}",ENROOT_CONFIG_PATH="${ENROOT_CONFIG_HOST}",UV_LINK_MODE=copy \
        /bin/bash -lc 'ulimit -s 8192; exec /bin/bash'
    exit
fi

"${SRUN_ARGS[@]}" \
    --pty \
    --container-image="${IMAGE}" \
    --container-mounts="${REPO_ROOT}:/workspace/flashdreams,${JOB_HOME_HOST}:/workspace/home,${UV_CACHE_HOST}:/root/.cache/uv,${HF_CACHE_HOST}:/root/.cache/huggingface,${FLASHDREAMS_CACHE_HOST}:/root/.cache/flashdreams,${TRITON_CACHE_HOST}:/root/.cache/triton,/lustre:/lustre,/cm:/cm,/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d,/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels" \
    --container-workdir=/workspace/flashdreams \
    --container-writable \
    --container-remap-root \
    --export=ALL,HOME=/workspace/home,HF_HOME=/root/.cache/huggingface,UV_LINK_MODE=copy,TRITON_CACHE_DIR=/root/.cache/triton,FLASHDREAMS_CACHE_DIR=/root/.cache/flashdreams,ENROOT_CONFIG_PATH="${ENROOT_CONFIG_HOST}" \
    /bin/bash -lc 'ulimit -s 8192; exec /bin/bash'
