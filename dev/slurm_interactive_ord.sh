#!/usr/bin/env bash
# ORD-compatible interactive Slurm launcher for FlashDreams.
#
# Usage:
#   bash dev/slurm_interactive_ord.sh [NUM_GPUS] [OPTIONS]
#
# Examples:
#   # 4 GPUs using the ORD defaults
#   bash dev/slurm_interactive_ord.sh 4
#
#   # Request a two-node interactive shell
#   bash dev/slurm_interactive_ord.sh 4 --nodes 2

set -euo pipefail

NUM_GPUS="${SLURM_GPUS_PER_NODE:-4}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-nvr_torontoai_videogen}"
SLURM_PARTITION="${SLURM_PARTITION:-interactive,polar,grizzly,polar3,polar4}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_CPUS_PER_GPU="${SLURM_CPUS_PER_GPU:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_EXCLUSIVE="${SLURM_EXCLUSIVE:-auto}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)          SLURM_ACCOUNT="$2";           shift 2 ;;
        --account=*)        SLURM_ACCOUNT="${1#*=}";      shift   ;;
        --partition)        SLURM_PARTITION="$2";         shift 2 ;;
        --partition=*)      SLURM_PARTITION="${1#*=}";    shift   ;;
        --nodes)            SLURM_NODES="$2";             shift 2 ;;
        --nodes=*)          SLURM_NODES="${1#*=}";        shift   ;;
        --cpus-per-gpu)     SLURM_CPUS_PER_GPU="$2";      shift 2 ;;
        --cpus-per-gpu=*)   SLURM_CPUS_PER_GPU="${1#*=}"; shift   ;;
        --qos)              SLURM_QOS="$2";               shift 2 ;;
        --qos=*)            SLURM_QOS="${1#*=}";          shift   ;;
        --no-qos)           SLURM_QOS="";                 shift   ;;
        --time)             SLURM_TIME="$2";              shift 2 ;;
        --time=*)           SLURM_TIME="${1#*=}";         shift   ;;
        --exclusive)        SLURM_EXCLUSIVE=1;            shift   ;;
        --no-exclusive)     SLURM_EXCLUSIVE=0;            shift   ;;
        -h|--help)          awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        --*)                echo "Unknown option: $1" >&2; exit 2 ;;
        *)                  NUM_GPUS="$1";                shift   ;;
    esac
done

REPO_ROOT="${FLASHDREAMS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FLASHDREAMS_TEST_IMAGE:-docker://ghcr.io#nvidia/flashdreams:base-v0.3-20260424-55bd566}"

if [[ -z "${SLURM_CPUS_PER_GPU}" ]]; then
    if (( NUM_GPUS >= 8 )); then
        SLURM_CPUS_PER_GPU=16
    else
        SLURM_CPUS_PER_GPU=30
    fi
fi

if [[ "${SLURM_EXCLUSIVE}" == "auto" ]]; then
    if (( NUM_GPUS >= 8 )); then
        SLURM_EXCLUSIVE=1
    else
        SLURM_EXCLUSIVE=0
    fi
fi

UV_CACHE_HOST="${FLASHDREAMS_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHDREAMS_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHDREAMS_CACHE_HOST="${FLASHDREAMS_CACHE_DIR:-${HOME}/.cache/flashdreams}"
TRITON_CACHE_HOST="${FLASHDREAMS_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHDREAMS_CACHE_HOST}" "${TRITON_CACHE_HOST}"

SRUN_ARGS=(
    srun
    -A "${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --nodes="${SLURM_NODES}"
    --gpus-per-node="${NUM_GPUS}"
    --cpus-per-gpu="${SLURM_CPUS_PER_GPU}"
    --time="${SLURM_TIME}"
)

if [[ -n "${SLURM_QOS}" ]]; then
    SRUN_ARGS+=(--qos="${SLURM_QOS}")
fi

if [[ "${SLURM_EXCLUSIVE}" != "0" ]]; then
    SRUN_ARGS+=(--exclusive)
fi

"${SRUN_ARGS[@]}" \
    --pty \
    --container-image="${IMAGE}" \
    --container-mounts="${REPO_ROOT}:/workspace/flashdreams,${UV_CACHE_HOST}:/root/.cache/uv,${HF_CACHE_HOST}:/root/.cache/huggingface,${FLASHDREAMS_CACHE_HOST}:/root/.cache/flashdreams,${TRITON_CACHE_HOST}:/root/.cache/triton,/lustre:/lustre" \
    --container-workdir=/workspace/flashdreams \
    --container-writable \
    --container-mount-home \
    --container-remap-root \
    --export=ALL,HF_HOME=/root/.cache/huggingface,UV_LINK_MODE=copy,TRITON_CACHE_DIR=/root/.cache/triton \
    /bin/bash
