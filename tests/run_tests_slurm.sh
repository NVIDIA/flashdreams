#!/usr/bin/env bash
# Run the flashdreams test suite on a Slurm-allocated GPU node.
#
# Use this on a login / dev machine without a local GPU. The script submits
# a single srun job that pulls the configured container (Pyxis/enroot),
# installs flashdreams + integration packages on the fly, then invokes pytest.
# Caches for uv / huggingface / flashdreams are bind-mounted from the host.
#
# Usage:
#   ./tests/run_tests_slurm.sh --partition NAME --account NAME
#                              [--qos NAME] [--gpus N] [--cpus-per-gpu N]
#                              [--nodes N] [--time HH:MM:SS]
#                              [--rebuild-image]
#                              [--] [TEST_TARGET...]
#
# Required arguments:
#   --partition NAME   Slurm partition (e.g. batch, interactive)
#   --account  NAME    Slurm account, becomes srun -A (e.g. nvr_torontoai_videogen)
#
# Optional arguments (defaults in brackets):
#   --qos      NAME       Slurm QOS [unset]
#   --gpus     N          --gpus-per-node [1]
#   --cpus-per-gpu N      CPUs allocated per GPU [36]
#   --nodes    N          Number of nodes [1]
#   --time     T          Walltime HH:MM:SS [04:00:00]
#   --rebuild-image       Force re-import of the container image even if cached
#
# Container-image caching:
#   The first run does `enroot import` on the login node and stores the
#   resulting .sqsh under FLASHDREAMSIMAGE_CACHE_DIR. Subsequent runs pass the
#   local .sqsh straight to --container-image, avoiding a multi-minute
#   docker→sqsh conversion inside every srun. To force a rebuild, either pass
#   --rebuild-image or `rm` the .sqsh file.
#
# Environment overrides:
#   FLASHDREAMS_TEST_IMAGE        (default: nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04)
#   FLASHDREAMS_UV_CACHE_DIR      (default: ${HOME}/.cache/uv)
#   FLASHDREAMS_HF_CACHE_DIR      (default: ${HOME}/.cache/huggingface)
#   FLASHDREAMS_CACHE_DIR         (default: ${HOME}/.cache/flashdreams)
#   FLASHDREAMS_TRITON_CACHE_DIR  (default: ${HOME}/.cache/triton)
#   FLASHDREAMSIMAGE_CACHE_DIR   (default: ${FLASHDREAMS_CACHE_DIR}/containers)
#
# Examples:
#   # All tests, full node on the batch partition
#   ./tests/run_tests_slurm.sh --partition batch --account nvr_torontoai_videogen --gpus 4
#
#   # Interactive QOS + a specific test file
#   ./tests/run_tests_slurm.sh --partition batch --account nvr_torontoai_videogen \
#       --qos interactive --gpus 4 --time 02:00:00 \
#       -- flashdreams/tests/test_attention.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHDREAMS_TEST_IMAGE:-nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04}"

SLURM_PARTITION=""
SLURM_ACCOUNT=""
SLURM_QOS=""
SLURM_GPUS="1"
SLURM_CPUS_PER_GPU="36"
SLURM_NODES="1"
SLURM_TIME="04:00:00"
REBUILD_IMAGE=0

usage() {
    sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

slurm_diagnostic_help() {
    cat >&2 <<'MSG'

Hint: Slurm rejected the job. Common causes on this kind of cluster:
  * QOSMinGRES / "Job violates accounting/QOS policy"
      The QOS for this partition requires more GPUs than you requested.
      Try a full node, e.g.  --gpus 8
      Or use a lower-floor QOS, e.g.  --qos interactive  (often paired with
      a smaller --gpus value).
  * Inspect available QOSs/limits:
      sacctmgr show qos format=name,MinTRES,MaxTRESPerUser,MaxJobs,MaxWall
      sinfo -o "%P %a %l %D %G"
MSG
}

TEST_TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)     SLURM_PARTITION="$2";       shift 2 ;;
        --partition=*)   SLURM_PARTITION="${1#*=}";  shift   ;;
        --account)       SLURM_ACCOUNT="$2";         shift 2 ;;
        --account=*)     SLURM_ACCOUNT="${1#*=}";    shift   ;;
        --qos)           SLURM_QOS="$2";             shift 2 ;;
        --qos=*)         SLURM_QOS="${1#*=}";        shift   ;;
        --gpus)          SLURM_GPUS="$2";            shift 2 ;;
        --gpus=*)        SLURM_GPUS="${1#*=}";       shift   ;;
        --cpus-per-gpu)  SLURM_CPUS_PER_GPU="$2";    shift 2 ;;
        --cpus-per-gpu=*) SLURM_CPUS_PER_GPU="${1#*=}"; shift ;;
        --nodes)         SLURM_NODES="$2";           shift 2 ;;
        --nodes=*)       SLURM_NODES="${1#*=}";      shift   ;;
        --time)          SLURM_TIME="$2";            shift 2 ;;
        --time=*)        SLURM_TIME="${1#*=}";       shift   ;;
        --rebuild-image) REBUILD_IMAGE=1;            shift   ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; TEST_TARGETS+=("$@"); break ;;
        --*)             echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)               TEST_TARGETS+=("$1"); shift ;;
    esac
done

if [[ -z "${SLURM_PARTITION}" || -z "${SLURM_ACCOUNT}" ]]; then
    echo "Error: --partition and --account are required." >&2
    usage >&2
    exit 2
fi

UV_CACHE_HOST="${FLASHDREAMS_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHDREAMS_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHDREAMS_CACHE_HOST="${FLASHDREAMS_CACHE_DIR:-${HOME}/.cache/flashdreams}"
TRITON_CACHE_HOST="${FLASHDREAMS_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
IMAGE_CACHE_DIR="${FLASHDREAMSIMAGE_CACHE_DIR:-${FLASHDREAMS_CACHE_HOST}/containers}"
mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHDREAMS_CACHE_HOST}" \
    "${TRITON_CACHE_HOST}" "${IMAGE_CACHE_DIR}"

# Cache the container image as a local .sqsh on the login node so that each
# srun reuses it instead of doing a multi-minute docker→sqsh import.
# Cache key is the image reference with /:@ replaced by _ so it's a flat name.
IMAGE_KEY="$(printf '%s' "${IMAGE}" | tr '/:@' '___')"
IMAGE_SQSH="${IMAGE_CACHE_DIR}/${IMAGE_KEY}.sqsh"

if [[ "${REBUILD_IMAGE}" -eq 1 && -f "${IMAGE_SQSH}" ]]; then
    echo "=== Removing cached image (--rebuild-image): ${IMAGE_SQSH} ==="
    rm -f "${IMAGE_SQSH}"
fi

if [[ ! -f "${IMAGE_SQSH}" ]]; then
    if command -v enroot >/dev/null 2>&1; then
        echo "=== Importing container image to ${IMAGE_SQSH} (one-time, may take several minutes) ==="
        # enroot import accepts both 'docker://registry.example.com/image:tag'
        # and 'docker://registry.example.com#image:tag'. Pass through any
        # explicit URI; otherwise prefix with docker://.
        case "${IMAGE}" in
            *://*) IMAGE_URI="${IMAGE}" ;;
            *)     IMAGE_URI="docker://${IMAGE}" ;;
        esac
        # Import to a tmp file then atomic-rename so a Ctrl-C doesn't leave a
        # half-written sqsh that future runs would happily reuse.
        TMP_SQSH="${IMAGE_SQSH}.partial.$$"
        import_rc=0
        enroot import --output "${TMP_SQSH}" "${IMAGE_URI}" || import_rc=$?
        if [[ "${import_rc}" -eq 0 ]]; then
            mv "${TMP_SQSH}" "${IMAGE_SQSH}"
        else
            rm -f "${TMP_SQSH}"
            echo "Error: enroot import failed (rc=${import_rc}). Falling back to in-srun import for this run." >&2
            IMAGE_SQSH=""
        fi
    else
        echo "Warning: 'enroot' not on PATH; pyxis will re-import the image inside each srun (slow)." >&2
        IMAGE_SQSH=""
    fi
fi

CONTAINER_IMAGE_ARG="${IMAGE_SQSH:-${IMAGE}}"
echo "=== Using container image: ${CONTAINER_IMAGE_ARG} ==="

SRUN_ARGS=(
    --account="${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --nodes="${SLURM_NODES}"
    --gpus-per-node="${SLURM_GPUS}"
    --cpus-per-gpu="${SLURM_CPUS_PER_GPU}"
    --time="${SLURM_TIME}"
)
if [[ -n "${SLURM_QOS}" ]]; then
    SRUN_ARGS+=(--qos="${SLURM_QOS}")
fi

SRUN_CONTAINER_ARGS=(
    --container-image="${CONTAINER_IMAGE_ARG}"
    --container-mounts="${REPO_ROOT}:/workspace/flashdreams,${UV_CACHE_HOST}:/root/.cache/uv,${HF_CACHE_HOST}:/root/.cache/huggingface,${FLASHDREAMS_CACHE_HOST}:/root/.cache/flashdreams,${TRITON_CACHE_HOST}:/root/.cache/triton"
    --container-workdir=/workspace/flashdreams
    --container-writable
    --container-mount-home
    --container-remap-root
    --export=ALL,HF_HOME=/root/.cache/huggingface,UV_LINK_MODE=copy,TRITON_CACHE_DIR=/root/.cache/triton
)

SRUN_CMD=(
    srun
    "${SRUN_ARGS[@]}"
    "${SRUN_CONTAINER_ARGS[@]}"
    bash -s --
    "${TEST_TARGETS[@]}"
)

echo "=== srun invocation ==="
printf '%q ' "${SRUN_CMD[@]}"
printf "<<'EOF'\n"
cat <<'EOF'
set -euo pipefail

# -- bootstrap: install system deps + uv into the raw CUDA base image --------
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3 python3-dev python3-venv ffmpeg gcc g++ ninja-build \
  libnccl-dev curl git ca-certificates unzip
rm -rf /var/lib/apt/lists/*
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --clear
uv sync --frozen --extra dev

exec bash /workspace/flashdreams/tests/run_tests_local.sh "$@"
EOF
echo "EOF"

rc=0
"${SRUN_CMD[@]}" <<'EOF' || rc=$?
set -euo pipefail

# -- bootstrap: install system deps + uv into the raw CUDA base image --------
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3 python3-dev python3-venv ffmpeg gcc g++ ninja-build \
  libnccl-dev curl git ca-certificates unzip
rm -rf /var/lib/apt/lists/*
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# UV_PROJECT_ENVIRONMENT is exported via srun --export so the venv lives
# outside the bind-mounted workspace, avoiding root-owned .venv on the host.
uv venv --clear
uv sync --frozen --extra dev

exec bash /workspace/flashdreams/tests/run_tests_local.sh "$@"
EOF

if [[ ${rc} -ne 0 ]]; then
    slurm_diagnostic_help
    exit "${rc}"
fi
