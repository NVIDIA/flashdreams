#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Benchmark a FlashDreams runner with multi-GPU FlashVSR post-processing.

Usage:
  integrations/flashvsr/scripts/benchmark_postprocess_multigpu.sh [runner arguments...]

Environment overrides:
  NPROC_PER_NODE=8
  RUNNER_SLUG=wan21-t2v-1.3b-480p
  POSTPROCESS_PRESET=flashvsr-v1.1-full-attn
  WARMUP_RUNS=1
  BENCHMARK_RUNS=1
  BENCHMARK_ROOT=<persistent result directory>
  COMPILE_CACHE_ROOT=<Torch compile cache directory; defaults to node-local /tmp>
  TORCHINDUCTOR_COMPILE_THREADS=1
  GPU_SAMPLE_INTERVAL_SECONDS=1
  MIN_GPU_MEMORY_MIB=1024
  POST_OUTPUT_EXIT_GRACE_SECONDS=0

Every argument is forwarded to flashdreams-run. For example:
  .../benchmark_postprocess_multigpu.sh --prompt "A cat surfing."

Set POST_OUTPUT_EXIT_GRACE_SECONDS to a positive value only to enable the
legacy post-output termination watchdog while diagnosing teardown failures.

The first launch warms the model and per-rank Torch compile caches. Measured
launches reuse the same Inductor and Triton artifacts. The default cache is
node-local to avoid distributed-filesystem metadata stalls; set
COMPILE_CACHE_ROOT explicitly if persistence across allocations is required.
Each launch records rank
markers, torchrun/NCCL logs, GPU-utilization samples, outputs, and wall time.
EOF
}

script_path=$(readlink -f "${BASH_SOURCE[0]}")

find_venv_binary() {
    local name=$1
    local candidate
    for candidate in \
        "${UV_PROJECT_ENVIRONMENT:-}/bin/${name}" \
        "${VIRTUAL_ENV:-}/bin/${name}" \
        "${PWD}/.venv/bin/${name}"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    command -v "${name}"
}

rank_entry() {
    shift
    local run_dir=${1:?rank entry requires a run directory}
    shift

    local rank=${RANK:?torchrun did not set RANK}
    local local_rank=${LOCAL_RANK:?torchrun did not set LOCAL_RANK}
    local world_size=${WORLD_SIZE:?torchrun did not set WORLD_SIZE}
    local python_bin runner_bin runner_pid watchdog_pid completion_marker
    local -a global_args=()

    # ``--no-instantiate`` is a top-level tyro flag and must precede the
    # runner slug. Supporting it here makes launcher/config smoke checks cheap.
    if [[ ${1:-} == "--no-instantiate" ]]; then
        global_args+=("$1")
        shift
    fi

    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_BASE:?}/rank-${local_rank}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_BASE:?}/rank-${local_rank}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_CACHE_BASE:?}/rank-${local_rank}"
    mkdir -p \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${TORCH_EXTENSIONS_DIR}" \
        "${run_dir}/output"

    python_bin=$(find_venv_binary python)
    runner_bin=$(find_venv_binary flashdreams-run)
    completion_marker="${run_dir}/output/stats_${RUNNER_SLUG}.json"
    "${python_bin}" - <<'PY'
import os

import torch

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
props = torch.cuda.get_device_properties(local_rank)
print(
    "FD_RANK_READY "
    f"rank={rank} local_rank={local_rank} world_size={world_size} "
    f"cuda_device={torch.cuda.current_device()} gpu={props.name!r}",
    flush=True,
)
PY

    printf 'FD_RANK_COMMAND rank=%s runner=%q' "${rank}" "${runner_bin}"
    printf ' %q' \
        "${global_args[@]}" \
        "${RUNNER_SLUG:?}" \
        --postprocess.preset "${POSTPROCESS_PRESET:?}" \
        --output-dir "${run_dir}/output" "$@"
    printf '\n'

    set +e
    "${runner_bin}" \
        "${global_args[@]}" \
        "${RUNNER_SLUG}" \
        --postprocess.preset "${POSTPROCESS_PRESET}" \
        --output-dir "${run_dir}/output" "$@" &
    runner_pid=$!
    watchdog_pid=""
    if ((POST_OUTPUT_EXIT_GRACE_SECONDS > 0)); then
        (
            while kill -0 "${runner_pid}" 2>/dev/null; do
                if [[ -s ${completion_marker} ]]; then
                    sleep "${POST_OUTPUT_EXIT_GRACE_SECONDS:?}"
                    if kill -0 "${runner_pid}" 2>/dev/null; then
                        echo "FD_RANK_POST_OUTPUT_TERMINATE rank=${rank} pid=${runner_pid}"
                        kill -TERM "${runner_pid}" 2>/dev/null || true
                    fi
                    exit 0
                fi
                sleep 1
            done
        ) &
        watchdog_pid=$!
    fi
    wait "${runner_pid}"
    local status=$?
    if [[ -n ${watchdog_pid} ]]; then
        kill "${watchdog_pid}" 2>/dev/null || true
        wait "${watchdog_pid}" 2>/dev/null || true
    fi
    if ((status != 0 && POST_OUTPUT_EXIT_GRACE_SECONDS > 0)) \
        && [[ -s ${completion_marker} ]]; then
        echo "FD_RANK_ACCEPT_OUTPUT rank=${rank} runner_status=${status} marker=${completion_marker}"
        status=0
    fi
    set -e
    echo "FD_RANK_EXIT rank=${rank} status=${status}"
    return "${status}"
}

if [[ ${1:-} == "--rank-entry" ]]; then
    rank_entry "$@"
    exit $?
fi

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RUNNER_SLUG=${RUNNER_SLUG:-wan21-t2v-1.3b-480p}
POSTPROCESS_PRESET=${POSTPROCESS_PRESET:-flashvsr-v1.1-full-attn}
WARMUP_RUNS=${WARMUP_RUNS:-1}
BENCHMARK_RUNS=${BENCHMARK_RUNS:-1}
GPU_SAMPLE_INTERVAL_SECONDS=${GPU_SAMPLE_INTERVAL_SECONDS:-1}
MIN_GPU_MEMORY_MIB=${MIN_GPU_MEMORY_MIB:-1024}
POST_OUTPUT_EXIT_GRACE_SECONDS=${POST_OUTPUT_EXIT_GRACE_SECONDS:-0}

for value_name in NPROC_PER_NODE WARMUP_RUNS BENCHMARK_RUNS MIN_GPU_MEMORY_MIB POST_OUTPUT_EXIT_GRACE_SECONDS; do
    value=${!value_name}
    if [[ ! ${value} =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done
if ((NPROC_PER_NODE < 2)); then
    echo "NPROC_PER_NODE must be at least 2 for a multi-GPU benchmark." >&2
    exit 2
fi
if ((BENCHMARK_RUNS < 1)); then
    echo "BENCHMARK_RUNS must be at least 1." >&2
    exit 2
fi
if [[ ${POSTPROCESS_PRESET} != "flashvsr-v1.1-full-attn" ]]; then
    echo "Multi-GPU FlashVSR requires POSTPROCESS_PRESET=flashvsr-v1.1-full-attn." >&2
    exit 2
fi

visible_gpu_count=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
if ((visible_gpu_count < NPROC_PER_NODE)); then
    echo "Requested ${NPROC_PER_NODE} ranks, but only ${visible_gpu_count} GPUs are visible." >&2
    exit 2
fi

cache_parent=${FLASHDREAMS_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/flashdreams}
node_local_cache_parent=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
COMPILE_CACHE_ROOT=${COMPILE_CACHE_ROOT:-${node_local_cache_parent}/flashdreams-postprocess-multigpu-${USER:-user}}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${cache_parent}/benchmarks/postprocess-multigpu/$(date -u +%Y%m%dT%H%M%SZ)}
export TORCHINDUCTOR_CACHE_BASE="${COMPILE_CACHE_ROOT}/torchinductor/world-${NPROC_PER_NODE}"
export TRITON_CACHE_BASE="${COMPILE_CACHE_ROOT}/triton/world-${NPROC_PER_NODE}"
export TORCH_EXTENSIONS_CACHE_BASE="${COMPILE_CACHE_ROOT}/torch-extensions/world-${NPROC_PER_NODE}"
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${COMPILE_CACHE_ROOT}/cuda}
export RUNNER_SLUG POSTPROCESS_PRESET POST_OUTPUT_EXIT_GRACE_SECONDS
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL}
# Avoid creating a large subprocess compile pool per rank. Compilation still
# runs concurrently across ranks, while this bounds CPU oversubscription.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

mkdir -p \
    "${BENCHMARK_ROOT}" \
    "${TORCHINDUCTOR_CACHE_BASE}" \
    "${TRITON_CACHE_BASE}" \
    "${TORCH_EXTENSIONS_CACHE_BASE}" \
    "${CUDA_CACHE_PATH}"

results_tsv="${BENCHMARK_ROOT}/results.tsv"
printf 'phase\trun\telapsed_seconds\tranks\tactive_gpus\tresult_dir\n' >"${results_tsv}"

monitor_gpus() {
    local output=$1
    while true; do
        local timestamp
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
        nvidia-smi \
            --query-gpu=index,uuid,utilization.gpu,memory.used \
            --format=csv,noheader,nounits \
            | awk -v timestamp="${timestamp}" '{print timestamp "," $0}' \
            >>"${output}"
        sleep "${GPU_SAMPLE_INTERVAL_SECONDS}"
    done
}

summarize_gpu_samples() {
    local samples=$1
    local summary=$2
    {
        printf 'gpu_index\tuuid\tmax_util_percent\tmax_memory_mib\n'
        awk -F',' '
            function trim(value) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                return value
            }
            {
                gpu_index = trim($2)
                uuid[gpu_index] = trim($3)
                util = trim($4) + 0
                memory = trim($5) + 0
                if (!(gpu_index in max_util) || util > max_util[gpu_index]) max_util[gpu_index] = util
                if (!(gpu_index in max_memory) || memory > max_memory[gpu_index]) max_memory[gpu_index] = memory
            }
            END {
                for (gpu_index in max_util) {
                    print gpu_index "\t" uuid[gpu_index] "\t" max_util[gpu_index] "\t" max_memory[gpu_index]
                }
            }
        ' "${samples}" | sort -n -k1,1
    } >"${summary}"
}

verify_rank_logs() {
    local torchrun_dir=$1
    local markers=$2
    find "${torchrun_dir}" -type f -name stdout.log -exec grep -h 'FD_RANK_READY' {} + \
        | sort -u >"${markers}" || true
    local rank_count
    rank_count=$(sed -n 's/.* rank=\([0-9][0-9]*\) .*/\1/p' "${markers}" | sort -nu | wc -l)
    if ((rank_count != NPROC_PER_NODE)); then
        echo "Expected ${NPROC_PER_NODE} rank markers, found ${rank_count}:" >&2
        cat "${markers}" >&2
        return 1
    fi
    printf '%s\n' "${rank_count}"
}

run_once() {
    local phase=$1
    local run_number=$2
    shift 2
    local run_dir="${BENCHMARK_ROOT}/${phase}-${run_number}"
    local samples="${run_dir}/gpu_samples.csv"
    local summary="${run_dir}/gpu_summary.tsv"
    local markers="${run_dir}/rank_markers.log"
    local monitor_pid start_ns end_ns elapsed_ms status rank_count active_gpus
    local torchrun_bin

    mkdir -p "${run_dir}/torchrun"
    : >"${samples}"
    monitor_gpus "${samples}" &
    monitor_pid=$!
    start_ns=$(date +%s%N)

    torchrun_bin=$(find_venv_binary torchrun)
    set +e
    "${torchrun_bin}" \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="${NPROC_PER_NODE}" \
        --max-restarts=0 \
        --log-dir="${run_dir}/torchrun" \
        --redirects=3 \
        --tee=3 \
        --no-python \
        "${script_path}" --rank-entry "${run_dir}" "$@"
    status=$?
    set -e

    end_ns=$(date +%s%N)
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    summarize_gpu_samples "${samples}" "${summary}"

    if ((status != 0)); then
        echo "${phase} run ${run_number} failed with status ${status}; logs: ${run_dir}" >&2
        return "${status}"
    fi

    rank_count=$(verify_rank_logs "${run_dir}/torchrun" "${markers}")
    active_gpus=$(awk -v min_memory="${MIN_GPU_MEMORY_MIB}" '
        NR > 1 && $3 > 0 && $4 >= min_memory { count++ }
        END { print count + 0 }
    ' "${summary}")
    if ((active_gpus < NPROC_PER_NODE)); then
        echo "Only ${active_gpus}/${NPROC_PER_NODE} GPUs crossed both utilization and memory thresholds." >&2
        cat "${summary}" >&2
        return 1
    fi

    elapsed_ms=$(((end_ns - start_ns) / 1000000))
    printf '%s\t%s\t%s.%03d\t%s\t%s\t%s\n' \
        "${phase}" "${run_number}" "$((elapsed_ms / 1000))" "$((elapsed_ms % 1000))" \
        "${rank_count}" "${active_gpus}" "${run_dir}" | tee -a "${results_tsv}"
    echo "Verified ${rank_count} ranks and ${active_gpus} active GPUs; details: ${run_dir}"
}

echo "Benchmark root: ${BENCHMARK_ROOT}"
echo "Compile cache: ${COMPILE_CACHE_ROOT}"
echo "Runner: ${RUNNER_SLUG}"
echo "Postprocess preset: ${POSTPROCESS_PRESET}"
echo "Ranks: ${NPROC_PER_NODE}"

for ((run = 1; run <= WARMUP_RUNS; run++)); do
    run_once warmup "${run}" "$@"
done
for ((run = 1; run <= BENCHMARK_RUNS; run++)); do
    run_once benchmark "${run}" "$@"
done

echo "Benchmark complete: ${results_tsv}"
