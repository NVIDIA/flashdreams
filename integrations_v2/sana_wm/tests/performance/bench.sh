#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stack-matched SANA-WM benchmark harness. By default this runs both sibling
# SANA-WM variants. Set SANA_WM_VARIANT=bidirectional or
# SANA_WM_VARIANT=streaming to run only one comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SANA_TEST_DIR}/common.sh"
REPO_ROOT="${SANA_TEST_REPO_ROOT}"

_abspath() {
    sana_test_abspath "$1"
}

_is_true() {
    sana_test_is_true "$1"
}

_write_command() {
    local path="$1"
    shift
    mkdir -p "$(dirname "${path}")"
    printf "%q " "$@" > "${path}"
    printf "\n" >> "${path}"
}

SANA_REPO="$(_abspath "${SANA_REPO:-${SANA_TEST_DIR}/Sana}")"
PATCH_FILE="${SANA_TEST_DIR}/changes_bidirectional.patch"
STREAMING_PATCH_FILE="${SANA_TEST_DIR}/changes_streaming.patch"

SANA_WM_VARIANT="${SANA_WM_VARIANT:-both}"
BENCH_SIDE="${BENCH_SIDE:-both}"
case "${SANA_WM_VARIANT}" in
    both|bidirectional|streaming) ;;
    *)
        echo "[bench] ERROR: SANA_WM_VARIANT must be both, bidirectional, or streaming; got ${SANA_WM_VARIANT}" >&2
        exit 1
        ;;
esac
case "${BENCH_SIDE}" in
    upstream|flashdreams|both) ;;
    *)
        echo "[bench] ERROR: BENCH_SIDE must be upstream, flashdreams, or both; got ${BENCH_SIDE}" >&2
        exit 1
        ;;
esac

DEVICE_LABEL="${DEVICE_LABEL:-GPU}"
STAGE1_PRECISION="${STAGE1_PRECISION:-bf16}"
REFINER_PRECISION="${REFINER_PRECISION:-${STAGE1_PRECISION}}"
QUANT_BACKEND="${QUANT_BACKEND:-auto}"
CHART_LABEL="${CHART_LABEL:-${DEVICE_LABEL}}"
BENCH_PRECISIONS="${BENCH_PRECISIONS-bf16,fp8,fp4}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"
BIDIRECTIONAL_WARMUP_GENERATIONS="${BIDIRECTIONAL_WARMUP_GENERATIONS:-${WARMUP_RUNS}}"
BIDIRECTIONAL_MEASURED_GENERATIONS="${BIDIRECTIONAL_MEASURED_GENERATIONS:-${MEASURED_RUNS}}"
BENCH_DRY_RUN="${BENCH_DRY_RUN:-0}"

if [[ "${SANA_WM_VARIANT}" == "both" ]]; then
    BIDIRECTIONAL_OUTPUT_ENV=()
    STREAMING_OUTPUT_ENV=()
    BIDIRECTIONAL_GENERATION_ENV=(
        BIDIRECTIONAL_WARMUP_GENERATIONS="${BIDIRECTIONAL_WARMUP_GENERATIONS}"
        BIDIRECTIONAL_MEASURED_GENERATIONS="${BIDIRECTIONAL_MEASURED_GENERATIONS}"
    )
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        OUTPUT_BASE="$(_abspath "${OUTPUT_DIR}")"
        BIDIRECTIONAL_OUTPUT_ENV=(OUTPUT_DIR="${OUTPUT_BASE}/bidirectional")
        if [[ -n "${BENCH_PRECISIONS}" ]]; then
            STREAMING_OUTPUT_ENV=(OUTPUT_DIR="${OUTPUT_BASE}/streaming")
        else
            STREAMING_OUTPUT_ENV=(OUTPUT_DIR="${OUTPUT_BASE}/streaming/${STAGE1_PRECISION}")
        fi
    fi

    STREAMING_PRECISION_ENV=(BENCH_PRECISIONS=)
    if [[ -n "${BENCH_PRECISIONS}" ]]; then
        STREAMING_PRECISION_ENV=(BENCH_PRECISIONS="${BENCH_PRECISIONS}")
    fi

    echo "[bench] running SANA-WM_bidirectional benchmark"
    env \
        SANA_WM_VARIANT=bidirectional \
        BENCH_PRECISIONS="${BENCH_PRECISIONS}" \
        STAGE1_PRECISION=bf16 \
        REFINER_PRECISION=bf16 \
        "${BIDIRECTIONAL_GENERATION_ENV[@]}" \
        "${BIDIRECTIONAL_OUTPUT_ENV[@]}" \
        bash "${SCRIPT_DIR}/bench.sh"

    echo "[bench] running SANA-WM_streaming benchmark"
    env \
        SANA_WM_VARIANT=streaming \
        "${STREAMING_PRECISION_ENV[@]}" \
        "${STREAMING_OUTPUT_ENV[@]}" \
        bash "${SCRIPT_DIR}/bench.sh"

    echo "[bench] all requested SANA-WM benchmarks done."
    exit 0
fi

_reject_bidirectional_low_precision() {
    echo "[bench] ERROR: upstream SANA-WM_bidirectional benchmarks are BF16-only." >&2
    echo "        Use BENCH_SIDE=flashdreams for FlashDreams-only SANA-WM_bidirectional FP8/FP4 diagnostics." >&2
    echo "        For upstream comparisons, use STAGE1_PRECISION=bf16 REFINER_PRECISION=bf16, or set SANA_WM_VARIANT=streaming." >&2
    exit 1
}

_validate_precision() {
    local name="$1"
    local value="$2"
    case "${value}" in
        bf16|fp8|fp4) ;;
        *)
            echo "[bench] ERROR: ${name} must be bf16, fp8, or fp4; got ${value}" >&2
            exit 1
            ;;
    esac
}

_is_quant_precision() {
    case "$1" in
        fp8|fp4) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${SANA_WM_VARIANT}" == "streaming" ]]; then
    DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/outputs/bench/streaming/${STAGE1_PRECISION}"
    DEFAULT_SWEEP_OUTPUT_DIR="${SCRIPT_DIR}/outputs/bench/streaming"
    NUM_FRAMES="${NUM_FRAMES:-241}"
    CFG_SCALE="${CFG_SCALE:-1.0}"
    NO_REFINER="${NO_REFINER:-0}"
else
    DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/outputs/bench"
    DEFAULT_SWEEP_OUTPUT_DIR="${SCRIPT_DIR}/outputs/bench"
    NUM_FRAMES="${NUM_FRAMES:-121}"
    CFG_SCALE="${CFG_SCALE:-5.0}"
    NO_REFINER="${NO_REFINER:-0}"
fi
if [[ -n "${BENCH_PRECISIONS}" ]]; then
    OUTPUT_DIR="$(_abspath "${OUTPUT_DIR:-${DEFAULT_SWEEP_OUTPUT_DIR}}")"
else
    OUTPUT_DIR="$(_abspath "${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}")"
fi
UPSTREAM_ROOT="${OUTPUT_DIR}/upstream"
FLASHDREAMS_ROOT="${OUTPUT_DIR}/flashdreams"

IMAGE_PATH="$(_abspath "${IMAGE_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.png}")"
PROMPT_PATH="$(_abspath "${PROMPT_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.txt}")"
CAMERA_PATH="$(_abspath "${CAMERA_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_pose.npy}")"
INTRINSICS_PATH="$(_abspath "${INTRINSICS_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_intrinsics.npy}")"
STREAMING_ACTION="${STREAMING_ACTION-w-80,dw-40,w-80,aw-40}"
FPS="${FPS:-16}"
STEP="${STEP:-60}"
SEED="${SEED:-42}"
FORCE_CUDNN_SDPA="${FORCE_CUDNN_SDPA:-0}"
COMPILE_STAGE1="${COMPILE_STAGE1:-0}"

STREAMING_OUTPUT_MODE="${STREAMING_OUTPUT_MODE:-mp4}"
STREAMING_NO_MP4="${STREAMING_NO_MP4:-0}"
STREAMING_PROFILE_CUDA="${STREAMING_PROFILE_CUDA:-0}"
STREAMING_NO_COMPILE="${STREAMING_NO_COMPILE:-0}"
STREAMING_DENOISING_STEP_LIST="${STREAMING_DENOISING_STEP_LIST:-1000,960,889,727,0}"
STREAMING_NUM_FRAME_PER_BLOCK="${STREAMING_NUM_FRAME_PER_BLOCK:-3}"
STREAMING_REFINER_BLOCK_SIZE="${STREAMING_REFINER_BLOCK_SIZE:-3}"
STREAMING_REFINER_KV_MAX_FRAMES="${STREAMING_REFINER_KV_MAX_FRAMES:-11}"
STREAMING_REFINER_SEED="${STREAMING_REFINER_SEED:-${SEED}}"
STREAMING_SINK_SIZE="${STREAMING_SINK_SIZE:-1}"
STREAMING_NUM_CACHED_BLOCKS="${STREAMING_NUM_CACHED_BLOCKS:-2}"
STREAMING_CRF="${STREAMING_CRF:-18}"
STREAMING_PRESET="${STREAMING_PRESET:-medium}"
STREAMING_ENCODER="${STREAMING_ENCODER:-${SANA_WM_STREAMING_MP4_ENCODER:-libx264}}"

_validate_precision "STAGE1_PRECISION" "${STAGE1_PRECISION}"
_validate_precision "REFINER_PRECISION" "${REFINER_PRECISION}"

if [[ "${SANA_WM_VARIANT}" == "streaming" ]] && _is_true "${NO_REFINER}"; then
    echo "[bench] ERROR: SANA_WM_VARIANT=streaming benchmarks the full upstream streaming stack; NO_REFINER=1 is not supported." >&2
    exit 1
fi
if [[ "${SANA_WM_VARIANT}" == "streaming" ]] && _is_true "${COMPILE_STAGE1}"; then
    echo "[bench] ERROR: COMPILE_STAGE1=1 is bidirectional-only. Streaming compile parity is controlled by STREAMING_NO_COMPILE." >&2
    exit 1
fi
if [[ "${SANA_WM_VARIANT}" == "bidirectional" ]]; then
    if [[ "${BENCH_SIDE}" != "flashdreams" ]] && [[ "${STAGE1_PRECISION}" != "bf16" || "${REFINER_PRECISION}" != "bf16" ]]; then
        _reject_bidirectional_low_precision
    fi
    if [[ -n "${BENCH_PRECISIONS}" ]]; then
        IFS=',' read -r -a BIDIRECTIONAL_PRECISION_LIST <<< "${BENCH_PRECISIONS}"
        for PRECISION_RAW in "${BIDIRECTIONAL_PRECISION_LIST[@]}"; do
            PRECISION="${PRECISION_RAW//[[:space:]]/}"
            if [[ -z "${PRECISION}" ]]; then
                continue
            fi
            case "${PRECISION}" in
                bf16|fp8|fp4) ;;
                *)
                    echo "[bench] ERROR: unsupported BENCH_PRECISIONS entry: ${PRECISION}" >&2
                    exit 1
                    ;;
            esac
        done
    fi
fi

LOW_PRECISION=0
if _is_quant_precision "${STAGE1_PRECISION}" || _is_quant_precision "${REFINER_PRECISION}"; then
    LOW_PRECISION=1
fi
UPSTREAM_QUANT=0
if [[ "${LOW_PRECISION}" == "1" && "${SANA_WM_VARIANT}" == "streaming" && "${BENCH_SIDE}" != "flashdreams" ]]; then
    UPSTREAM_QUANT=1
fi

if _is_true "${BENCH_DRY_RUN}"; then
    echo "[bench] dry run"
    echo "        variant: ${SANA_WM_VARIANT}"
    echo "        side: ${BENCH_SIDE}"
    echo "        output: ${OUTPUT_DIR}"
    echo "        precision: stage1=${STAGE1_PRECISION} refiner=${REFINER_PRECISION}"
    if [[ "${UPSTREAM_QUANT}" == "1" ]]; then
        echo "        upstream quant extra: yes"
    else
        echo "        upstream quant extra: no"
    fi
    if [[ "${SANA_WM_VARIANT}" == "bidirectional" ]]; then
        echo "        bidirectional generations: warmup=${BIDIRECTIONAL_WARMUP_GENERATIONS} measured=${BIDIRECTIONAL_MEASURED_GENERATIONS}"
        if [[ -n "${BENCH_PRECISIONS}" ]]; then
            echo "        precision sweep: ${BENCH_PRECISIONS} (non-BF16 rows are FlashDreams-only)"
        fi
    elif [[ -n "${BENCH_PRECISIONS}" ]]; then
        echo "        precision sweep: ${BENCH_PRECISIONS}"
    else
        echo "        runs: warmup=${WARMUP_RUNS} measured=${MEASURED_RUNS}"
    fi
    exit 0
fi

if [[ -n "${BENCH_PRECISIONS}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    SWEEP_ITEMS=()
    IFS=',' read -r -a PRECISION_LIST <<< "${BENCH_PRECISIONS}"
    for PRECISION_RAW in "${PRECISION_LIST[@]}"; do
        PRECISION="${PRECISION_RAW//[[:space:]]/}"
        if [[ -z "${PRECISION}" ]]; then
            continue
        fi
        case "${PRECISION}" in
            bf16|fp8|fp4) ;;
            *)
                echo "[bench] ERROR: unsupported BENCH_PRECISIONS entry: ${PRECISION}" >&2
                exit 1
                ;;
        esac
        PRECISION_LABEL="${PRECISION^^}"
        PRECISION_OUTPUT_DIR="${OUTPUT_DIR}/${PRECISION}"
        ROW_BENCH_SIDE="${BENCH_SIDE}"
        if [[ "${SANA_WM_VARIANT}" == "bidirectional" && "${PRECISION}" != "bf16" ]]; then
            if [[ "${BENCH_SIDE}" == "upstream" ]]; then
                echo "[bench] skipping upstream SANA-WM_bidirectional ${PRECISION_LABEL}; upstream is BF16-only"
                continue
            fi
            ROW_BENCH_SIDE="flashdreams"
        fi
        echo "[bench] precision sweep row ${PRECISION_LABEL} -> ${PRECISION_OUTPUT_DIR}"
        BENCH_PRECISIONS="" \
            SANA_REPO="${SANA_REPO}" \
            SANA_WM_VARIANT="${SANA_WM_VARIANT}" \
            BENCH_SIDE="${ROW_BENCH_SIDE}" \
            OUTPUT_DIR="${PRECISION_OUTPUT_DIR}" \
            STAGE1_PRECISION="${PRECISION}" \
            REFINER_PRECISION="${PRECISION}" \
            QUANT_BACKEND="${QUANT_BACKEND}" \
            DEVICE_LABEL="${DEVICE_LABEL}" \
            CHART_LABEL="${CHART_LABEL}" \
            bash "${SCRIPT_DIR}/bench.sh"
        SWEEP_ITEMS+=(--item "${PRECISION_LABEL}:${PRECISION_OUTPUT_DIR}/bench.json")
    done
    if [[ "${#SWEEP_ITEMS[@]}" -eq 0 ]]; then
        echo "[bench] ERROR: BENCH_PRECISIONS produced no precision rows." >&2
        exit 1
    fi
    echo "[bench] aggregating precision sweep -> ${OUTPUT_DIR}/bench.md"
    ( cd "${SANA_TEST_DIR}" && \
        uv run python "${SCRIPT_DIR}/bench_sweep_summary.py" \
            "${SWEEP_ITEMS[@]}" \
            --output-json "${OUTPUT_DIR}/bench.json" \
            --output-md "${OUTPUT_DIR}/bench.md" \
            --output-chart-md "${OUTPUT_DIR}/perf.md" )
    echo "[bench] done."
    echo "        precision summary: ${OUTPUT_DIR}/bench.md"
    echo "        precision data: ${OUTPUT_DIR}/perf.md"
    exit 0
fi

sana_test_prepare_sana_repo "${SANA_REPO}" "${PATCH_FILE}"
if [[ "${SANA_WM_VARIANT}" == "streaming" ]]; then
    if [[ ! -f "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm_streaming.py" ]]; then
        echo "[setup] ERROR: pinned Sana checkout has no SANA-WM streaming entrypoint." >&2
        echo "        Expected: inference_video_scripts/wm/inference_sana_wm_streaming.py" >&2
        exit 1
    fi
    sana_test_apply_patch_once "${STREAMING_PATCH_FILE}"
fi
UPSTREAM_COMMIT="$(git rev-parse HEAD)"
FLASHDREAMS_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf "unknown")"

UV_SYNC_ARGS=(uv sync)
UV_SYNC_ENV=()
NVTE_ARCH_MARKER="${SANA_TEST_DIR}/.venv/.flashdreams_nvte_cuda_archs"
NVTE_MARK_ARCH=0
if [[ "${UPSTREAM_QUANT}" == "1" ]]; then
    UV_SYNC_ARGS+=(--extra quant)
    export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
    export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
    NVTE_MARK_ARCH=1
    NVTE_ARCH_SETTING="${NVTE_CUDA_ARCHS:-default}"
    NVTE_INSTALLED=0
    if [[ -x "${SANA_TEST_DIR}/.venv/bin/python" ]]; then
        if ( cd "${SANA_TEST_DIR}" && .venv/bin/python - <<'PY' )
import importlib.util

raise SystemExit(0 if importlib.util.find_spec("transformer_engine") else 1)
PY
        then
            NVTE_INSTALLED=1
        fi
    fi
    if [[ "${NVTE_INSTALLED}" != "1" ]] || [[ ! -f "${NVTE_ARCH_MARKER}" ]] || [[ "$(cat "${NVTE_ARCH_MARKER}")" != "${NVTE_ARCH_SETTING}" ]]; then
        echo "[setup] refreshing TransformerEngine build for CUDA archs ${NVTE_ARCH_SETTING}"
        UV_SYNC_ARGS+=(--reinstall-package transformer-engine)
        UV_SYNC_ENV=(env UV_NO_CACHE=1)
    fi
    echo "[setup] TransformerEngine build env: NVTE_FRAMEWORK=${NVTE_FRAMEWORK} NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-default} NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP}"
    echo "[setup] seeding TransformerEngine build tools in isolated venv"
    ( cd "${SANA_TEST_DIR}" && \
        uv venv --allow-existing --python "3.12" .venv >/dev/null && \
        uv pip install --python .venv/bin/python setuptools wheel pybind11 )
elif [[ "${LOW_PRECISION}" == "1" ]]; then
    echo "[setup] skipping TransformerEngine quant extra; low-precision rows are FlashDreams-only."
fi
echo "[setup] ensuring Python deps via ${UV_SYNC_ARGS[*]} (isolated venv)"
( cd "${SANA_TEST_DIR}" && "${UV_SYNC_ENV[@]}" "${UV_SYNC_ARGS[@]}" )
if [[ "${NVTE_MARK_ARCH}" == "1" ]]; then
    printf "%s\n" "${NVTE_ARCH_SETTING}" > "${NVTE_ARCH_MARKER}"
fi
UPSTREAM_PYTHONPATH="$(sana_test_upstream_pythonpath "${SANA_REPO}")"

if [[ "${BENCH_SIDE}" != "flashdreams" ]]; then
    mkdir -p "${UPSTREAM_ROOT}"
fi
if [[ "${BENCH_SIDE}" != "upstream" ]]; then
    mkdir -p "${FLASHDREAMS_ROOT}"
fi

# Remove stale run_* dirs from prior (possibly longer) benchmarks. bench_summary
# reads every run_* dir, so leftovers from an earlier run with more MEASURED_RUNS
# would silently pollute this invocation's aggregate.
if [[ "${BENCH_SIDE}" != "flashdreams" ]]; then
    rm -rf "${UPSTREAM_ROOT:?}"/run_*
fi
if [[ "${BENCH_SIDE}" != "upstream" ]]; then
    rm -rf "${FLASHDREAMS_ROOT:?}"/run_*
fi

UPSTREAM_REFINER_ARGS=()
FLASHDREAMS_REFINER_ARGS=()
if _is_true "${NO_REFINER}"; then
    UPSTREAM_REFINER_ARGS+=(--no_refiner)
    FLASHDREAMS_REFINER_ARGS+=(--no-refiner)
fi
UPSTREAM_BACKEND_ARGS=()
FLASHDREAMS_BACKEND_ARGS=()
if _is_true "${FORCE_CUDNN_SDPA}"; then
    if [[ "${SANA_WM_VARIANT}" != "streaming" ]]; then
        UPSTREAM_BACKEND_ARGS+=(--force_cudnn_sdpa)
    fi
    FLASHDREAMS_BACKEND_ARGS+=(--force-cudnn-sdpa)
fi
UPSTREAM_COMPILE_ARGS=()
FLASHDREAMS_COMPILE_ARGS=()
if _is_true "${COMPILE_STAGE1}"; then
    UPSTREAM_COMPILE_ARGS+=(--compile_stage1)
    FLASHDREAMS_COMPILE_ARGS+=(--compile-stage1)
fi
UPSTREAM_PRECISION_ARGS=(
    --stage1_precision "${STAGE1_PRECISION}"
    --refiner_precision "${REFINER_PRECISION}"
)
FLASHDREAMS_PRECISION_ARGS=(
    --stage1-precision "${STAGE1_PRECISION}"
    --refiner-precision "${REFINER_PRECISION}"
    --quant-backend "${QUANT_BACKEND}"
)

STREAMING_CAMERA_SOURCE="camera"
UPSTREAM_STREAMING_CAMERA_ARGS=(--camera "${CAMERA_PATH}")
FLASHDREAMS_STREAMING_CAMERA_ARGS=(--camera-source camera --camera-path "${CAMERA_PATH}")
if [[ -n "${STREAMING_ACTION}" ]]; then
    STREAMING_CAMERA_SOURCE="action"
    UPSTREAM_STREAMING_CAMERA_ARGS=(--action "${STREAMING_ACTION}")
    FLASHDREAMS_STREAMING_CAMERA_ARGS=(--camera-source action --action "${STREAMING_ACTION}" --camera-path "${CAMERA_PATH}")
fi
UPSTREAM_STREAMING_MODE_ARGS=(--output_mode "${STREAMING_OUTPUT_MODE}")
FLASHDREAMS_STREAMING_MODE_ARGS=(--output-mode "${STREAMING_OUTPUT_MODE}")
if _is_true "${STREAMING_NO_MP4}"; then
    UPSTREAM_STREAMING_MODE_ARGS+=(--no_mp4)
fi
if _is_true "${STREAMING_PROFILE_CUDA}"; then
    UPSTREAM_STREAMING_MODE_ARGS+=(--profile_cuda)
fi
if _is_true "${STREAMING_NO_COMPILE}"; then
    UPSTREAM_STREAMING_MODE_ARGS+=(--no_compile)
fi
FLASHDREAMS_STREAMING_COMPILE_ARGS=()
if [[ "${SANA_WM_VARIANT}" == "streaming" ]] && ! _is_true "${STREAMING_NO_COMPILE}"; then
    FLASHDREAMS_STREAMING_COMPILE_ARGS+=(--compile-streaming-refiner)
fi

if [[ "${SANA_WM_VARIANT}" == "bidirectional" ]]; then
    TOTAL_RUNS=1
else
    TOTAL_RUNS=$(( WARMUP_RUNS + MEASURED_RUNS ))
fi

_run_upstream_once() {
    local i="$1"
    local upstream_out="${UPSTREAM_ROOT}/run_${i}"
    local -a upstream_cmd

    mkdir -p "${upstream_out}"

    if [[ "${SANA_WM_VARIANT}" == "streaming" ]]; then
        upstream_cmd=(
            env "PYTHONPATH=${UPSTREAM_PYTHONPATH}"
            uv run python "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm_streaming.py"
            --image "${IMAGE_PATH}"
            --prompt "${PROMPT_PATH}"
            "${UPSTREAM_STREAMING_CAMERA_ARGS[@]}"
            --intrinsics "${INTRINSICS_PATH}"
            --output_dir "${upstream_out}"
            --name upstream
            --num_frames "${NUM_FRAMES}"
            --fps "${FPS}"
            --cfg_scale "${CFG_SCALE}"
            --seed "${SEED}"
            --benchmark_json "${upstream_out}/stats.json"
            --denoising_step_list "${STREAMING_DENOISING_STEP_LIST}"
            --num_frame_per_block "${STREAMING_NUM_FRAME_PER_BLOCK}"
            --refiner_block_size "${STREAMING_REFINER_BLOCK_SIZE}"
            --refiner_kv_max_frames "${STREAMING_REFINER_KV_MAX_FRAMES}"
            --refiner_seed "${STREAMING_REFINER_SEED}"
            --sink_size "${STREAMING_SINK_SIZE}"
            --num_cached_blocks "${STREAMING_NUM_CACHED_BLOCKS}"
            --streaming_crf "${STREAMING_CRF}"
            --streaming_preset "${STREAMING_PRESET}"
            --streaming_encoder "${STREAMING_ENCODER}"
            "${UPSTREAM_PRECISION_ARGS[@]}"
            "${UPSTREAM_STREAMING_MODE_ARGS[@]}"
        )
    else
        upstream_cmd=(
            env "PYTHONPATH=${UPSTREAM_PYTHONPATH}"
            uv run python "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm.py"
            --image "${IMAGE_PATH}"
            --prompt "${PROMPT_PATH}"
            --camera "${CAMERA_PATH}"
            --intrinsics "${INTRINSICS_PATH}"
            --output_dir "${upstream_out}"
            --name upstream
            --num_frames "${NUM_FRAMES}"
            --fps "${FPS}"
            --step "${STEP}"
            --cfg_scale "${CFG_SCALE}"
            --seed "${SEED}"
            --no_action_overlay
            --stats_json "${upstream_out}/stats.json"
            --warmup_generations "${BIDIRECTIONAL_WARMUP_GENERATIONS}"
            --measured_generations "${BIDIRECTIONAL_MEASURED_GENERATIONS}"
            "${UPSTREAM_PRECISION_ARGS[@]}"
            "${UPSTREAM_BACKEND_ARGS[@]}"
            "${UPSTREAM_COMPILE_ARGS[@]}"
            "${UPSTREAM_REFINER_ARGS[@]}"
        )
    fi

    echo "[bench] upstream ${SANA_WM_VARIANT} run ${i}/${TOTAL_RUNS}"
    _write_command "${upstream_out}/command.txt" "${upstream_cmd[@]}"
    ( cd "${SANA_TEST_DIR}" && "${upstream_cmd[@]}" )
}

_run_flashdreams_once() {
    local i="$1"
    local flashdreams_out="${FLASHDREAMS_ROOT}/run_${i}"
    local -a flashdreams_cmd

    mkdir -p "${flashdreams_out}"

    if [[ "${SANA_WM_VARIANT}" == "streaming" ]]; then
        flashdreams_cmd=(
            uv run python "${SANA_TEST_DIR}/run_flashdreams_streaming.py"
            --image-path "${IMAGE_PATH}"
            --prompt-path "${PROMPT_PATH}"
            "${FLASHDREAMS_STREAMING_CAMERA_ARGS[@]}"
            --intrinsics-path "${INTRINSICS_PATH}"
            --output-dir "${flashdreams_out}"
            --name flashdreams
            --num-frames "${NUM_FRAMES}"
            --fps "${FPS}"
            --cfg-scale "${CFG_SCALE}"
            --flow-shift "8.0"
            --seed "${SEED}"
            --refiner-seed "${STREAMING_REFINER_SEED}"
            --stats-json "${flashdreams_out}/stats.json"
            --denoising-step-list "${STREAMING_DENOISING_STEP_LIST}"
            --num-frame-per-block "${STREAMING_NUM_FRAME_PER_BLOCK}"
            --num-cached-blocks "${STREAMING_NUM_CACHED_BLOCKS}"
            --sink-size "${STREAMING_SINK_SIZE}"
            --refiner-block-size "${STREAMING_REFINER_BLOCK_SIZE}"
            --refiner-kv-max-frames "${STREAMING_REFINER_KV_MAX_FRAMES}"
            "${FLASHDREAMS_STREAMING_MODE_ARGS[@]}"
            "${FLASHDREAMS_STREAMING_COMPILE_ARGS[@]}"
            "${FLASHDREAMS_PRECISION_ARGS[@]}"
            "${FLASHDREAMS_BACKEND_ARGS[@]}"
        )
    else
        flashdreams_cmd=(
            uv run python "${SANA_TEST_DIR}/run_flashdreams_bidirectional.py"
            --image-path "${IMAGE_PATH}"
            --prompt-path "${PROMPT_PATH}"
            --camera-path "${CAMERA_PATH}"
            --intrinsics-path "${INTRINSICS_PATH}"
            --output-dir "${flashdreams_out}"
            --name flashdreams
            --num-frames "${NUM_FRAMES}"
            --fps "${FPS}"
            --step "${STEP}"
            --cfg-scale "${CFG_SCALE}"
            --seed "${SEED}"
            --stats-json "${flashdreams_out}/stats.json"
            --warmup-generations "${BIDIRECTIONAL_WARMUP_GENERATIONS}"
            --measured-generations "${BIDIRECTIONAL_MEASURED_GENERATIONS}"
            "${FLASHDREAMS_PRECISION_ARGS[@]}"
            "${FLASHDREAMS_BACKEND_ARGS[@]}"
            "${FLASHDREAMS_COMPILE_ARGS[@]}"
            "${FLASHDREAMS_REFINER_ARGS[@]}"
        )
    fi

    echo "[bench] FlashDreams ${SANA_WM_VARIANT} run ${i}/${TOTAL_RUNS}"
    _write_command "${flashdreams_out}/command.txt" "${flashdreams_cmd[@]}"
    ( cd "${SANA_TEST_DIR}" && "${flashdreams_cmd[@]}" )
}

if [[ "${BENCH_SIDE}" != "flashdreams" ]]; then
    for ((i = 0; i < TOTAL_RUNS; i++)); do
        _run_upstream_once "${i}"
    done
fi

if [[ "${BENCH_SIDE}" != "upstream" ]]; then
    for ((i = 0; i < TOTAL_RUNS; i++)); do
        _run_flashdreams_once "${i}"
    done
fi

SUMMARY_JSON="${OUTPUT_DIR}/bench.json"
SUMMARY_MD="${OUTPUT_DIR}/bench.md"
SUMMARY_CHART_MD="${OUTPUT_DIR}/perf.md"
SUMMARY_FLAGS=()
if _is_true "${NO_REFINER}"; then
    SUMMARY_FLAGS+=(--no-refiner)
fi
if _is_true "${COMPILE_STAGE1}"; then
    SUMMARY_FLAGS+=(--compile-stage1)
fi
if _is_true "${FORCE_CUDNN_SDPA}"; then
    SUMMARY_FLAGS+=(--force-cudnn-sdpa)
fi
SUMMARY_WARMUP_RUNS="${WARMUP_RUNS}"
if [[ "${SANA_WM_VARIANT}" == "bidirectional" ]]; then
    SUMMARY_WARMUP_RUNS="${BIDIRECTIONAL_WARMUP_GENERATIONS}"
fi
SUMMARY_ARGS=(
    uv run python "${SCRIPT_DIR}/bench_summary.py"
    --variant "${SANA_WM_VARIANT}"
    --bench-side "${BENCH_SIDE}"
    --warmup-runs "${SUMMARY_WARMUP_RUNS}"
    --image-path "${IMAGE_PATH}"
    --prompt-path "${PROMPT_PATH}"
    --camera-path "${CAMERA_PATH}"
    --camera-source "${STREAMING_CAMERA_SOURCE}"
    --intrinsics-path "${INTRINSICS_PATH}"
    --num-frames "${NUM_FRAMES}"
    --seed "${SEED}"
    --device-label "${DEVICE_LABEL}"
    --chart-label "${CHART_LABEL}"
    --stage1-precision "${STAGE1_PRECISION}"
    --refiner-precision "${REFINER_PRECISION}"
    --quant-backend "${QUANT_BACKEND}"
    --upstream-commit "${UPSTREAM_COMMIT}"
    --flashdreams-commit "${FLASHDREAMS_COMMIT}"
    --output-json "${SUMMARY_JSON}"
    --output-md "${SUMMARY_MD}"
    "${SUMMARY_FLAGS[@]}"
)
if [[ "${BENCH_SIDE}" != "flashdreams" ]]; then
    SUMMARY_ARGS+=(--upstream-dir "${UPSTREAM_ROOT}")
fi
if [[ "${SANA_WM_VARIANT}" == "streaming" ]]; then
    SUMMARY_ARGS+=(
        --action "${STREAMING_ACTION}"
        --output-mode "${STREAMING_OUTPUT_MODE}"
        --denoising-step-list "${STREAMING_DENOISING_STEP_LIST}"
        --num-frame-per-block "${STREAMING_NUM_FRAME_PER_BLOCK}"
        --refiner-block-size "${STREAMING_REFINER_BLOCK_SIZE}"
        --refiner-kv-max-frames "${STREAMING_REFINER_KV_MAX_FRAMES}"
    )
fi
if [[ "${BENCH_SIDE}" != "upstream" ]]; then
    SUMMARY_ARGS+=(--flashdreams-dir "${FLASHDREAMS_ROOT}" --output-chart-md "${SUMMARY_CHART_MD}")
fi

echo "[bench] summarising -> ${SUMMARY_MD}"
( cd "${SANA_TEST_DIR}" && "${SUMMARY_ARGS[@]}" )

echo "[bench] done."
echo "        summary: ${SUMMARY_MD}"
if [[ -f "${SUMMARY_CHART_MD}" ]]; then
    echo "        chart data: ${SUMMARY_CHART_MD}"
fi
