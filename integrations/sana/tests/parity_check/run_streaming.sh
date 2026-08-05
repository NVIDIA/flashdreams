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

# Patch pinned upstream SANA-WM with streaming instrumentation, run it and the
# FlashDreams streaming integration on the same demo input, dump decoded uint8
# frames, compute frame parity, and report chunk-boundary continuity.

set -euo pipefail

PARITY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_TEST_DIR="$(cd "${PARITY_SCRIPT_DIR}/.." && pwd)"
source "${SANA_TEST_DIR}/common.sh"

SANA_REPO="$(sana_test_abspath "${SANA_REPO:-${SANA_TEST_DIR}/Sana}")"
PATCH_FILE="${SANA_TEST_DIR}/changes_streaming.patch"

OUTPUT_DIR="$(sana_test_abspath "${OUTPUT_DIR:-${PARITY_SCRIPT_DIR}/outputs/parity/streaming}")"
UPSTREAM_OUT="${OUTPUT_DIR}/upstream"
FLASHDREAMS_OUT="${OUTPUT_DIR}/flashdreams"
IMAGE_PATH="$(sana_test_abspath "${IMAGE_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.png}")"
PROMPT_PATH="$(sana_test_abspath "${PROMPT_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.txt}")"
CAMERA_PATH="$(sana_test_abspath "${CAMERA_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_pose.npy}")"
INTRINSICS_PATH="$(sana_test_abspath "${INTRINSICS_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_intrinsics.npy}")"

NUM_FRAMES="${NUM_FRAMES:-241}"
FPS="${FPS:-16}"
CFG_SCALE="${CFG_SCALE:-1.0}"
FLOW_SHIFT="${FLOW_SHIFT:-8.0}"
SEED="${SEED:-42}"
REFINER_SEED="${REFINER_SEED:-${STREAMING_REFINER_SEED:-${SEED}}}"
NO_REFINER="${NO_REFINER:-0}"
FORCE_CUDNN_SDPA="${FORCE_CUDNN_SDPA:-0}"
STREAMING_NO_COMPILE="${STREAMING_NO_COMPILE:-1}"
STAGE1_PRECISION="${STAGE1_PRECISION:-bf16}"
REFINER_PRECISION="${REFINER_PRECISION:-${STAGE1_PRECISION}}"
QUANT_BACKEND="${QUANT_BACKEND:-auto}"
STREAMING_ACTION="${STREAMING_ACTION-w-80,dw-40,w-80,aw-40}"
TRANSLATION_SPEED="${TRANSLATION_SPEED:-0.025}"
ROTATION_SPEED_DEG="${ROTATION_SPEED_DEG:-0.6}"
STREAMING_DENOISING_STEP_LIST="${STREAMING_DENOISING_STEP_LIST:-1000,960,889,727,0}"
STREAMING_NUM_FRAME_PER_BLOCK="${STREAMING_NUM_FRAME_PER_BLOCK:-3}"
STREAMING_NUM_CACHED_BLOCKS="${STREAMING_NUM_CACHED_BLOCKS:-2}"
STREAMING_REFINER_BLOCK_SIZE="${STREAMING_REFINER_BLOCK_SIZE:-3}"
STREAMING_REFINER_KV_MAX_FRAMES="${STREAMING_REFINER_KV_MAX_FRAMES:-11}"
STREAMING_SINK_SIZE="${STREAMING_SINK_SIZE:-1}"
STREAMING_CRF="${STREAMING_CRF:-18}"
STREAMING_PRESET="${STREAMING_PRESET:-medium}"
STREAMING_ENCODER="${STREAMING_ENCODER:-${SANA_WM_STREAMING_MP4_ENCODER:-libx264}}"
STREAMING_OUTPUT_MODE="${STREAMING_OUTPUT_MODE:-mp4}"
STREAMING_SAMPLE_FRAME_STRIDE="${STREAMING_SAMPLE_FRAME_STRIDE:-1}"

if sana_test_is_true "${NO_REFINER}"; then
    echo "[run] ERROR: streaming parity uses the full upstream streaming stack; NO_REFINER=1 is not supported." >&2
    exit 1
fi
if [[ "${STREAMING_OUTPUT_MODE}" != "mp4" && "${STREAMING_OUTPUT_MODE}" != "cpu" && "${STREAMING_OUTPUT_MODE}" != "discard" ]]; then
    echo "[run] ERROR: STREAMING_OUTPUT_MODE must be mp4, cpu, or discard; got ${STREAMING_OUTPUT_MODE}" >&2
    exit 1
fi
if [[ "${STREAMING_SAMPLE_FRAME_STRIDE}" != "1" ]]; then
    echo "[run] ERROR: streaming parity and continuity checks require STREAMING_SAMPLE_FRAME_STRIDE=1." >&2
    exit 1
fi

sana_test_prepare_sana_repo "${SANA_REPO}" "${PATCH_FILE}"
if [[ ! -f "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm_streaming.py" ]]; then
    echo "[run] ERROR: pinned Sana checkout has no SANA-WM streaming entrypoint." >&2
    echo "      Expected: inference_video_scripts/wm/inference_sana_wm_streaming.py" >&2
    exit 1
fi
sana_test_sync_venv
UPSTREAM_PYTHONPATH="$(sana_test_upstream_pythonpath "${SANA_REPO}")"

mkdir -p "${UPSTREAM_OUT}" "${FLASHDREAMS_OUT}"

UPSTREAM_FRAMES="${UPSTREAM_OUT}/frames.npz"
FLASHDREAMS_FRAMES="${FLASHDREAMS_OUT}/frames.npy"
UPSTREAM_STATS="${UPSTREAM_OUT}/stats.json"
FLASHDREAMS_STATS="${FLASHDREAMS_OUT}/stats.json"

UPSTREAM_PRECISION_ARGS=(
    --stage1_precision "${STAGE1_PRECISION}"
    --refiner_precision "${REFINER_PRECISION}"
)
FLASHDREAMS_PRECISION_ARGS=(
    --stage1-precision "${STAGE1_PRECISION}"
    --refiner-precision "${REFINER_PRECISION}"
    --quant-backend "${QUANT_BACKEND}"
)
UPSTREAM_CAMERA_ARGS=(--camera "${CAMERA_PATH}")
FLASHDREAMS_CAMERA_ARGS=(--camera-source camera --camera-path "${CAMERA_PATH}")
if [[ -n "${STREAMING_ACTION}" ]]; then
    UPSTREAM_CAMERA_ARGS=(--action "${STREAMING_ACTION}")
    FLASHDREAMS_CAMERA_ARGS=(--camera-source action --action "${STREAMING_ACTION}" --camera-path "${CAMERA_PATH}")
fi
UPSTREAM_MODE_ARGS=(--output_mode "${STREAMING_OUTPUT_MODE}")
FLASHDREAMS_MODE_ARGS=(--output-mode "${STREAMING_OUTPUT_MODE}")
if sana_test_is_true "${STREAMING_NO_COMPILE}"; then
    UPSTREAM_MODE_ARGS+=(--no_compile)
fi
FLASHDREAMS_COMPILE_ARGS=()
if ! sana_test_is_true "${STREAMING_NO_COMPILE}"; then
    FLASHDREAMS_COMPILE_ARGS+=(--compile-streaming-refiner)
fi
FLASHDREAMS_BACKEND_ARGS=()
if sana_test_is_true "${FORCE_CUDNN_SDPA}"; then
    FLASHDREAMS_BACKEND_ARGS+=(--force-cudnn-sdpa)
fi

echo "[run] upstream SANA-WM streaming -> ${UPSTREAM_OUT}"
( cd "${SANA_TEST_DIR}" && \
    PYTHONPATH="${UPSTREAM_PYTHONPATH}" \
    uv run python "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm_streaming.py" \
        --image "${IMAGE_PATH}" \
        --prompt "${PROMPT_PATH}" \
        "${UPSTREAM_CAMERA_ARGS[@]}" \
        --intrinsics "${INTRINSICS_PATH}" \
        --output_dir "${UPSTREAM_OUT}" \
        --name upstream \
        --num_frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --cfg_scale "${CFG_SCALE}" \
        --flow_shift "${FLOW_SHIFT}" \
        --seed "${SEED}" \
        --benchmark_json "${UPSTREAM_STATS}" \
        --sample_frames_npz "${UPSTREAM_FRAMES}" \
        --sample_frame_stride "${STREAMING_SAMPLE_FRAME_STRIDE}" \
        --denoising_step_list "${STREAMING_DENOISING_STEP_LIST}" \
        --num_frame_per_block "${STREAMING_NUM_FRAME_PER_BLOCK}" \
        --num_cached_blocks "${STREAMING_NUM_CACHED_BLOCKS}" \
        --sink_size "${STREAMING_SINK_SIZE}" \
        --refiner_block_size "${STREAMING_REFINER_BLOCK_SIZE}" \
        --refiner_kv_max_frames "${STREAMING_REFINER_KV_MAX_FRAMES}" \
        --refiner_seed "${REFINER_SEED}" \
        --streaming_crf "${STREAMING_CRF}" \
        --streaming_preset "${STREAMING_PRESET}" \
        --streaming_encoder "${STREAMING_ENCODER}" \
        "${UPSTREAM_PRECISION_ARGS[@]}" \
        "${UPSTREAM_MODE_ARGS[@]}" )

echo "[run] FlashDreams SANA-WM streaming -> ${FLASHDREAMS_OUT}"
( cd "${SANA_TEST_DIR}" && \
    uv run python "${SANA_TEST_DIR}/run_flashdreams_streaming.py" \
        --image-path "${IMAGE_PATH}" \
        --prompt-path "${PROMPT_PATH}" \
        "${FLASHDREAMS_CAMERA_ARGS[@]}" \
        --intrinsics-path "${INTRINSICS_PATH}" \
        --output-dir "${FLASHDREAMS_OUT}" \
        --name flashdreams \
        --num-frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --cfg-scale "${CFG_SCALE}" \
        --flow-shift "${FLOW_SHIFT}" \
        --seed "${SEED}" \
        --refiner-seed "${REFINER_SEED}" \
        --stats-json "${FLASHDREAMS_STATS}" \
        --dump-frames "${FLASHDREAMS_FRAMES}" \
        --denoising-step-list "${STREAMING_DENOISING_STEP_LIST}" \
        --num-frame-per-block "${STREAMING_NUM_FRAME_PER_BLOCK}" \
        --num-cached-blocks "${STREAMING_NUM_CACHED_BLOCKS}" \
        --sink-size "${STREAMING_SINK_SIZE}" \
        --refiner-block-size "${STREAMING_REFINER_BLOCK_SIZE}" \
        --refiner-kv-max-frames "${STREAMING_REFINER_KV_MAX_FRAMES}" \
        "${FLASHDREAMS_MODE_ARGS[@]}" \
        "${FLASHDREAMS_COMPILE_ARGS[@]}" \
        "${FLASHDREAMS_PRECISION_ARGS[@]}" \
        "${FLASHDREAMS_BACKEND_ARGS[@]}" )

echo "[diff] summarising streaming frame parity -> ${OUTPUT_DIR}/parity.json"
( cd "${SANA_TEST_DIR}" && \
    uv run python "${PARITY_SCRIPT_DIR}/diff_parity.py" \
        --upstream "${UPSTREAM_FRAMES}" \
        --flashdreams "${FLASHDREAMS_FRAMES}" \
        --output "${OUTPUT_DIR}/parity.json" )

echo "[diff] summarising streaming continuity -> ${OUTPUT_DIR}/continuity.json"
( cd "${SANA_TEST_DIR}" && \
    uv run python "${PARITY_SCRIPT_DIR}/streaming_continuity.py" \
        --upstream "${UPSTREAM_FRAMES}" \
        --flashdreams "${FLASHDREAMS_FRAMES}" \
        --chunk-size "$(( STREAMING_REFINER_BLOCK_SIZE * 8 ))" \
        --output "${OUTPUT_DIR}/continuity.json" )

echo "[run] done."
echo "      upstream frames   : ${UPSTREAM_FRAMES}"
echo "      flashdreams frames: ${FLASHDREAMS_FRAMES}"
echo "      parity JSON       : ${OUTPUT_DIR}/parity.json"
echo "      continuity JSON   : ${OUTPUT_DIR}/continuity.json"
