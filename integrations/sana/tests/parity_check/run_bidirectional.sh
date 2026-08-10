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

# Patch pinned upstream SANA-WM with bidirectional instrumentation, run it and
# the FlashDreams bidirectional integration on the same demo input, dump
# decoded uint8 frames, and compute mean |Delta| / 255.

set -euo pipefail

PARITY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_TEST_DIR="$(cd "${PARITY_SCRIPT_DIR}/.." && pwd)"
source "${SANA_TEST_DIR}/common.sh"

SANA_REPO="$(sana_test_abspath "${SANA_REPO:-${SANA_TEST_DIR}/Sana}")"
PATCH_FILE="${SANA_TEST_DIR}/changes_bidirectional.patch"

OUTPUT_DIR="$(sana_test_abspath "${OUTPUT_DIR:-${PARITY_SCRIPT_DIR}/outputs/parity}")"
UPSTREAM_OUT="${OUTPUT_DIR}/upstream"
FLASHDREAMS_OUT="${OUTPUT_DIR}/flashdreams"
IMAGE_PATH="$(sana_test_abspath "${IMAGE_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.png}")"
PROMPT_PATH="$(sana_test_abspath "${PROMPT_PATH:-${SANA_REPO}/asset/sana_wm/demo_0.txt}")"
CAMERA_PATH="$(sana_test_abspath "${CAMERA_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_pose.npy}")"
INTRINSICS_PATH="$(sana_test_abspath "${INTRINSICS_PATH:-${SANA_REPO}/asset/sana_wm/demo_0_intrinsics.npy}")"
NUM_FRAMES="${NUM_FRAMES:-121}"
FPS="${FPS:-16}"
STEP="${STEP:-60}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SEED="${SEED:-42}"
NO_REFINER="${NO_REFINER:-1}"
FORCE_CUDNN_SDPA="${FORCE_CUDNN_SDPA:-0}"
COMPILE_STAGE1="${COMPILE_STAGE1:-0}"
STAGE1_PRECISION="${STAGE1_PRECISION:-bf16}"
REFINER_PRECISION="${REFINER_PRECISION:-${STAGE1_PRECISION}}"
QUANT_BACKEND="${QUANT_BACKEND:-auto}"

sana_test_prepare_sana_repo "${SANA_REPO}" "${PATCH_FILE}"
sana_test_sync_venv
UPSTREAM_PYTHONPATH="$(sana_test_upstream_pythonpath "${SANA_REPO}")"

mkdir -p "${UPSTREAM_OUT}" "${FLASHDREAMS_OUT}"

UPSTREAM_FRAMES="${UPSTREAM_OUT}/frames.npy"
FLASHDREAMS_FRAMES="${FLASHDREAMS_OUT}/frames.npy"
UPSTREAM_STATS="${UPSTREAM_OUT}/stats.json"
FLASHDREAMS_STATS="${FLASHDREAMS_OUT}/stats.json"

UPSTREAM_REFINER_ARGS=()
FLASHDREAMS_REFINER_ARGS=()
if sana_test_is_true "${NO_REFINER}"; then
    UPSTREAM_REFINER_ARGS+=(--no_refiner)
    FLASHDREAMS_REFINER_ARGS+=(--no-refiner)
fi
UPSTREAM_BACKEND_ARGS=()
FLASHDREAMS_BACKEND_ARGS=()
if sana_test_is_true "${FORCE_CUDNN_SDPA}"; then
    UPSTREAM_BACKEND_ARGS+=(--force_cudnn_sdpa)
    FLASHDREAMS_BACKEND_ARGS+=(--force-cudnn-sdpa)
fi
UPSTREAM_COMPILE_ARGS=()
FLASHDREAMS_COMPILE_ARGS=()
if sana_test_is_true "${COMPILE_STAGE1}"; then
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

echo "[run] upstream SANA-WM bidirectional -> ${UPSTREAM_OUT}"
( cd "${SANA_TEST_DIR}" && \
    PYTHONPATH="${UPSTREAM_PYTHONPATH}" \
    uv run python "${SANA_REPO}/inference_video_scripts/wm/inference_sana_wm.py" \
        --image "${IMAGE_PATH}" \
        --prompt "${PROMPT_PATH}" \
        --camera "${CAMERA_PATH}" \
        --intrinsics "${INTRINSICS_PATH}" \
        --output_dir "${UPSTREAM_OUT}" \
        --name upstream \
        --num_frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --step "${STEP}" \
        --cfg_scale "${CFG_SCALE}" \
        --seed "${SEED}" \
        --no_action_overlay \
        --dump_frames "${UPSTREAM_FRAMES}" \
        --stats_json "${UPSTREAM_STATS}" \
        "${UPSTREAM_PRECISION_ARGS[@]}" \
        "${UPSTREAM_BACKEND_ARGS[@]}" \
        "${UPSTREAM_COMPILE_ARGS[@]}" \
        "${UPSTREAM_REFINER_ARGS[@]}" )

echo "[run] FlashDreams SANA-WM bidirectional -> ${FLASHDREAMS_OUT}"
( cd "${SANA_TEST_DIR}" && \
    uv run python "${SANA_TEST_DIR}/run_flashdreams_bidirectional.py" \
        --image-path "${IMAGE_PATH}" \
        --prompt-path "${PROMPT_PATH}" \
        --camera-path "${CAMERA_PATH}" \
        --intrinsics-path "${INTRINSICS_PATH}" \
        --output-dir "${FLASHDREAMS_OUT}" \
        --name flashdreams \
        --num-frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --step "${STEP}" \
        --cfg-scale "${CFG_SCALE}" \
        --seed "${SEED}" \
        --dump-frames "${FLASHDREAMS_FRAMES}" \
        --stats-json "${FLASHDREAMS_STATS}" \
        "${FLASHDREAMS_PRECISION_ARGS[@]}" \
        "${FLASHDREAMS_BACKEND_ARGS[@]}" \
        "${FLASHDREAMS_COMPILE_ARGS[@]}" \
        "${FLASHDREAMS_REFINER_ARGS[@]}" )

echo "[diff] summarising parity -> ${OUTPUT_DIR}/parity.json"
( cd "${SANA_TEST_DIR}" && \
    uv run python "${PARITY_SCRIPT_DIR}/diff_parity.py" \
        --upstream "${UPSTREAM_FRAMES}" \
        --flashdreams "${FLASHDREAMS_FRAMES}" \
        --output "${OUTPUT_DIR}/parity.json" )

echo "[run] done."
echo "      upstream frames   : ${UPSTREAM_FRAMES}"
echo "      flashdreams frames: ${FLASHDREAMS_FRAMES}"
echo "      parity JSON       : ${OUTPUT_DIR}/parity.json"
