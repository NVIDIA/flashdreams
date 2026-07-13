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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/lingbot-world-v2"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/robbyant/lingbot-world-v2.git"
PIN_COMMIT="94f43115de8d4a4f9f282126528c300a0b232c5f"

CHECKPOINT_REPO="${CHECKPOINT_REPO:-robbyant/lingbot-world-v2-14b-causal-fast}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-lingbot-world-v2-14b-causal-fast}"
EXAMPLE_IDX="${EXAMPLE_IDX:-03}"
FRAME_NUM="${FRAME_NUM:-81}"
NPROC="${NPROC:-1}"
SIZE="${SIZE:-480*832}"
BASE_SEED="${BASE_SEED:-42}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-18}"
SINK_SIZE="${SINK_SIZE:-6}"
PROMPT="${PROMPT:-}"
SAVE_DIR="${SAVE_DIR:-output}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    echo "[setup] repo already present at ${REPO_DIR}, skipping clone"
fi

cd "${REPO_DIR}"

CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${PIN_COMMIT}" ]]; then
    echo "[setup] checking out pinned commit ${PIN_COMMIT}"
    git checkout "${PIN_COMMIT}"
else
    echo "[setup] already at pinned commit ${PIN_COMMIT}, skipping checkout"
fi

if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[setup] patch already applied, skipping"
elif git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[setup] applying ${PATCH_FILE}"
    git apply "${PATCH_FILE}"
else
    echo "[setup] ERROR: ${PATCH_FILE} neither cleanly applies nor is already applied." >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]] || [[ -z "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]]; then
    echo "[setup] downloading ${CHECKPOINT_REPO} -> ${CHECKPOINT_DIR}"
    uv run huggingface-cli download "${CHECKPOINT_REPO}" --local-dir "./${CHECKPOINT_DIR}"
else
    echo "[setup] ${CHECKPOINT_DIR} exists, skipping checkpoint download"
fi

echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

if [[ -z "${PROMPT}" ]]; then
    PROMPT="$(tr '\n' ' ' < "examples/${EXAMPLE_IDX}/prompt.txt" 2>/dev/null || true)"
fi
if [[ -z "${PROMPT}" ]] && [[ "${EXAMPLE_IDX}" == "03" ]]; then
    PROMPT="A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds - gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere."
fi
if [[ -z "${PROMPT}" ]]; then
    PROMPT="A sweeping cinematic journey through a fantasy world with smooth camera motion."
fi

mkdir -p "${SAVE_DIR}"
SAVE_FILE="${SAVE_DIR}/lingbot-world-v2-parity-${EXAMPLE_IDX}-${NPROC}gpu.mp4"

TORCHRUN_ARGS=(--standalone --nnodes=1 "--nproc_per_node=${NPROC}")
MODEL_ARGS=(
    generate.py
    --task i2v-A14B
    --infer_mode causal_fast
    --size "${SIZE}"
    --ckpt_dir "${CHECKPOINT_DIR}"
    --image "examples/${EXAMPLE_IDX}/image.jpg"
    --action_path "examples/${EXAMPLE_IDX}"
    --frame_num "${FRAME_NUM}"
    --chunk_size 3
    --base_seed "${BASE_SEED}"
    --offload_model False
    --prompt "${PROMPT}"
    --local_attn_size "${LOCAL_ATTN_SIZE}"
    --sink_size "${SINK_SIZE}"
    --save_file "${SAVE_FILE}"
)

if (( NPROC > 1 )); then
    MODEL_ARGS+=(--dit_fsdp --t5_fsdp --ulysses_size "${NPROC}")
else
    MODEL_ARGS+=(--ulysses_size 1)
fi

echo "[run] starting LingBot-World v2 official parity run [${NPROC} GPU(s)]"
echo "[run] output: ${REPO_DIR}/${SAVE_FILE}"
uv run python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" "${MODEL_ARGS[@]}"
