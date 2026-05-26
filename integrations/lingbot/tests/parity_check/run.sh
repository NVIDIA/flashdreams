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

# Pull LingBot-World, apply local patch, and run the benchmark.
# Idempotent: re-running skips clone / checkout / downloads / patch when
# already in place, and just re-runs the benchmark.

set -euo pipefail

# Resolve the directory containing this script so the script can be invoked
# from anywhere. ``../changes.patch`` in the original was implicit; here we
# anchor everything to ``SCRIPT_DIR``.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/lingbot-world"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/Robbyant/lingbot-world.git"
PIN_COMMIT="9660e9405fbc887655e2bc79ac09d61fa81128ae"

# ---------------------------------------------------------------- clone + pin
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

# ------------------------------------------------------------------- patching
if [[ -f "${PATCH_FILE}" ]]; then
    # ``git apply --reverse --check`` succeeds iff the patch is *already*
    # applied. ``git apply --check`` succeeds iff the patch is *cleanly
    # applicable*. We use both to choose between apply / skip / fail-loudly.
    if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] patch already applied, skipping"
    elif git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] applying ${PATCH_FILE}"
        git apply "${PATCH_FILE}"
    else
        echo "[setup] ERROR: ${PATCH_FILE} neither cleanly applies nor is" \
             "already applied; tree may be partially patched or out of sync." >&2
        exit 1
    fi
else
    echo "[setup] no patch file at ${PATCH_FILE}, skipping"
fi

# --------------------------------------------------------------- HF downloads
if [[ ! -d "lingbot-world-base-cam" ]] \
        || [[ -z "$(ls -A lingbot-world-base-cam 2>/dev/null)" ]]; then
    echo "[setup] downloading lingbot-world-base-cam"
    uv run huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
else
    echo "[setup] lingbot-world-base-cam exists, skipping download"
fi

# ``lingbot_world_fast`` is a directory of HF files, not a regular file -- use
# ``-d`` + empty-dir guard, same shape as the check above.
if [[ ! -d "lingbot-world-base-cam/lingbot_world_fast" ]] \
        || [[ -z "$(ls -A lingbot-world-base-cam/lingbot_world_fast 2>/dev/null)" ]]; then
    echo "[setup] downloading lingbot-world-fast"
    uv run huggingface-cli download robbyant/lingbot-world-fast --local-dir ./lingbot-world-base-cam/lingbot_world_fast
else
    echo "[setup] lingbot-world-fast exists, skipping download"
fi

# ------------------------------------------------------------------- pip deps
# Materialize the isolated venv defined by ``${SCRIPT_DIR}/pyproject.toml``.
# ``uv sync`` is no-op-fast when the venv is already in sync. Run it from
# ${SCRIPT_DIR} so uv finds *this* project's pyproject (not flashdreams').
# All subsequent ``uv run`` calls (from inside ${REPO_DIR}) walk up and
# resolve to the same ``${SCRIPT_DIR}/.venv``.
echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

# ----------------------------------------------------------------- benchmark
PROMPT="The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."

echo "[run] starting benchmark [1 GPU]"
FORCE_CUDNN_ATTN=1 \
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir lingbot-world-base-cam \
    --image examples/00/image.jpg \
    --action_path examples/00 \
    --ulysses_size 1 \
    --frame_num 237 \
    --base_seed 42 \
    --offload_model False \
    --prompt "${PROMPT}"

echo "[run] starting benchmark [4 GPUs]"
FORCE_CUDNN_ATTN=1 \
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir lingbot-world-base-cam \
    --image examples/00/image.jpg \
    --action_path examples/00 \
    --ulysses_size 4 \
    --frame_num 237 \
    --base_seed 42 \
    --offload_model False \
    --prompt "${PROMPT}"
