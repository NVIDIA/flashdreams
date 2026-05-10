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

# Pull Self-Forcing, apply local patch, and run the benchmark.
# Idempotent: re-running skips clone / checkout / downloads / patch when
# already in place, and just re-runs the benchmark.

set -euo pipefail

# Resolve the directory containing this script so the script can be invoked
# from anywhere. ``../changes.patch`` in the original was implicit; here we
# anchor everything to ``SCRIPT_DIR``.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/Self-Forcing"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/guandeh17/Self-Forcing.git"
PIN_COMMIT="33593df3e81fa3ec10239271dd2c100facac6de1"

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

# --------------------------------------------------------------- HF downloads
if [[ ! -d "wan_models/Wan2.1-T2V-1.3B" ]] \
        || [[ -z "$(ls -A wan_models/Wan2.1-T2V-1.3B 2>/dev/null)" ]]; then
    echo "[setup] downloading Wan2.1-T2V-1.3B"
    uv run huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir-use-symlinks False \
        --local-dir wan_models/Wan2.1-T2V-1.3B
else
    echo "[setup] wan_models/Wan2.1-T2V-1.3B exists, skipping download"
fi

if [[ ! -f "checkpoints/self_forcing_dmd.pt" ]]; then
    echo "[setup] downloading self_forcing_dmd.pt"
    uv run huggingface-cli download gdhe17/Self-Forcing \
        checkpoints/self_forcing_dmd.pt \
        --local-dir .
else
    echo "[setup] checkpoints/self_forcing_dmd.pt exists, skipping download"
fi

# ------------------------------------------------------------------- pip deps
# Materialize the isolated venv defined by ``${SCRIPT_DIR}/pyproject.toml``.
# ``uv sync`` is no-op-fast when the venv is already in sync. Run it from
# ${SCRIPT_DIR} so uv finds *this* project's pyproject (not flashdreams').
# All subsequent ``uv run`` calls (from inside ${REPO_DIR}) walk up and
# resolve to the same ``${SCRIPT_DIR}/.venv``.
echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

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

# ----------------------------------------------------------------- benchmark
echo "[run] starting benchmark"
FORCE_CUDNN_ATTN=1 uv run python benchmark.py \
    --enable_torch_compile \
    --use_taehv
