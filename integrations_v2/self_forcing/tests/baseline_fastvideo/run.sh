#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/FastVideo"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/hao-ai-lab/FastVideo.git"
PIN_COMMIT="af2ee9c78a55ba4922ac36f40e99d07438410904"

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

if [[ -f "${PATCH_FILE}" ]]; then
    if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] patch already applied, skipping"
    elif git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "[setup] applying ${PATCH_FILE}"
        git apply "${PATCH_FILE}"
    else
        echo "[setup] ERROR: ${PATCH_FILE} neither applies cleanly nor is already applied." >&2
        exit 1
    fi
else
    echo "[setup] no patch file at ${PATCH_FILE}, skipping"
fi

echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

echo "[run] starting FastVideo Self-Forcing benchmark"
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
FASTVIDEO_FORCE_CUDNN_SDPA=1 \
PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
"${SCRIPT_DIR}/.venv/bin/python" "${REPO_DIR}/examples/inference/basic/benchmark_self_forcing_causal.py" \
    --enable_torch_compile \
    --output "./videos/offline.mp4" \
    --stats_output "./videos/stats_offline.json"
