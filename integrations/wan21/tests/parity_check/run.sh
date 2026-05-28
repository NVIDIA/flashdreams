#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/Wan2.1"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/Wan-Video/Wan2.1.git"
PIN_COMMIT="9737cba9c1c3c4d04b33fcad41c111989865d315"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
CKPT_DIR="${REPO_DIR}/Wan2.1-T2V-1.3B"

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

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "[setup] creating isolated venv at ${SCRIPT_DIR}/.venv"
    uv venv --python 3.12 "${SCRIPT_DIR}/.venv"
fi

echo "[setup] installing Wan2.1 requirements"
uv pip install --python "${VENV_PYTHON}" "torch>=2.11" "torchvision>=0.26"

FILTERED_REQUIREMENTS="$(mktemp)"
"${VENV_PYTHON}" - <<'PY' "${REPO_DIR}/requirements.txt" "${FILTERED_REQUIREMENTS}"
import re
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
skip = re.compile(r"^\s*(torch|torchvision|flash_attn)\b")

with open(src_path, encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
    for line in src:
        if skip.match(line):
            continue
        dst.write(line)
PY

uv pip install --python "${VENV_PYTHON}" -r "${FILTERED_REQUIREMENTS}"
rm -f "${FILTERED_REQUIREMENTS}"

# Upstream requirements.txt currently misses einops, but wan/modules/vae.py
# imports it at module-import time.
uv pip install --python "${VENV_PYTHON}" "einops"

echo "[setup] installing huggingface_hub CLI"
uv pip install --python "${VENV_PYTHON}" "huggingface_hub[cli]"

echo "[setup] installing flashdreams package for env alignment (no-deps)"
uv pip install --python "${VENV_PYTHON}" --no-deps -e "${REPO_ROOT}/flashdreams"

if [[ ! -d "${CKPT_DIR}" ]] || [[ -z "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]]; then
    echo "[setup] downloading Wan2.1-T2V-1.3B checkpoint"
    "${SCRIPT_DIR}/.venv/bin/hf" download \
        Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir "${CKPT_DIR}"
else
    echo "[setup] checkpoint already present at ${CKPT_DIR}, skipping download"
fi

echo "[run] starting official Wan2.1 parity run with forced cuDNN SDPA + torch.compile"
FORCE_CUDNN_ATTN=1 \
WAN_FORCE_TORCH_COMPILE=1 \
WAN_TORCH_COMPILE_MODE=default \
PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
"${VENV_PYTHON}" "${REPO_DIR}/generate.py" \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir "${CKPT_DIR}" \
    --sample_shift 8 \
    --sample_guide_scale 6 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
