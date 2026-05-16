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

# Pull HY-WorldPlay, optionally apply the local patch, and run the
# upstream WAN-5B benchmark. Idempotent: re-running skips clone /
# checkout / downloads / patch when already in place, and just re-runs
# the benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/HY-WorldPlay"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"
REPO_URL="https://github.com/Tencent-Hunyuan/HY-WorldPlay.git"
# Pinned to the head of ``main`` at the time this integration was
# scaffolded. Bump when re-baselining; the patch may need to be
# refreshed too.
PIN_COMMIT="HEAD"

# Where the WAN-5B HuggingFace checkpoints live inside ${REPO_DIR}.
HF_MODELS_DIR="${REPO_DIR}/hf_models"
HF_REPO="tencent/HY-WorldPlay"
NUM_GPU="${NUM_GPU:-1}"
NUM_CHUNK="${NUM_CHUNK:-1}"
POSE="${POSE:-w-4}"
SEED="${SEED:-0}"
PROMPT="${PROMPT:-First-person view walking around ancient Athens, with Greek architecture and marble structures}"
IMAGE_PATH="${IMAGE_PATH:-${REPO_DIR}/assets/img/test.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/parity}"

# ---------------------------------------------------------------- clone + pin
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    echo "[setup] repo already present at ${REPO_DIR}, skipping clone"
fi

cd "${REPO_DIR}"

if [[ "${PIN_COMMIT}" != "HEAD" ]]; then
    CURRENT_COMMIT="$(git rev-parse HEAD)"
    if [[ "${CURRENT_COMMIT}" != "${PIN_COMMIT}" ]]; then
        echo "[setup] checking out pinned commit ${PIN_COMMIT}"
        git checkout "${PIN_COMMIT}"
    else
        echo "[setup] already at pinned commit ${PIN_COMMIT}, skipping checkout"
    fi
fi

# ------------------------------------------------------------------- patching
# ``changes.patch`` is optional in phase 1 (we don't need any upstream
# edits to reproduce the baseline). Wire the same apply / skip /
# fail-loudly machinery as ``self_forcing/parity_check`` so a future
# patch (e.g. ``EventProfiler`` per-chunk timing, attention dispatcher
# routing) can be dropped in without touching this script.
if [[ -f "${PATCH_FILE}" ]]; then
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
# ``huggingface-cli download`` treats positional args after the repo id as
# *exact filenames*, not directory prefixes -- so passing
# ``wan_transformer wan_distilled_model`` matches zero files and silently
# exits 0 with nothing fetched (prints "Fetching 0 files: 0it [00:00]").
# Use ``--include`` glob patterns to grab whole subdirectories instead.
#
# Total payload is ~52 GiB (wan_transformer ~10 GiB, wan_distilled_model
# ~42 GiB), so the first run takes a while; subsequent runs no-op via the
# directory-not-empty guard below.
WAN_TRANSFORMER_CONFIG="${HF_MODELS_DIR}/wan_transformer/config.json"
WAN_DISTILLED_CKPT="${HF_MODELS_DIR}/wan_distilled_model/model.pt"
if [[ ! -f "${WAN_TRANSFORMER_CONFIG}" || ! -f "${WAN_DISTILLED_CKPT}" ]]; then
    echo "[setup] downloading ${HF_REPO} {wan_transformer/, wan_distilled_model/} -> ${HF_MODELS_DIR}"
    uv run huggingface-cli download "${HF_REPO}" \
        --include "wan_transformer/*" "wan_distilled_model/*" \
        --local-dir "${HF_MODELS_DIR}"
else
    echo "[setup] HY-WorldPlay WAN models already present in ${HF_MODELS_DIR}, skipping download"
fi

# ------------------------------------------------------------------- pip deps
# Materialize the isolated venv defined by ``${SCRIPT_DIR}/pyproject.toml``.
# ``uv sync`` is no-op-fast when the venv is already in sync. Run it from
# ${SCRIPT_DIR} so uv finds *this* project's pyproject (not flashdreams').
# All subsequent ``uv run`` calls (from inside ${REPO_DIR}) walk up and
# resolve to the same ``${SCRIPT_DIR}/.venv``.
echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

# --------------------------------------------------------------------- inputs
if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "[setup] ERROR: --image_path ${IMAGE_PATH} does not exist." >&2
    echo "        Set IMAGE_PATH env var to a valid first-frame jpg/png," >&2
    echo "        or check that ${REPO_DIR}/assets/img/test.png is present." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------- benchmark
# Mirrors the upstream invocation in ``HY-WorldPlay/wan/README.md``:
#   PYTHONPATH=$(pwd):$(pwd)/wan torchrun --nproc_per_node=NUM_GPU \
#       wan/generate.py --input "..." --image_path ... \
#       --num_chunk N --pose ... \
#       --ar_model_path .../wan_transformer \
#       --ckpt_path .../wan_distilled_model/model.pt \
#       --out outputs
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/wan:${PYTHONPATH:-}"

echo "[run] starting upstream WAN-5B benchmark [${NUM_GPU} GPU(s), num_chunk=${NUM_CHUNK}, pose=${POSE}]"
uv run torchrun --nproc_per_node="${NUM_GPU}" wan/generate.py \
    --input "${PROMPT}" \
    --image_path "${IMAGE_PATH}" \
    --num_chunk "${NUM_CHUNK}" \
    --pose "${POSE}" \
    --ar_model_path "${HF_MODELS_DIR}/wan_transformer" \
    --ckpt_path "${HF_MODELS_DIR}/wan_distilled_model/model.pt" \
    --out "${OUTPUT_DIR}"

echo "[run] done; outputs under ${OUTPUT_DIR}"
