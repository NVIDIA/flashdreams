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

# Pull HY-WorldPlay and run the upstream WAN-5B benchmark. Idempotent:
# re-running skips clone / checkout / downloads when already in place,
# and just re-runs the benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Override-friendly so the ~30 GiB clone + ~52 GiB HF checkpoints can be
# parked outside /home (e.g. ``REPO_DIR=~/scratch/HY-WorldPlay bash run.sh``).
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}/HY-WorldPlay}"
REPO_URL="https://github.com/Tencent-Hunyuan/HY-WorldPlay.git"
# Pinned to the head of ``main`` at the time this integration was
# scaffolded. Bump when re-baselining.
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
# ``git clone`` refuses non-empty targets, but ``bench_batch.sh`` may
# have already populated ``${REPO_DIR}/hf_models/`` (distilled
# checkpoint download). Init + fetch + checkout in place when the
# directory exists, preserving anything HF already dropped there.
if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] repo already present at ${REPO_DIR}, skipping clone"
elif [[ -d "${REPO_DIR}" ]]; then
    echo "[setup] ${REPO_DIR} exists but is not a git repo; init + fetch in place"
    git -C "${REPO_DIR}" init -q
    git -C "${REPO_DIR}" remote add origin "${REPO_URL}"
    git -C "${REPO_DIR}" fetch --depth=1 origin HEAD
    git -C "${REPO_DIR}" checkout -f FETCH_HEAD
else
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
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

# Upstream's ``wan/generate.py`` ``torch.load(..., map_location=self.device)``
# fits the entire 40 GiB distilled .pt onto the GPU all at once -- OOMs on
# anything below ~48 GiB VRAM. Flip the load to CPU so the file lands in RAM
# first and ``load_state_dict`` then moves tensors GPU-side one at a time
# (peak GPU footprint = model size, not file size). Idempotent: re-runs
# find the substring already gone and no-op.
if grep -q "map_location=self.device" "${REPO_DIR}/wan/generate.py"; then
    echo "[setup] patching wan/generate.py to load checkpoint to CPU first (avoid 40 GiB GPU OOM)"
    sed -i 's|map_location=self.device|map_location="cpu"|' "${REPO_DIR}/wan/generate.py"
fi

# --------------------------------------------------------------- HF downloads
# ``hf download`` treats positional args after the repo id as
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
    # ``hf download`` treats *any* positional after the repo id as an
    # exact filename and silently ignores ``--include`` once one is
    # set. Pass each glob via a *separate* ``--include`` flag so neither
    # collapses into the positional slot.
    uv run hf download "${HF_REPO}" \
        --include "wan_transformer/*" \
        --include "wan_distilled_model/*" \
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
#
# The lightweight sync below covers the *native* plugin path only.
# Vendor's ``wan/generate.py`` additionally requires four heavy deps
# (kept out of the sub-venv's ``pyproject.toml`` because their resolution
# toll on the repo-root lock was deemed too high once parity closed).
# This script ``uv pip install``s them on demand below unless
# ``SKIP_HEAVY_DEPS=1`` is set.
echo "[setup] ensuring Python deps via uv sync (isolated venv)"
( cd "${SCRIPT_DIR}" && uv sync )

if [[ "${SKIP_HEAVY_DEPS:-0}" != "1" ]]; then
    echo "[setup] installing vendor-only heavy deps (sageattention, cloudpickle, accelerate, transformers==4.57.6)"
    echo "        set SKIP_HEAVY_DEPS=1 to skip if you only need the native plugin"
    ( cd "${SCRIPT_DIR}" && uv pip install \
        sageattention \
        cloudpickle \
        "accelerate>=0.30" \
        "transformers==4.57.6" )
else
    echo "[setup] SKIP_HEAVY_DEPS=1 -> assuming vendor heavy deps already installed (or running native plugin only)"
fi

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
#
# Set ``USE_KV_CACHE_TRUE=1`` to swap in ``run_vendor_use_kv_cache.py``:
# a runtime monkey-patch that coerces ``WanPipeline.use_kv_cache=True``
# (vendor's ``predict`` defaults it to ``False`` at line 707 of
# ``pipeline_wan_w_mem_relative_rope.py``). The mode re-baselines the
# vendor reference against the cache-prefill code path the native HY
# runner mirrors -- the phase 2b.6 acceptance baseline; see
# ``docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md``.
# Default (no env var) keeps producing the
# phase-1 ``use_kv_cache=False`` baseline so older parity numbers
# stay comparable.
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/wan:${PYTHONPATH:-}"

if [[ "${USE_KV_CACHE_TRUE:-0}" == "1" ]]; then
    GENERATE_SCRIPT="${SCRIPT_DIR}/run_vendor_use_kv_cache.py"
    echo "[run] USE_KV_CACHE_TRUE=1 -> wrapping wan/generate.py via ${GENERATE_SCRIPT}"
else
    GENERATE_SCRIPT="${REPO_DIR}/wan/generate.py"
fi

echo "[run] starting upstream WAN-5B benchmark [${NUM_GPU} GPU(s), num_chunk=${NUM_CHUNK}, pose=${POSE}]"
uv run torchrun --nproc_per_node="${NUM_GPU}" "${GENERATE_SCRIPT}" \
    --input "${PROMPT}" \
    --image_path "${IMAGE_PATH}" \
    --num_chunk "${NUM_CHUNK}" \
    --pose "${POSE}" \
    --ar_model_path "${HF_MODELS_DIR}/wan_transformer" \
    --ckpt_path "${HF_MODELS_DIR}/wan_distilled_model/model.pt" \
    --out "${OUTPUT_DIR}"

echo "[run] done; outputs under ${OUTPUT_DIR}"
