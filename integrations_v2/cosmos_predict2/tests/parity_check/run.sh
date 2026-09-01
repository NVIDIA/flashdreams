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

# Pull cosmos-predict2.5 at a pinned commit, apply our local patch, and
# run the upstream T2V + I2V base-model inference cmds *using cosmos's
# own pyproject.toml + uv lockfile*. Idempotent: re-running skips
# clone / checkout / patch / sync when already in place and just re-runs
# inference.
#
# We deliberately do NOT stack on top of the flashdreams venv here.
# Cosmos-predict2.5's installable closure includes a CUDA/torch combo
# pinned to prebuilt wheels on NVIDIA's custom index
# (flash-attn / decord / transformer-engine / natten / torch==2.9.1 +
# cu130, see `cosmos-predict2.5/packages/cosmos-oss/pyproject.toml`
# lines 126-156 / 176-206). Those wheels are ABI-bonded to that torch
# version, so they don't compose with flashdreams' `torch>=2.11` pin.
# Using cosmos's own pyproject as-is is the only sane way to land
# upstream parity quickly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/cosmos-predict2.5"
PATCH_FILE="${SCRIPT_DIR}/changes.patch"

REPO_URL="https://github.com/nvidia-cosmos/cosmos-predict2.5.git"
PIN_COMMIT="441b89740d91922737008a61e7f71407d47944e7"

# ---------------------------------------------------------------- clone + pin
# The container ships without ``git-lfs``, so cosmos-predict2.5's
# ``.gitattributes`` LFS filter rules make ``git clone`` / ``git
# checkout`` abort with "git-lfs filter-process: 1: git-lfs: not found"
# / "Clone succeeded, but checkout failed.". ``GIT_LFS_SKIP_SMUDGE=1``
# alone isn't enough -- it's an env var the ``git-lfs`` binary itself
# reads, and that binary doesn't exist. We instead use ``git -c
# filter.lfs.*`` flags to point git at no-op replacements for the smudge
# / clean / process filter and mark the filter as not required, so git
# itself never invokes ``git-lfs``. Cosmos only LFS-tracks docs assets;
# the model weights inference actually needs come from HF Hub at
# runtime, so skipping the smudge is safe end-to-end.
GIT_NO_LFS=(
    git
    -c "filter.lfs.smudge=cat"
    -c "filter.lfs.clean=cat"
    -c "filter.lfs.process="
    -c "filter.lfs.required=false"
)

# Recover from a previous broken clone (``.git`` exists but the working
# tree was never populated because the lfs smudge filter aborted).
# ``examples/inference.py`` is a regular text file we always need, so
# use it as the canary.
if [[ -d "${REPO_DIR}/.git" && ! -f "${REPO_DIR}/examples/inference.py" ]]; then
    echo "[setup] previous clone at ${REPO_DIR} looks broken (missing files); removing"
    rm -rf "${REPO_DIR}"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[setup] cloning ${REPO_URL} -> ${REPO_DIR}"
    "${GIT_NO_LFS[@]}" clone "${REPO_URL}" "${REPO_DIR}"
else
    echo "[setup] repo already present at ${REPO_DIR}, skipping clone"
fi

(
    cd "${REPO_DIR}"
    CURRENT_COMMIT="$("${GIT_NO_LFS[@]}" rev-parse HEAD)"
    if [[ "${CURRENT_COMMIT}" != "${PIN_COMMIT}" ]]; then
        echo "[setup] checking out pinned commit ${PIN_COMMIT}"
        "${GIT_NO_LFS[@]}" checkout "${PIN_COMMIT}"
    else
        echo "[setup] already at pinned commit ${PIN_COMMIT}, skipping checkout"
    fi
)

# ------------------------------------------------------------------- patching
# Local edits we layer on top of the pinned upstream commit live in
# ``changes.patch`` (RNG layout swap to match flashdreams' [B,T,C,H,W]
# order in both ``text2world_model.py`` /
# ``text2world_model_rectified_flow.py``, default seed 0->42).
#
# ``git apply --reverse --check`` succeeds iff the patch is *already*
# applied; ``git apply --check`` succeeds iff the patch is *cleanly
# applicable*. We use both to choose between apply / skip / fail-loudly.
if [[ -f "${PATCH_FILE}" ]]; then
    (
        cd "${REPO_DIR}"
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
    )
else
    echo "[setup] no patch file at ${PATCH_FILE}, skipping"
fi

# ----------------------------------------- LFS asset fetch (git-lfs-less)
# Cosmos's ``.gitattributes`` marks ``assets/**`` as git-lfs-tracked,
# so our git-lfs-less clone left them as 128B pointer stubs (text like
# ``version https://git-lfs.github.com/spec/v1\noid sha256:...\nsize
# ...``), not the real content. We don't want to install git-lfs in the
# container, so instead pull the three assets the parity-check inference
# cmds actually need from ``media.githubusercontent.com`` (which serves
# resolved LFS content for public GitHub repos without requiring a local
# git-lfs binary). ``raw.githubusercontent.com`` only ever returns the
# pointer text, hence the explicit ``media.`` host below.
LFS_BASE="https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/${PIN_COMMIT}"
download_if_pointer() {
    local relpath="$1"
    local path="${REPO_DIR}/${relpath}"
    # The pointer signature is fixed; detecting it makes this step
    # idempotent — on re-runs the file is already real content and we
    # short-circuit.
    if head -c 40 "${path}" 2>/dev/null | grep -q "git-lfs.github.com"; then
        echo "[setup] re-fetching LFS asset ${relpath}"
        curl -sSfL "${LFS_BASE}/${relpath}" -o "${path}"
    else
        echo "[setup] LFS asset ${relpath} already resolved, skipping"
    fi
}
download_if_pointer assets/base/robot_welding.json
download_if_pointer assets/base/robot_welding.jpg
download_if_pointer assets/base/robot_welding.txt

# ------------------------------------------------------------------- sync + run
# From here on we operate inside ``cosmos-predict2.5/`` so ``uv`` picks
# up *its* ``pyproject.toml`` (which knows about its own workspace
# packages ``cosmos-oss`` / ``cosmos-cuda`` / ``cosmos-gradio`` and the
# NVIDIA custom indexes that host prebuilt flash-attn / decord /
# transformer-engine / natten wheels). The venv lands at
# ``cosmos-predict2.5/.venv`` (default ``UV_PROJECT_ENVIRONMENT``).
cd "${REPO_DIR}"

# ``uv sync`` is no-op-fast once the venv matches the lockfile. The
# ``--extra=cu130`` flag activates the ``cu130 = ["cosmos-oss[cu130_torch29]"]``
# extra at the cosmos-predict2.5 level, which transitively pulls the
# torch 2.9.1 + cu130 wheel set from
# ``https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu130_torch29/simple``.
echo "[setup] ensuring cosmos deps via 'uv sync --extra=cu130'"
uv sync --extra=cu130

# ----------------------------------------------------------- inference T2V
echo "[run] T2V: assets/base/robot_welding.json -> outputs/base_text2world"
export COSMOS_PARITY_STATS_PATH="outputs/base_text2world/step_stats.json"
export FORCE_CUDNN_ATTN=1
export COSMOS_FORCE_TORCH_ATTN=1
export COSMOS_FORCE_TORCH_COMPILE=1
export COSMOS_TORCH_COMPILE_MODE=default
uv run --extra=cu130 python examples/inference.py \
    -i assets/base/robot_welding.json \
    -o outputs/base_text2world \
    --inference-type=text2world \
    --model=2B/post-trained \
    --disable-guardrails \
    --num-output-frames=93 \
    --resolution=720,1280

# ----------------------------------------------------------- inference I2V (disabled for apples-to-apples T2V perf)
# echo "[run] I2V: assets/base/robot_welding.json -> outputs/base_image2world"
# export COSMOS_PARITY_STATS_PATH="outputs/base_image2world/step_stats.json"
# uv run --extra=cu130 python examples/inference.py \
#     -i assets/base/robot_welding.json \
#     -o outputs/base_image2world \
#     --inference-type=image2world \
#     --model=2B/post-trained \
#     --disable-guardrails \
#     --num-output-frames=93 \
#     --resolution=720,1280
