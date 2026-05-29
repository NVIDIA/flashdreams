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

# Run HY-WorldPlay WAN-5B I2V video generation end-to-end via Docker.
#
# Boots the flashdreams container, lazily provisions the parity sub-venv
# and upstream tree + ~52 GB of model checkpoints on first run, then
# invokes ``flashdreams-run hy-worldplay-wan-i2v-5b`` with the
# configured prompt / image / pose / chunk-count knobs. Reuses the same
# env-var contract as
# ``integrations/hy_worldplay/tests/parity_check/run.sh``.
#
# Usage:
#   ./integrations/hy_worldplay/run-docker.sh
#
#   IMAGE_PATH=/devwork/flashdreams/cat_surf.jpg \
#   PROMPT="First-person view of a cat surfing..." \
#   NUM_CHUNK=8 POSE='w-31' \
#       ./integrations/hy_worldplay/run-docker.sh
#
#   NUM_GPU=4 ./integrations/hy_worldplay/run-docker.sh   # multi-GPU via torchrun
#
# Env vars (with defaults):
#   FLASHDREAMS_IMAGE  Docker image with the prebuilt flashdreams env.
#   REPO_HOST_PATH     Host path to the flashdreams repo. Defaults to
#                      this script's grandparent dir.
#   HF_TOKEN           Required. HuggingFace token with read access to
#                      ``tencent/HY-WorldPlay``.
#   NUM_GPU            GPU count. 1 -> direct invocation; >=2 -> torchrun.
#   IMAGE_PATH         Host path to the first-frame RGB image. Defaults
#                      to the upstream test image once provisioning is done.
#   PROMPT             Inference text prompt.
#   NUM_CHUNK          Autoregressive chunk count (1 chunk ~= 1 s @ 16 fps).
#   POSE               Camera pose trajectory string. The parser prepends
#                      an identity pose for the input frame, so ``w-N``
#                      produces ``N + 1`` latents; pick ``N == num_chunk * 4 - 1``
#                      (e.g. ``w-3`` for ``num_chunk=1``, ``w-31`` for ``num_chunk=8``).
#   SEED               RNG seed.
#   OUTPUT_SUBDIR      Subdirectory under ``outputs/`` for the rendered
#                      .mp4 + per-chunk stats. Defaults to a timestamped
#                      ``hy-worldplay/YYYYMMDD-HHMMSS`` slot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- config
FLASHDREAMS_IMAGE="${FLASHDREAMS_IMAGE:-ghcr.io/nvidia/flashdreams:base-v0.3-20260424-55bd566}"
REPO_HOST_PATH="${REPO_HOST_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# Mount the repo's *parent* at /workspace so the in-container layout
# mirrors the existing /devwork:/workspace convention sibling teammates
# already use (cf. terminals/1.txt). REPO_HOST_PATH itself lands at
# /workspace/<basename> regardless of where the host checkout lives.
REPO_HOST_PARENT="$(dirname "${REPO_HOST_PATH}")"
REPO_BASENAME="$(basename "${REPO_HOST_PATH}")"
CONTAINER_REPO="/workspace/${REPO_BASENAME}"

PARITY_REL="integrations/hy_worldplay/tests/parity_check"
HY_TREE_REL="${PARITY_REL}/HY-WorldPlay"
HF_MODELS_REL="${HY_TREE_REL}/hf_models"

NUM_GPU="${NUM_GPU:-1}"
NUM_CHUNK="${NUM_CHUNK:-1}"
POSE="${POSE:-w-3}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-First-person view walking around ancient Athens, with Greek architecture and marble structures}"
IMAGE_PATH="${IMAGE_PATH:-${REPO_HOST_PATH}/${HY_TREE_REL}/assets/img/test.png}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-hy-worldplay/$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---------------------------------------------------------------- preflight
if [[ -z "${HF_TOKEN:-}" ]]; then
    cat >&2 <<'EOF'
[error] HF_TOKEN env var is required.
        Get a HuggingFace token with read access to tencent/HY-WorldPlay
        at https://huggingface.co/settings/tokens, then:

          export HF_TOKEN=hf_...

EOF
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "[error] docker is not on PATH" >&2
    exit 1
fi

if [[ ! -d "${REPO_HOST_PATH}/integrations/hy_worldplay" ]]; then
    echo "[error] flashdreams repo not found at REPO_HOST_PATH=${REPO_HOST_PATH}" >&2
    exit 1
fi

# Marker files that indicate the parity sub-venv + HF weights are already
# in place. When missing, the first-run setup inside the container clones
# the upstream tree, downloads weights (~52 GB), and syncs the sub-venv.
WAN_TRANSFORMER_CONFIG="${REPO_HOST_PATH}/${HF_MODELS_REL}/wan_transformer/config.json"
WAN_DISTILLED_CKPT="${REPO_HOST_PATH}/${HF_MODELS_REL}/wan_distilled_model/model.pt"
NEEDS_PROVISION=0
if [[ ! -f "${WAN_TRANSFORMER_CONFIG}" || ! -f "${WAN_DISTILLED_CKPT}" ]]; then
    NEEDS_PROVISION=1
fi

# Resolve IMAGE_PATH into a container-side path. If the image lives under
# the workspace parent (which we already bind-mount to /workspace), use
# the natural in-container path; otherwise add a read-only -v mount of
# the image's parent dir at /inputs.
HOST_IMAGE_PATH="$(realpath -m "${IMAGE_PATH}")"
EXTRA_MOUNT=()
if [[ "${HOST_IMAGE_PATH}" == "${REPO_HOST_PARENT}/"* ]]; then
    CONTAINER_IMAGE_PATH="/workspace/${HOST_IMAGE_PATH#"${REPO_HOST_PARENT}/"}"
else
    IMAGE_DIR_HOST="$(dirname "${HOST_IMAGE_PATH}")"
    IMAGE_BASENAME="$(basename "${HOST_IMAGE_PATH}")"
    EXTRA_MOUNT+=("-v" "${IMAGE_DIR_HOST}:/inputs:ro")
    CONTAINER_IMAGE_PATH="/inputs/${IMAGE_BASENAME}"
fi

# When provisioning isn't needed yet we still want a useful error if the
# user-supplied IMAGE_PATH doesn't exist (would surface as a confusing
# diffusers PIL error inside the container otherwise). Skip the check when
# provisioning is pending because the default IMAGE_PATH lives inside the
# upstream tree that hasn't been cloned yet.
if [[ "${NEEDS_PROVISION}" == "0" && ! -f "${HOST_IMAGE_PATH}" ]]; then
    echo "[error] IMAGE_PATH does not exist on host: ${HOST_IMAGE_PATH}" >&2
    exit 1
fi

# ---------------------------------------------------------------- run summary
echo "[run] image=${FLASHDREAMS_IMAGE}"
echo "[run] repo=${REPO_HOST_PATH}"
echo "[run] gpus=${NUM_GPU} | num_chunk=${NUM_CHUNK} | pose=${POSE} | seed=${SEED}"
echo "[run] image_path=${HOST_IMAGE_PATH} -> ${CONTAINER_IMAGE_PATH}"
echo "[run] output_subdir=outputs/${OUTPUT_SUBDIR}"
if [[ "${NEEDS_PROVISION}" == "1" ]]; then
    echo "[run] first-time setup: will clone HY-WorldPlay, download ~52 GB"
    echo "      of model weights, and sync the parity sub-venv. This is"
    echo "      slow (10-30 min depending on bandwidth) but idempotent --"
    echo "      subsequent runs skip the whole block."
fi

# ---------------------------------------------------------------- docker run
# All host->container var passing goes through ``-e`` so the heredoc body
# stays single-quoted (no host-side $-expansion surprises with prompts
# that contain $/`/", etc).
docker run --rm \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
    -v "${REPO_HOST_PARENT}:/workspace" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${HOME}/.cache/flashdreams:/root/.cache/flashdreams" \
    "${EXTRA_MOUNT[@]}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e FLASHDREAMS_CACHE_DIR=/root/.cache/flashdreams \
    -e HY_IMAGE_PATH="${CONTAINER_IMAGE_PATH}" \
    -e HY_PROMPT="${PROMPT}" \
    -e HY_NUM_CHUNK="${NUM_CHUNK}" \
    -e HY_POSE="${POSE}" \
    -e HY_SEED="${SEED}" \
    -e HY_NUM_GPU="${NUM_GPU}" \
    -e HY_OUTPUT_SUBDIR="${OUTPUT_SUBDIR}" \
    -e HY_NEEDS_PROVISION="${NEEDS_PROVISION}" \
    -e HY_CONTAINER_REPO="${CONTAINER_REPO}" \
    -e HY_EXTRA_ARGS="${EXTRA_ARGS}" \
    -w "${CONTAINER_REPO}" \
    "${FLASHDREAMS_IMAGE}" bash -lc '
set -euo pipefail

PARITY=integrations/hy_worldplay/tests/parity_check
# The container runs as root but the repo is bind-mounted from the host
# user. Without this, git inside `parity_check/run.sh` errors with
# "fatal: detected dubious ownership in repository at ${HY_CONTAINER_REPO}".
git config --global --add safe.directory "${HY_CONTAINER_REPO}" >/dev/null 2>&1 || true
git config --global --add safe.directory "${HY_CONTAINER_REPO}/${PARITY}/HY-WorldPlay" >/dev/null 2>&1 || true

# One-time provisioning: clone upstream, download weights, sync sub-venv.
# parity_check/run.sh is idempotent and also runs the upstream baseline
# benchmark at the end -- that "wastes" a few minutes on the first run
# but is the cheapest way to keep this script in sync with parity setup.
if [[ "${HY_NEEDS_PROVISION}" == "1" ]]; then
    echo "[setup] one-time HY-WorldPlay provisioning"
    bash "${PARITY}/run.sh"
fi

mkdir -p "outputs/${HY_OUTPUT_SUBDIR}"

echo "[infer] starting flashdreams-run hy-worldplay-wan-i2v-5b (gpus=${HY_NUM_GPU})"
read -r -a EXTRA_ARG_ARRAY <<<"${HY_EXTRA_ARGS}"
if [[ "${HY_NUM_GPU}" -le 1 ]]; then
    uv run --project "${PARITY}" flashdreams-run hy-worldplay-wan-i2v-5b \
        --image-path "${HY_IMAGE_PATH}" \
        --prompt "${HY_PROMPT}" \
        --ckpt-path "${PARITY}/HY-WorldPlay/hf_models/wan_distilled_model/model.pt" \
        --num-chunk "${HY_NUM_CHUNK}" \
        --pose "${HY_POSE}" \
        --seed "${HY_SEED}" \
        --output-dir "outputs/${HY_OUTPUT_SUBDIR}" \
        "${EXTRA_ARG_ARRAY[@]}"
else
    uv run --project "${PARITY}" torchrun \
        --nproc_per_node="${HY_NUM_GPU}" --no-python \
        flashdreams-run hy-worldplay-wan-i2v-5b \
        --image-path "${HY_IMAGE_PATH}" \
        --prompt "${HY_PROMPT}" \
        --ckpt-path "${PARITY}/HY-WorldPlay/hf_models/wan_distilled_model/model.pt" \
        --num-chunk "${HY_NUM_CHUNK}" \
        --pose "${HY_POSE}" \
        --seed "${HY_SEED}" \
        --output-dir "outputs/${HY_OUTPUT_SUBDIR}" \
        "${EXTRA_ARG_ARRAY[@]}"
fi

echo "[done] wrote outputs/${HY_OUTPUT_SUBDIR}/hy-worldplay-wan-i2v-5b.mp4"
'

echo "[host] generation complete."
echo "[host] mp4: ${REPO_HOST_PATH}/outputs/${OUTPUT_SUBDIR}/hy-worldplay-wan-i2v-5b.mp4"
