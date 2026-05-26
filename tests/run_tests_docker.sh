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
#
# Run the flashdreams test suite inside a fresh NVIDIA PyTorch docker container.
#
# Use this on a local machine that has docker and at least one GPU.
# The script installs flashdreams + integration packages on the fly, then
# invokes pytest. Caches for uv / huggingface / flashdreams are bind-mounted
# from the host so subsequent runs are fast.
#
# Usage:
#   ./tests/run_tests_docker.sh [TEST_TARGET...]
#
# Environment overrides:
#   FLASHDREAMS_TEST_IMAGE         (default: nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04)
#   FLASHDREAMS_UV_CACHE_DIR       (default: ${HOME}/.cache/uv)
#   FLASHDREAMS_HF_CACHE_DIR       (default: ${HOME}/.cache/huggingface)
#   FLASHDREAMS_CACHE_DIR          (default: ${HOME}/.cache/flashdreams)
#   FLASHDREAMS_TRITON_CACHE_DIR   (default: ${HOME}/.cache/triton)
#
# Examples:
#   # Run all tests
#   ./tests/run_tests_docker.sh
#
#   # Run a specific test file
#   ./tests/run_tests_docker.sh flashdreams/tests/test_attention.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHDREAMS_TEST_IMAGE:-nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04}"

UV_CACHE_HOST="${FLASHDREAMS_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHDREAMS_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHDREAMS_CACHE_HOST="${FLASHDREAMS_CACHE_DIR:-${HOME}/.cache/flashdreams}"
TRITON_CACHE_HOST="${FLASHDREAMS_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHDREAMS_CACHE_HOST}" "${TRITON_CACHE_HOST}"

docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/flashdreams" \
    -v "${UV_CACHE_HOST}:/root/.cache/uv" \
    -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
    -v "${FLASHDREAMS_CACHE_HOST}:/root/.cache/flashdreams" \
    -v "${TRITON_CACHE_HOST}:/root/.cache/triton" \
    -v "${HOME}/.netrc:/root/.netrc:ro" \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRITON_CACHE_DIR=/root/.cache/triton \
    -e UV_LINK_MODE=copy \
    -e UV_PROJECT_ENVIRONMENT=/tmp/flashdreams-venv \
    -w /workspace/flashdreams \
    "${IMAGE}" \
    bash -s -- "$@" <<'EOF'
set -euo pipefail

# -- bootstrap: install system deps + uv into the raw CUDA base image --------
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  python3 python3-dev python3-venv ffmpeg gcc g++ ninja-build \
  libnccl-dev curl git ca-certificates unzip
rm -rf /var/lib/apt/lists/*
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# UV_PROJECT_ENVIRONMENT is set via docker -e so the venv lives outside the
# bind-mounted workspace, avoiding root-owned .venv on the host.
uv venv --clear
uv sync --frozen --extra dev

exec bash /workspace/flashdreams/tests/run_tests_local.sh "$@"
EOF
