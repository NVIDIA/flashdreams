#!/bin/bash
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
# -----------------------------------------------------------------------------
# build_with_docker.sh -- Build & push the flashdreams base image (multi-arch)
# -----------------------------------------------------------------------------
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Builds `docker/Dockerfile` for both linux/arm64 and linux/amd64 in a single
# buildx invocation and pushes the resulting manifest list to the target
# registry/tag(s) you specify.
#
# USAGE
# -----
#   bash docker/build_with_docker.sh REGISTRY/IMAGE:TAG [REGISTRY/IMAGE:TAG ...]
#
# At least one fully-qualified image tag is required. Multiple tags are
# supported (e.g. to push to more than one registry at once).
#
# Examples:
#   bash docker/build_with_docker.sh ghcr.io/myorg/flashdreams:latest
#   bash docker/build_with_docker.sh myregistry.example.com/flashdreams:v1.0
#   bash docker/build_with_docker.sh reg1/img:tag reg2/img:tag
#
# PREREQUISITES
# -------------
#   1. Docker with Buildx (docker buildx version).
#   2. A Buildx builder capable of both linux/amd64 and linux/arm64.
#      A default local builder can use QEMU emulation for the non-native arch;
#      configure remote Buildx nodes separately if you want native builders.
#   3. You are logged in to your target container registry:
#          docker login <registry>
#   4. The working directory is the repo root (the build context is ".") and
#      `docker/Dockerfile` is reachable. Invoke as:
#          bash docker/build_with_docker.sh REGISTRY/IMAGE:TAG
#
# To build without pushing (for local testing), edit this script to replace
# `--push` with `--load` -- note that `--load` is single-arch only, so you
# will also need to drop one of the `--platform` values.
#
# FLAG NOTES
# ----------
#   --platform linux/arm64,linux/amd64
#       Produce a multi-arch manifest so consumers on arm64 and amd64 can
#       pull the same tag.
#
#   --allow network.host + --network host
#       Allows the build to use the host's configured network path.
#
#   --push
#       Upload the resulting images and manifest list directly to the
#       registry. Implies no local `docker images` entry for this build.
# -----------------------------------------------------------------------------

set -eu -o pipefail

if [[ $# -eq 0 ]]; then
    echo "Error: at least one target image tag is required." >&2
    echo "" >&2
    echo "Usage: bash docker/build_with_docker.sh REGISTRY/IMAGE:TAG [REGISTRY/IMAGE:TAG ...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  bash docker/build_with_docker.sh ghcr.io/myorg/flashdreams:latest" >&2
    echo "  bash docker/build_with_docker.sh myregistry.example.com/flashdreams:v1.0" >&2
    exit 1
fi

TAG_ARGS=()
for tag in "$@"; do
    TAG_ARGS+=(-t "$tag")
done

docker buildx build \
    --platform linux/arm64,linux/amd64 \
    --allow network.host \
    --network host \
    --push \
    "${TAG_ARGS[@]}" \
    -f docker/Dockerfile .
