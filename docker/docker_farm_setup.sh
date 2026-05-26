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
# -----------------------------------------------------------------------------
# docker_farm_setup.sh — Create a Docker Buildx "farm" for multi-platform builds
# -----------------------------------------------------------------------------
#
# REFERENCES:
#   - https://oneuptime.com/blog/post/2026-02-08-how-to-set-up-a-docker-build-farm-with-multiple-builders/view
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Creates a Buildx builder named "farm" that has multiple backends (nodes):
#   1. Your current machine — for linux/amd64 (and optionally arm64 via emulation)
#   2. dgx-spark (remote) — for native linux/arm64 builds
#
# Buildx will then use the right node for the platform you request (e.g. when you
# pass --platform linux/arm64 it can run the build on dgx-spark instead of
# slow QEMU emulation on an amd64 host).
#
# ABOUT dgx-spark
# ---------------
# "dgx-spark" is the short host alias (configured in your ~/.ssh/config) for a
# shared NVIDIA DGX Spark workstation used by this project as a native arm64
# (Grace) build node. The script assumes:
#   - You have an account on the machine.
#   - `ssh dgx-spark` resolves and logs in non-interactively (SSH key in place,
#     Host entry in ~/.ssh/config) as the same $USER that runs this script.
#   - Docker is installed and runnable by your user on that host.
#
# To request access / get the SSH Host block and onboarding steps, contact
#   qiwu@nvidia.com
#
# If you do not need native arm64 builds, skip this script entirely — buildx
# will emulate arm64 via QEMU on your amd64 workstation (slower, but works).
#
# PREREQUISITES
# -------------
#   - Docker with Buildx (docker buildx version).
#   - SSH access to dgx-spark as the same user you run this script as ($USER).
#     Test with:  ssh dgx-spark true
#     If this fails, see the "ABOUT dgx-spark" section above for access help.
#   - If "farm" already exists and you want a clean setup, remove it first:
#       docker buildx use default
#       docker buildx rm farm
#
# HOW TO RUN
# ----------
#   From the repo root:
#     bash docker/docker_farm_setup.sh
#   Or from this directory:
#     ./docker_farm_setup.sh
#
# This is a one-time setup per workstation. After it completes you can
# invoke `bash docker/build_with_docker.sh` to produce multi-arch images
# that use both nodes automatically.
#
# AFTER SETUP
# ----------
#   - The script sets "farm" as the default builder. Your normal build commands
#     then use the farm; pass --platform to choose arch:
#
#     docker buildx build --platform linux/amd64 -t myimg:amd64 -f Dockerfile .
#     docker buildx build --platform linux/arm64 -t myimg:arm64 -f Dockerfile .
#
#   - List builders and see nodes:
#     docker buildx ls
#     docker buildx inspect farm
#
#   - Switch back to the default Docker builder:
#     docker buildx use default
#
#   - Remove the farm builder (after switching away):
#     docker buildx use default && docker buildx rm farm
#
# -----------------------------------------------------------------------------
set -e

# Create a new builder named "farm" with current machine as first node.
# No endpoint = use local Docker daemon. linux/amd64 is typical for dev workstations.
docker buildx create \
  --name farm \
  --driver docker-container \
  --platform linux/amd64

# Add dgx-spark as a second node for native linux/arm64 builds.
# --append adds this endpoint to the existing "farm" builder.
# Builds for --platform linux/arm64 will be scheduled on this node when available.
docker buildx create \
  --name farm \
  --append \
  --platform linux/arm64 \
  --driver-opt network=host \
  ssh://$USER@dgx-spark

# Start the builder containers (local and on dgx-spark) so they are ready for builds.
docker buildx inspect farm --bootstrap

# Use "farm" as the default builder so subsequent buildx build commands use it.
docker buildx use farm

# Other useful commands
#  docker buildx ls
#  docker buildx inspect farm
#  docker buildx use default
#  docker buildx rm farm
