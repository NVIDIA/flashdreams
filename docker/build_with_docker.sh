#!/bin/bash
# -----------------------------------------------------------------------------
# build_with_docker.sh — Build & push the flashdreams base image (multi-arch)
# -----------------------------------------------------------------------------
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Builds `docker/Dockerfile` for both linux/arm64 and linux/amd64 in a single
# buildx invocation and pushes the resulting manifest list to the GitHub
# Container Registry (GHCR).
#
# Tag scheme:
#     ghcr.io/nvidia/flashdreams:<TAG>-<YYYYMMDD>-<git-sha>
#
# The date + short git SHA make every build uniquely addressable, while the
# TAG prefix (e.g. "base-v0.3") tracks the Dockerfile's major revision.
#
# PREREQUISITES
# -------------
#   1. Docker with Buildx (docker buildx version).
#   2. A buildx builder capable of both linux/amd64 and linux/arm64 nodes.
#      The companion script `docker_farm_setup.sh` sets one up named "farm",
#      using the shared NVIDIA DGX Spark host (aliased "dgx-spark" in
#      ~/.ssh/config) as the native arm64 node. If you skip that, buildx
#      falls back to QEMU emulation for the non-native arch — works but
#      significantly slower.
#      To request access to dgx-spark (SSH config, account), contact
#      qiwu@nvidia.com.
#   3. You are logged in to the GitHub Container Registry:
#          docker login ghcr.io
#      (use a GitHub personal access token with `write:packages` scope).
#   4. The working directory is the repo root (the build context is ".") and
#      `docker/Dockerfile` is reachable. Invoke as:
#          bash docker/build_with_docker.sh
#   5. The working tree is a git checkout (the tag embeds `git rev-parse --short HEAD`).
#
# HOW TO RUN
# ----------
#   bash docker/build_with_docker.sh
#
# To build without pushing (for local testing), drop `--push` and add
# `--load` instead — note that `--load` is single-arch only, so you'll also
# need to drop one of the `--platform` values.
#
# FLAG NOTES
# ----------
#   --platform linux/arm64,linux/amd64
#       Produce a multi-arch manifest so consumers on arm64 (DGX Spark,
#       Grace) and amd64 (standard GPU workstations, IPP5) can pull the
#       same tag.
#
#   --allow network.host + --network host
#       Required inside NVIDIA infra so the build can reach internal apt
#       mirrors / PyPI proxies / the GitLab registry without a bridged
#       network. Not a secret-leak risk here because the build context is
#       public-to-the-company.
#
#   --push
#       Upload the resulting images and manifest list directly to the
#       registry. Implies no local `docker images` entry for this build.
#
# TAG BUMP CHECKLIST
# ------------------
# When cutting a new base image (e.g. after a Dockerfile or dependency
# change), bump TAG below (base-v0.3 -> base-v0.4), commit, then run this
# script. Update the container references in the top-level README.md so
# downstream users know which tag to pull.
# -----------------------------------------------------------------------------

set -eu -o pipefail

TAG=base-v0.3-$(date +%Y%m%d)-$(git rev-parse --short HEAD)

REGISTRIES=${@:-ghcr.io/nvidia/flashdreams}
REGISTRIES_ARGS=()
for registry in $REGISTRIES; do
    REGISTRIES_ARGS+=(-t $registry:$TAG)
done

docker buildx build \
    --platform linux/arm64,linux/amd64 \
    --allow network.host \
    --network host \
    --push \
    "${REGISTRIES_ARGS[@]}" \
    -f docker/Dockerfile .
