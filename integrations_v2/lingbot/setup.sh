#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up a plain venv + pip environment for the LingBot v2 Cam2V app.
#
# Deliberately does NOT use uv -- everything here is `python3 -m venv` plus
# pip, so it works on boxes where uv isn't installed or the workspace
# resolution gets in the way.
#
#   bash setup.sh                    # create .venv and install everything
#   TORCH_INDEX=cu130 bash setup.sh  # different CUDA wheel index
#   KEEP_VENV=1 bash setup.sh        # reuse an existing .venv
#
# Then run with: bash run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"
TORCH_INDEX="${TORCH_INDEX:-cu132}"

# The repo root is a uv workspace manifest only ([tool.uv.workspace], no
# [project]/[build-system]), so it is NOT pip-installable -- installing it
# fails with "Multiple top-level packages discovered in a flat-layout".
# Every install below therefore targets a specific package directory.
if [ ! -f "$ROOT/flashdreams/pyproject.toml" ]; then
  echo "ERROR: expected repo root at $ROOT, but $ROOT/flashdreams/pyproject.toml is missing." >&2
  exit 1
fi

echo "=========================================="
echo "LingBot v2 (Cam2V) setup -- venv + pip"
echo "  repo root: $ROOT"
echo "  venv:      $VENV"
echo "  torch:     https://download.pytorch.org/whl/$TORCH_INDEX"
echo "=========================================="
echo

# --- 1. venv --------------------------------------------------------------
if [ -d "$VENV" ] && [ "${KEEP_VENV:-0}" != "1" ]; then
  echo "[1/7] Removing existing .venv (set KEEP_VENV=1 to reuse)..."
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  echo "[1/7] Creating venv..."
  # Debian/Ubuntu ship only python3, no bare `python`.
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else PYTHON_BIN=python; fi
  "$PYTHON_BIN" -m venv "$VENV"
else
  echo "[1/7] Reusing existing venv."
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 2. pip ---------------------------------------------------------------
echo "[2/7] Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# --- 3. torch -------------------------------------------------------------
# Installed from the CUDA index explicitly. A plain `pip install torch` pulls
# the CPU-only wheel, which fails at generation time with "Torch not compiled
# with CUDA enabled" rather than at install time. --force-reinstall because a
# stale CPU-only torch reads as "already satisfied" (pip's version matching
# ignores the +cu local build tag) and never gets replaced otherwise.
echo "[3/7] Installing torch + torchvision ($TORCH_INDEX)..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" --force-reinstall

# --- 4. torchaudio (CPU-only, on purpose) ---------------------------------
# LingBot never uses audio; torchaudio is only imported transitively
# (transformers.audio_utils does an unconditional `import torchaudio`). A
# CUDA-linked torchaudio hard-crashes at import when its baked-in CUDA
# version doesn't match torch's (_check_cuda_version() in
# torchaudio/_extension/utils.py), surfacing as a misleading
# "Could not import module 'X'". A CPU-only build ships no CUDA extension,
# so that check never runs -- zero functional loss, the model still runs on
# the CUDA torch installed above.
echo "[4/7] Installing CPU-only torchaudio..."
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps --no-cache-dir

# --- 5. pinned / hand-listed deps ----------------------------------------
# tyro 1.0.16 regresses the SuppressFixed subcommand-union path the CLI
# parser depends on ("Field runner is marked as Fixed or Suppress but is
# missing a default value"); 1.0.15 builds the same parser cleanly.
echo "[5/7] Installing pinned dependencies..."
pip install "tyro==1.0.15" "transformers>=5.0,<6" sentencepiece scipy opencv-python pillow numpy

# --- 6. flashdreams + cam2v + lingbot (editable) --------------------------
# Extras pull in the serving stack (aiohttp/aiortc) and runner deps rather
# than hand-listing them -- that list has drifted out of sync before and
# caused ModuleNotFoundError whack-a-mole at runtime.
#
# Uninstall stale non-editable copies first: a prior plain
# `pip install flashdreams` leaves a static site-packages copy that silently
# shadows the editable install, and source edits stop taking effect.
echo "[6/7] Installing flashdreams + cam2v + lingbot (editable)..."
pip uninstall -y flashdreams flashdreams-cam2v flashdreams-lingbot >/dev/null 2>&1 || true
pip install -e "$ROOT/flashdreams[local-window,runners,serving]"
pip install -e "$ROOT/apps/cam2v"
pip install -e "$HERE"

# flash-attn is an optional accelerator: the model runs without it, and its
# source build fails on boxes without a matching toolchain, so a failure here
# is reported but not fatal.
echo "      Trying optional flash-attn (non-fatal if it fails)..."
pip install flash-attn==2.6.3 --no-build-isolation || pip install flash-attn==2.6.3 --only-binary :all: || echo "      flash-attn not installed -- continuing without it."

# --- 7. verify ------------------------------------------------------------
echo "[7/7] Verifying..."
python -c "import torch, torchaudio; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} / torchaudio {torchaudio.__version__}')"
python -c "import lingbot.config as c; print('lingbot pipelines:', ', '.join(sorted(c.PIPELINE_CONFIGS)))"
command -v flashdreams-run-v2 >/dev/null && echo "flashdreams-run-v2: $(command -v flashdreams-run-v2)"

echo
echo "=========================================="
echo "Setup complete."
echo "=========================================="
echo "Run the WebRTC demo with:"
echo "  bash $HERE/run.sh"
echo
echo "Export HF_TOKEN first if the selected checkpoint repo needs auth."
