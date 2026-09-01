#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASHDREAMS_ROOT="$HERE/../.."

echo "=========================================="
echo "LingBot WebRTC - Complete Working Setup"
echo "=========================================="
echo ""

# Step 1: Clean everything
echo "[1/7] Cleaning old setup..."
cd "$HERE"
rm -rf .venv
find "$FLASHDREAMS_ROOT" -type d -name __pycache__ -delete 2>/dev/null || true
find "$FLASHDREAMS_ROOT" -type d -name "*.egg-info" -delete 2>/dev/null || true

# Step 2: Create venv with Python
echo "[2/7] Creating Python venv..."
python -m venv .venv
source .venv/bin/activate

# Step 3: Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 4: Install torch/torchvision with CUDA 13.2. torchaudio is
# installed separately in step 5b -- no released torchaudio build
# supports CUDA 13.2 yet (upstream release lag), so it can't be pinned
# alongside torch/torchvision here.
echo "[4/7] Installing PyTorch with CUDA 13.2..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132 --force-reinstall

# Step 5: Install all dependencies
echo "[5/7] Installing dependencies..."
pip install "transformers>=5.0,<6" sentencepiece scipy opencv-python
pip install aiohttp aiortc python-multipart loguru
# tyro 1.0.16 regresses the SuppressFixed subcommand-union path that
# flashdreams-run's single-runner CLI parser depends on (raises "Field
# runner is marked as Fixed or Suppress but is missing a default value");
# 1.0.15 builds the same parser cleanly.
pip install "tyro==1.0.15" pydantic fastapi uvicorn pillow numpy gradio websockets
pip install flash-attn==2.6.3 --no-build-isolation 2>/dev/null || pip install flash-attn==2.6.3 --only-binary :all: 2>/dev/null || true

# Step 5b: Install a CPU-only torchaudio. LingBot never uses audio --
# torchaudio is only imported transitively (transformers.audio_utils
# unconditionally does ``import torchaudio``, pulled in by every model's
# modeling file via processing_utils/modeling_layers). No released
# torchaudio build supports CUDA 13.2 yet, and any CUDA-linked build
# hard-crashes at import time if its baked-in CUDA version doesn't match
# torch's (``_check_cuda_version()`` in torchaudio/_extension/utils.py),
# surfacing through transformers as a misleading generic
# "Could not import module 'X'" error. A genuine CPU-only build has no
# CUDA extension, so that check never runs -- zero functional loss since
# the actual model still runs entirely on the CUDA torch install.
echo "[5b/7] Installing CPU-only torchaudio (audio unused, avoids CUDA-version mismatch crash)..."
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps --no-cache-dir
python -c "
import torch, torchaudio
print(f'torch {torch.__version__} / torchaudio {torchaudio.__version__} OK')
"

# Step 6: Install flashdreams + lingbot editable, without disturbing the
# pinned torch/torchaudio/transformers/tyro versions above. Uninstall any
# stale non-editable copy first -- a prior plain "pip install flashdreams"
# leaves a static site-packages copy that silently shadows the editable
# install and source edits stop taking effect.
echo "[6/7] Installing flashdreams + lingbot (editable)..."
pip uninstall -y flashdreams flashdreams-lingbot flashdreams-cam2v >/dev/null 2>&1 || true
pip install -e "$FLASHDREAMS_ROOT/flashdreams" --no-deps
pip install -e "$FLASHDREAMS_ROOT/apps/cam2v" --no-deps
pip install -e "$HERE" --no-deps

# Step 7: Clear caches
echo "[7/7] Clearing caches..."
find "$HERE" -type d -name __pycache__ -delete 2>/dev/null || true
cd "$HERE"

echo ""
echo "=========================================="
echo "✓✓✓ SETUP COMPLETE ✓✓✓"
echo "=========================================="
echo ""
echo "To run WebRTC server:"
echo "  cd $HERE"
echo "  bash run.sh"
echo ""
