#!/usr/bin/env python3
"""Pre-download all HuggingFace models needed for flashdream_public."""
import os
import sys
from pathlib import Path

# Set HF cache to ensure downloads go to the right place
os.environ['HF_HOME'] = os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))

print(f"[DOWNLOAD] HF_HOME = {os.environ['HF_HOME']}")
print("[DOWNLOAD] This will download ~50-100 GB of models (takes 1-2 hours)")
print()

models_to_download = [
    # OmniDreams world model
    "nvidia/Cosmos-Reason1-7B",
    "nvidia/Cosmos-Reason1-IFT-7B",

    # FlashDreams inference models
    "nvidia/Cosmos-1-Diffusion-7B-Text2World",
    "nvidia/Cosmos-1-Diffusion-7B-Video2World",

    # VAE/encoding models
    "stabilityai/sd-vae-ft-mse",
    "openai/clip-vit-large-patch14",
]

print(f"[DOWNLOAD] Models to download ({len(models_to_download)}):")
for model in models_to_download:
    print(f"  - {model}")
print()

try:
    from huggingface_hub import snapshot_download

    total_size = 0
    for i, model in enumerate(models_to_download, 1):
        print(f"[DOWNLOAD] [{i}/{len(models_to_download)}] Downloading {model}...")
        sys.stdout.flush()

        try:
            path = snapshot_download(
                model,
                cache_dir=os.environ['HF_HOME'],
                resume_download=True,
                local_files_only=False,
            )
            print(f"[DOWNLOAD] ✓ {model} cached at {path}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[DOWNLOAD] ⚠ {model} failed: {type(e).__name__}: {e}")
            sys.stdout.flush()
            continue

    print()
    print("="*70)
    print("[DOWNLOAD] ✓ Model download complete!")
    print("[DOWNLOAD] Now run: .\setup.bat")
    print("="*70)

except ImportError:
    print("[ERROR] huggingface_hub not installed")
    print("[ERROR] Run: pip install huggingface_hub")
    sys.exit(1)
