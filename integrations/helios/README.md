# Helios × FlashDreams

First-party FlashDreams integration for [Helios](https://github.com/PKU-YuanGroup/Helios) 14B real-time streaming video generation (closes [flashdreams#276](https://github.com/NVIDIA/flashdreams/issues/276)).

## Runner slugs

| Slug | Checkpoint | Notes |
|------|------------|-------|
| `helios-distilled-t2v-14b` | `BestWishYsh/Helios-Distilled` | Fastest — pyramid `[2,2,2]`, no CFG |
| `helios-base-t2v-14b` | `BestWishYsh/Helios-Base` | Highest quality — pyramid `[20,20,20]`, CFG 5.0 |
| `helios-distilled-t2v-14b-2gpu` | `BestWishYsh/Helios-Distilled` | Ulysses context parallelism (`torchrun`) |

Helios generates in **33-frame chunks** natively. Each FlashDreams `generate()` call produces one chunk and yields decoded pixels immediately.

## Install

```bash
# from the repo root
export HF_TOKEN=<your-hf-token>

# HeliosPyramidPipeline requires a recent diffusers build
pip install git+https://github.com/huggingface/diffusers.git

uv sync --project integrations/helios
```

## Run

```bash
uv run --project integrations/helios \
    flashdreams-run helios-distilled-t2v-14b \
    --prompt "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot" \
    --total-blocks 3 \
    --pixel-height 384 --pixel-width 640
```

Multi-GPU (2× H100):

```bash
torchrun --nproc_per_node=2 --no-python flashdreams-run helios-distilled-t2v-14b-2gpu \
  --total-blocks 8
```

## Tests

```bash
uv run pytest integrations/helios/tests/test_smoke.py -v

# GPU benchmark (requires CUDA + model weights)
python integrations/helios/tests/benchmark/run_benchmark.py --mode all
```

See the [Helios model page](https://nvidia.github.io/flashdreams/main/models/helios.html) for full setup notes.
