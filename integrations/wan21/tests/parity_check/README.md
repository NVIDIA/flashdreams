# Wan2.1 official parity check

This test runs the official Wan2.1 repository with the same command-line setup
from upstream README for `t2v-1.3B`, while forcing the cuDNN SDPA backend and `torch.compile`.

## What this harness does

1. Clones `Wan-Video/Wan2.1` and checks out a pinned commit.
2. Applies `changes.patch` (cuDNN SDPA enforcement in fallback attention path).
3. Creates an isolated `.venv` and installs upstream requirements.
4. Installs `huggingface_hub[cli]`.
5. Installs local `flashdreams` editable package with `--no-deps` for environment alignment.
6. Downloads `Wan-AI/Wan2.1-T2V-1.3B` if missing.
7. Runs:

```bash
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

The upstream official code path already uses `tqdm` over diffusion timesteps in `wan/text2video.py`.

## Run

```bash
bash integrations/wan21/tests/parity_check/run.sh
```
