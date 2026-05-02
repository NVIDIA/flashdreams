# FlashDreams

## Environment setup

Install all workspace packages (flashdreams core + every integration) into a venv:

```bash
uv sync --extra dev --group lint
```

Then run commands with `uv run` (auto-activates the venv):

```bash
uv run pytest flashdreams/tests
uv run --package flashdreams --extra examples flashdreams/examples/run_alpadreams.py --help
```

## Development

### Linting and type checking

The project uses [ruff](https://docs.astral.sh/ruff/) for formatting/linting and
[ty](https://docs.astral.sh/ty/) for static type checking, both enforced via
pre-commit:

```bash
# Run all checks (ruff + ty)
uv run pre-commit run -a

# Run ty type checker directly
uv run ty check
```

### IDE setup (VS Code)

Install the [ty extension](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty)
for VS Code.

To use ty as the primary language server (replaces Pylance):

```jsonc
// .vscode/settings.json
{
    "python.languageServer": "None"
}
```

To use ty only for type checking alongside Pylance (completions, hover, etc.):

```jsonc
// .vscode/settings.json
{
    "python.languageServer": "Pylance",
    "ty.disableLanguageServices": true
}
```

### Tests

```bash
# CPU-safe tests (excludes GPU-dependent tests)
uv run pytest -m "not manual"

# Single test file
uv run pytest flashdreams/tests/test_attention.py
```

## Instructions to run Alpadreams Inference

```bash
# 0. request interactive node with the pre-built container [IPP5 cluster as example].
# The image is a multi-arch manifest (linux/arm64 + linux/amd64); the runtime picks
# the right variant automatically. See `docker/README.md` for how it is built.
srun \
    --gpus-per-node=4 -q interactive --exclusive --nodes 1 --cpus-per-gpu 36 --pty \
    --partition=gtc_demo \
    --time=24:00:00  \
    --pty \
    --container-image=ghcr.io/nvidia/flashdreams:base-v0.3-20260424-55bd566 \
    --container-mounts=/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels,/home:/home,/cm:/cm,/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d \
    --container-remap-root \
    --container-mount-home \
    --container-writable \
    --container-workdir=$HOME/workspace/flashdreams \
    /bin/bash

# 1. setup credentials in the file `credentials/s3_checkpoint.secret` similarly with I4:
cat > credentials/s3_checkpoint.secret <<EOF
{
  "aws_access_key_id": "team-sil-videogen",
  "aws_secret_access_key": <YOUR-SIL-VIDEOGEN-PDX-KEY>,
  "endpoint_url": "https://pdx.s8k.io",
  "region_name": "us-east-1"
}
EOF

# 2. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 3. (optional) setup where to cache flashdreams checkpoints
export FLASHDREAMS_CACHE_DIR=~/.cache/flashdreams # default

# 4. Run inference script. Checkpoints and example data are auto-downloaded at first run.
# - single view on single GPU
# - add (--overwrite_config_name sv_2steps_chunk2_loc6_lightvae_lighttae_perf) for best perf.
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_alpadreams.py \
    --n_cameras 1 --total_blocks 20

# - multi view on 4 GPUs
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    flashdreams/examples/run_alpadreams.py \
    --n_cameras 4 --total_blocks 20

# - diffusion forcing AR model on bundled single-view example data
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    flashdreams/examples/run_alpadreams.py \
    --n_cameras 1 \
    --total_blocks 12 \
    --overwrite_config_name sv_35steps_chunk2_loc24_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m \
    --offload_text_encoder
```

## Instructions to run Alpadreams Bidirectional Model

Use the same container, S3 credential, Hugging Face token, and
`FLASHDREAMS_CACHE_DIR` setup as the Alpadreams inference section above. The
bidirectional recipe runs the single-view full-block Cosmos2 2B / 720p / HDMap
checkpoint; the checkpoint is configured in the recipe and downloaded from S3
on first use.

The runner uses the same bundled single-view Alpadreams example data as the
autoregressive demo and resizes inputs to the pixel resolution required by the
selected bidirectional config (`704x1280` for the default checkpoint).
The bidirectional recipe defaults to the checkpoint-trained 48 latent chunks
(189 decoded frames with the Wan decoder). If that is too large for your GPU
setup, pass `--num_chunks` to choose a smaller single-block length; for example,
`--num_chunks 24` yields 93 decoded frames.
Unlike the autoregressive demo, this bidirectional recipe generates one
full block per run, so keep `--total_blocks 1`.

```bash
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    flashdreams/examples/run_alpadreams.py \
    --total_blocks 1 \
    --num_chunks 24 \
    --overwrite_config_name sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m
```

Useful options:

- `--offload_text_encoder`: precomputes text and first-frame image embeddings,
  frees the one-shot encoders, then runs diffusion and decode. Use this when
  peak VRAM is tight.
- `--save_embeddings_path`: runs only the one-shot encoders, saves positive
  text, negative text, and first-frame image embeddings to a `.pt`, then exits.
- `--embeddings_path`: loads a `.pt` produced by `--save_embeddings_path` and
  skips loading the one-shot encoders during inference.
- `--num_chunks`: overrides the bidirectional recipe's latent chunk count. Omit
  it to use the checkpoint-trained default (`48`); lower it when runtime memory
  is tight.

The generated comparison video is written to
`outputs/{output_prefix}_{world_size}gpus.mp4`, where `output_prefix` is based
on the selected config name, with the HDMap condition stacked above the
generated RGB output. Per-step stats are saved under
`outputs/stats_{output_prefix}_{world_size}gpus.json` when profiling is
enabled by the selected config.

To convert an I4/SIL distributed checkpoint directory into the single-file
`.pt` format consumed by the FlashDreams bidirectional config:

```bash
uv run --package flashdreams --extra examples \
  python flashdreams/scripts/convert_i4_dcp2pt.py \
    --checkpoint_path s3://<bucket>/<path-to-dcp-checkpoint>/model \
    --credential_path credentials/s3_checkpoint.secret \
    --output_path checkpoints/output.pt
```

The converted `.pt` is saved pre-fusion: it preserves the training-time
padding-mask channel. Normal FlashDreams inference fuses the padding-mask
channel and output shuffle after loading the checkpoint through
`CosmosTransformer`. If you load the `.pt` directly into `CosmosDiTNetwork`,
call `update_parameters_after_loading_checkpoint()` before running forward.

## Instructions to run Self-forcing T2V Inference

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_causal_wan21.py \
    --total_blocks 7
```

## Instructions to run Causal-forcing T2V and I2V Inference

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
# - T2V
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_causal_wan21.py \
    --total_blocks 21 \
    --overwrite_config_name causal_forcing_framewise

# - I2V
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_causal_wan21.py \
    --total_blocks 21 \
    --overwrite_config_name causal_forcing_framewise \
    --prompt_or_txt_path assets/example_data/i2v/prompt.txt  \
    --image_path assets/example_data/i2v/image.jpg
```

## Instructions to run FastVideo Wan2.2 Causal T2V Inference

reference: [FastVideo official inference script](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py)

T2V only for now; the FastVideo Wan2.2 checkpoint's I2V protocol (one-shot
first-frame VAE-seed warmup) doesn't fit the unified streaming pipeline's
per-AR-step mask-injection I2V and isn't wired here.

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_causal_wan22.py \
    --total_blocks 21
```

## Instructions to run Lingbot-World Camera Control I2V Inference

reference: [Lingbot-World repo](https://github.com/robbyant/lingbot-world?tab=readme-ov-file#fast-inference)

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_lingbot_world.py \
    --total_blocks 21
```


## Instructions to run Bidirectional Wan2.1 T2V Inference

reference: [Wan2.1 official repo](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#run-text-to-video-generation)

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
#    The single entry point picks T2V (1.3B) when --image_path is omitted
#    and I2V (14B 480P) when --image_path is provided.
# - T2V (1.3B)
uv run --package flashdreams --extra examples \
  flashdreams/examples/run_wan21.py \
    --height 480 --width 832

# - I2V (14B 480P) — pass --image_path to switch modes
uv run --package flashdreams --extra examples \
  flashdreams/examples/run_wan21.py \
    --height 480 --width 832 \
    --image_path assets/example_data/i2v/image.jpg \
    --prompt_or_txt_path assets/example_data/i2v/prompt.txt

# - I2V (14B 480P) using the example data from Wan2.1 codebase.
uv run --package flashdreams --extra examples \
  flashdreams/examples/run_wan21.py \
    --image_path ../Wan2.1/examples/i2v_input.JPG \
    --prompt_or_txt_path "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
