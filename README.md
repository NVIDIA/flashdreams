# FlashDreams

## Environment setup

Install all workspace packages (flashdreams core + every integration) into a venv:

```bash
uv sync --extra dev --group lint
```

Then run commands with `uv run` (auto-activates the venv):

```bash
uv run pytest flashdreams/tests
uv run flashdreams-run --help
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

## Unified `flashdreams-run` CLI

`flashdreams-run` is the single console script for every recipe. It
dispatches over the runner registry (in-tree + plugin-discovered) and
exposes every overridable field as a CLI flag.

```bash
# List every registered runner.
uv run flashdreams-run --help

# Per-runner help: every overridable field is a flag.
uv run flashdreams-run wan21-t2v-1.3b-480p --help

# Single-GPU run.
uv run flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."

# I2V variant (--image-path defaults to the bundled demo frame).
uv run flashdreams-run wan21-i2v-14b-480p --prompt "..."

# Resolve the config without touching a GPU (good for debugging overrides).
uv run flashdreams-run template-offline --no-instantiate
```

I/O extras (mediapy + opencv) are needed by every runner; install with
`--extra runners` whenever you want to actually generate videos:

```bash
uv sync --extra dev --extra runners --group lint
```

### Multi-GPU (context-parallelism)

Recipe transformers auto-detect their CP size from `torch.distributed`'s
`WORLD` group, so multi-GPU is just torchrun + the same command:

```bash
uv run torchrun --nproc_per_node=4 --no-python \
    flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."
```

Why `--no-python`: torchrun's default is `python <training_script> ...`,
which would treat `flashdreams-run` as a relative path in cwd. With
`--no-python` torchrun execvps the binary directly, so PATH lookup
finds the venv shim. Once running, `Runner.__init__` initializes
`torch.distributed`, pins `cuda:LOCAL_RANK` per rank, and exposes
`self.local_rank` / `self.world_size` / `self.global_rank` /
`self.is_rank_zero` to the recipe; runners gate their I/O (mp4, stats
JSON, .pt dump, user-facing logs) on `is_rank_zero` so only one rank
writes outputs. There is no `cp_size` knob — the launcher is the single
source of truth.

Tyro's `--help` and parse-error banners print exactly once on rank 0;
non-zero ranks suppress them via tyro's distributed-mode hook.

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

# 1. huggingface
export HF_TOKEN=<YOUR-HF-TOKEN>
export HF_HOME=~/.cache/huggingface              # optional; this is the default
export FLASHDREAMS_CACHE_DIR=~/.cache/flashdreams # optional; this is the default

# 2. (internal team) flip checkpoint + example-data URLs back to s3://flashdreams.
#    Skip for external users. Requires the S3 credentials in step 3.
export FLASHDREAMS_INTERNAL_STORAGE=1

# 3. (only if step 2 is set, or you run a slug flagged "S3" below) S3 credentials.
cat > credentials/s3_checkpoint.secret <<EOF
{
  "aws_access_key_id": "team-sil-videogen",
  "aws_secret_access_key": <YOUR-SIL-VIDEOGEN-PDX-KEY>,
  "endpoint_url": "https://pdx.s8k.io",
  "region_name": "us-east-1"
}
EOF

# 4. Run inference. Checkpoints + example data are auto-downloaded on first run.
#    --example-data fills the per-camera path tuples from a bundled HDMap clip
#    + first frame; --example-data-uuid <uuid> picks one of the 32 single-view
#    clips at https://huggingface.co/datasets/nvidia-omni-dreams-lha/omni-dreams-samples/tree/main/data/single_view .
# - single view on single GPU (best-perf preset; fully HF-native)
uv run flashdreams-run \
    alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
    --example-data True --total-blocks 20

# - multi view on 4 GPUs (S3: --example-data still pulls multi-view clips from S3)
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    alpadreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
    --example-data True --total-blocks 20

# - diffusion forcing AR model (S3: chunk2 diffusion-forcing checkpoint not on HF yet)
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    alpadreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
    --example-data True --total-blocks 12
```

## Instructions to run Alpadreams Bidirectional Model

Use the same container, Hugging Face token, and `FLASHDREAMS_CACHE_DIR` setup
as the Alpadreams inference section above. The bidirectional recipe runs the
single-view full-block Cosmos2 2B / 720p / HDMap checkpoint; that checkpoint
is still S3-hosted, so the S3 credentials block from step 3 above is required
for this recipe.

The bidirectional recipe defaults to the checkpoint-trained 48 latent chunks
(189 decoded frames with the Wan decoder). To shrink the chunk for tighter VRAM
budgets, override the wrapped transformer's `len_t` directly. The recipe
generates one full block per run, so keep `--total-blocks 1`.

```bash
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    alpadreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
    --example-data True --total-blocks 1 \
    --pipeline.diffusion-model.transformer.len-t 24
```

Useful options:

- `--example-data`: lazy-syncs the bundled HDMap clips + first frames from S3
  and fills the per-camera path tuples. Skip it and pass
  `--hdmap-video-paths` / `--first-frame-paths` for production runs.
- `--save-embeddings-path`: runs only the one-shot encoders, saves positive
  text, negative text, and first-frame image embeddings to a `.pt`, then exits.
- `--embeddings-path`: loads a `.pt` produced by `--save-embeddings-path` and
  skips loading the one-shot encoders during inference (peak-VRAM win).
- `--pipeline.diffusion-model.transformer.len-t N`: overrides the
  bidirectional recipe's latent chunk count. Omit to use the
  checkpoint-trained default (`48`); lower when runtime memory is tight.

The generated comparison video is written to
`outputs/{runner_name}.mp4`, with the HDMap condition stacked above the
generated RGB output. Per-step stats are saved under
`outputs/stats_{runner_name}.json` when profiling is enabled by the selected
config.

To convert an I4/SIL distributed checkpoint directory into the single-file
`.pt` format consumed by the FlashDreams bidirectional config:

```bash
uv run python flashdreams/scripts/convert_i4_dcp2pt.py \
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

The Self-Forcing slugs ship as an out-of-tree plugin
(`flashdreams/plugins/self_forcing/`); install it once before invoking
its slugs through `flashdreams-run`.

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. install the plugin (one-time; declared as a uv workspace member, so
#    `uv sync` from the repo root is enough — this line is for clarity).
uv pip install -e flashdreams/plugins/self_forcing

# 2. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 3. Run inference. Checkpoint is auto-downloaded from huggingface at first run.
uv run flashdreams-run causal-wan21-self-forcing-t2v --total-blocks 7
```

## Instructions to run Causal-forcing T2V and I2V Inference

The Causal-Forcing slugs ship as a separate out-of-tree plugin
(`flashdreams/plugins/causal_forcing/`).

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. install the plugin.
uv pip install -e flashdreams/plugins/causal_forcing

# 2. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 3. Run inference. Checkpoint is auto-downloaded from huggingface at first run.
# - T2V
uv run flashdreams-run \
    causal-wan21-causal-forcing-framewise-t2v --total-blocks 21

# - I2V (out-of-tree plugin doesn't bundle demo assets; pass --image-path
#   and, optionally, --prompt-path explicitly).
uv run flashdreams-run \
    causal-wan21-causal-forcing-framewise-i2v --total-blocks 21 \
    --image-path assets/example_data/i2v/image.jpg \
    --prompt-path assets/example_data/i2v/prompt.txt
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

# 2. Run inference. Checkpoint is auto-downloaded from huggingface at first run.
uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b --total-blocks 21
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

# 2. Run inference. Checkpoint is auto-downloaded from huggingface at first run.
#    --example-data lazy-syncs the bundled prompt + first-frame + camera arrays
#    from S3 into assets/example_data/lingbot_world/ and fills the path defaults.
uv run python -m torch.distributed.run --nproc_per_node=1 --no-python flashdreams-run \
    lingbot-world-fast --example-data True --total-blocks 21
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

# 2. Run inference. Checkpoint is auto-downloaded from huggingface at first run.
# - T2V (1.3B)
uv run flashdreams-run wan21-t2v-1.3b-480p

# - I2V (14B 480P) -- --image-path defaults to the bundled
#   assets/example_data/i2v/image.jpg, override for a custom first frame.
uv run flashdreams-run wan21-i2v-14b-480p

# - I2V (14B 480P) using the example data from the Wan2.1 codebase.
uv run flashdreams-run wan21-i2v-14b-480p \
    --image-path ../Wan2.1/examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
