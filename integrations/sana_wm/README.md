# flashdreams-sana-wm

SANA-WM bidirectional image-to-video inference as a native `flashdreams-run`
plugin.

The integration owns its FlashDreams runner, pipeline config, packaged demo
assets, and a private vendored copy of the SANA-WM reference implementation
under `sana_wm._reference`. It does not require a sibling NVlabs/Sana checkout
or mutate `sys.path` at runtime. The vendored path preserves the standalone
SANA-WM sampling, camera conditioning, VAE decode, and LTX-2 refiner code path
so native FlashDreams runs can be checked against standalone output.

Only runtime code and packaged demo inputs live in this plugin.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone editable install also works:

```bash
uv pip install -e integrations/sana_wm
```

## Hugging Face setup

Weights default to the public Hugging Face release
`Efficient-Large-Model/SANA-WM_bidirectional` and are downloaded on first use.
Set the cache/token environment if your runtime needs it:

```bash
export HF_HOME=~/.cache/huggingface
export HF_TOKEN=<your-hf-token>
```

## Run

Once installed, the slug is discovered automatically by `flashdreams-run`:

```bash
# List every registered runner.
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run sana-wm-bidirectional --help

# Single-GPU run with the packaged demo_0 inputs.
uv run flashdreams-run sana-wm-bidirectional \
    --num-frames 161 \
    --step 60 \
    --output-dir outputs/sana_wm_flashdreams \
    --name demo_0
```

By default this uses the packaged `demo_0` image, prompt, camera trajectory,
and intrinsics. Override `--image`, `--prompt`, `--camera`, and `--intrinsics`
for custom inputs.

The model weights still default to the public Hugging Face release
`Efficient-Large-Model/SANA-WM_bidirectional`. The default pipeline uses the
packaged `configs/sana_wm_1600m_720p.yaml`, full-resolution `704 x 1280`, and
the LTX-2 refiner. Set `--pipeline.enable-refiner False` to decode stage-1
latents directly with the Sana VAE.

The packaged default config uses SANA-WM's Triton-fused GDN attention path
and the FLA short-convolution module, so the plugin declares `triton` and
`flash-linear-attention` as Linux runtime dependencies. The packaged demo
includes intrinsics. If `--intrinsics` is explicitly omitted for a custom
image, the runner falls back to the standalone Pi3X estimator, which must be
installed separately because it is an optional Git dependency in upstream
SANA-WM rather than part of the default FlashDreams path.

## Parity

For standalone parity, run both commands with the same image, prompt, camera,
intrinsics, seed, step count, refiner setting, and action overlay setting. The
native pipeline intentionally calls the same vendored generation logic as the
standalone SANA-WM script; differences should be treated as bugs.

## Tests

```bash
uv run --extra dev pytest integrations/sana_wm/tests
```
