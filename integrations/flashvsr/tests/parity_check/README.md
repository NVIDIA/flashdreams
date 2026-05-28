# FlashVSR parity check

Self-contained benchmark of upstream
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) with a small local
patch (`changes.patch`) that adds `EventProfiler`-based per-chunk timing
and JSON stats output (mirroring `flashdreams`'s pipeline profiling).

## Run

From this directory — i.e.

```
cd integrations/flashvsr/tests/parity_check/
```

run:

```bash
bash run.sh
```

That's it. The script is idempotent and self-contained: on first run
it clones upstream at a pinned commit, downloads
`JunhaoZhuang/FlashVSR-v1.1`, materializes the parity-check venv
(`./.venv/`) — including `block-sparse-attn`, `pytest`, and the
workspace's `flashvsr` package layered on as an editable install so
the candidate side is importable from here — applies `changes.patch`,
runs both parity tests (`test_tcdecoder_parity.py` and
`test_dit_parity.py`, see below), and runs the benchmark. Subsequent
runs skip whatever's already in place and just re-run the parity
tests + benchmark.

Override the input video and upscale factor with environment variables:

```bash
INPUT_PATH=/abs/path/to/clip.mp4 SCALE=4.0 bash run.sh
```

`INPUT_PATH` defaults to upstream's `examples/WanVSR/inputs/example4.mp4`.
`SCALE` defaults to `4.0`, matching the patched `benchmark.py` default; set
it to the same value on the FlashDreams side when comparing outputs or stats.

## Compare against FlashDreams

Run the matching FlashDreams preset from the repository root with the same clip
and scale:

```bash
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 \
    --input-path /abs/path/to/clip.mp4 \
    --chunk-size 8 \
    --pipeline.encoder.scale 2 \
    --pipeline.enable-sync-and-profile True
```

Use `--chunk-size 8` for parity with the upstream benchmark. The FlashDreams
runner defaults to `--chunk-size 16`, which groups the cold/steady sequence as
`13/16` raw frames and runs two DiT iterations per generated chunk. The
upstream tiny-long loop emits one 8-frame padded model step per chunk, so
`--chunk-size 8` uses the compatible `5/8` raw-frame schedule and gives
per-chunk profiler rows that line up with `videos/stats_offline.json`.

`flashvsr-v1.1-sparse-ratio-2.0` matches the upstream benchmark's default
`--sparse_ratio 2.0`; use `--pipeline.encoder.scale 2` only when `run.sh` is
also launched with `SCALE=2`.

## Outputs

Written under `FlashVSR/examples/WanVSR/`:

- `videos/offline.mp4` — generated upsampled video
- `videos/stats_offline.json` — per-chunk timings (`projector_ms`,
  `dit_ms`, `decoder_ms`, `color_ms`, `total_ms`, `total_ms_wo_finalize`)
  plus GPU memory stats, one entry per autoregressive chunk

The stage names line up with the seven events
`flashdreams.flashvsr.FlashVSRPipeline.generate` emits
(`pad`/`bicubic`/`projector`/`dit_concat`/`denoise`/`decoder`/`color`),
so the JSON is directly comparable to the in-tree
`stats_flashvsr-v1.1-sparse-ratio-2.0.json`.

## Parity tests

`run.sh` invokes both parity tests in this directory before the
benchmark; both run from the same parity-check venv where `flashvsr`
is layered on as an editable install (`flashdreams-flashvsr =
{ path = "../.." }` in this directory's `pyproject.toml`). To run
either manually:

```bash
cd integrations/flashvsr/tests/parity_check/
uv run pytest test_tcdecoder_parity.py test_dit_parity.py -v
```

Both tests auto-skip when `FlashVSR/` isn't cloned, when the relevant
checkpoint isn't staged under `$FLASHVSR_WEIGHTS_ROOT/FlashVSR-v1.1/`,
or when no GPU is available (chunk-parity + CUDA-graph cases).

### TC decoder parity (`test_tcdecoder_parity.py`)

Loads upstream's `examples/WanVSR/utils/TCDecoder.py` (out of the
cloned `FlashVSR/` sibling) and the live
`flashvsr.decoder.network.FlashVSR_TAEHV` candidate side-by-side, then
asserts:

- state-dict shapes match `TCDecoder.ckpt` for both reference and
  candidate (after the candidate's `decoder.<i>` → `decoder.blocks.<i>`
  remap),
- chunk-by-chunk numerical parity at fp32 cross-algorithm conv
  tolerance (`atol=2.5e-3 / rtol=1e-3`; see the inline comment in
  `test_tcdecoder_chunk_parity` for why bit-for-bit isn't reachable --
  legacy runs convs at `batch=1` while the candidate runs them at
  `batch=b*t*stride`, so cuDNN picks different kernels); the legacy
  side runs `decode_video(parallel=False)` because upstream's
  `parallel=True` path doesn't carry mem across calls,
- the candidate's CUDA-graph wrapper captures by chunk 4 and matches
  the eager path bit-for-bit (`atol=rtol=1e-5`; both sides share the
  same impl, only the launch path differs).

The upstream file is self-contained (only `torch` / `einops` / `tqdm`
/ stdlib), so it's loaded via `importlib.util.spec_from_file_location`
rather than through `diffsynth.*` -- no package import plumbing
required.

### DiT parity (`test_dit_parity.py`)

Loads upstream's `diffsynth.models.wan_video_dit.WanModel` plus the
streaming-forward wrapper
`diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video` and the
live `flashvsr.transformer.FlashVSRTransformer` candidate side-by-side,
then asserts:

- state-dict shapes match the downloaded
  `FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors`
  checkpoint for both upstream and candidate; the upstream model config
  is derived with `WanModelStateDictConverter().from_civitai(...)`,
- steady-state chunk-by-chunk numerical parity under the streaming
  KV-cache protocol with a calibrated bf16 envelope (`max_abs <= 2.5e-1`
  and `mean_abs <= 3.5e-2`); the upstream cold-start chunk seeds the
  candidate's self-attention KV cache directly because upstream executes
  cold start as one 6-latent-frame call while FlashDreams' production path
  splits work into 2-latent-frame internal steps. The wider envelope
  accounts for upstream's fp64 RoPE + public sparse-attention wrapper
  versus FlashDreams' fused RoPE + direct sparse-attention path.

The upstream `WanModel.forward` is training-only; the streaming
inference path the upsampler actually drives is
`model_fn_wan_video(dit, ...)`, which consumes `dit.patchify` /
`dit.freqs` / `dit.blocks` / `dit.head` / `dit.unpatchify` directly
and is what `FlashVSRTinyLongPipeline` calls per chunk. The test
mirrors that call site verbatim so a future refactor that lands
streaming into `WanModel.forward` will surface as a parity break here.

Unlike the TC decoder file the DiT references reach back into
`diffsynth.models` / `diffsynth.pipelines`, so it's loaded as a plain
package import out of the editable `diffsynth` install (`uv pip
install --no-deps -e ./FlashVSR` in `run.sh`).

## Block-Sparse-Attention

Upstream FlashVSR's Locality-Constrained Sparse Attention requires the
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
CUDA extension. `uv sync` (called from `run.sh`) builds and installs
it automatically into this directory's `.venv/`; the build env is
seeded by `[tool.uv.extra-build-dependencies]` in `pyproject.toml`
(setuptools / wheel / packaging / psutil / ninja + a runtime-matched
torch), and `[[tool.uv.dependency-metadata]]` declares the static
metadata so uv doesn't have to build the package just to resolve the
venv. No manual `python setup.py install` step is needed.

System prerequisites uv can't provide:

- `nvcc` (CUDA compiler) on `PATH` — typically from the system CUDA
  toolkit matching the project's torch CUDA version
- ~5–10 minutes of build time on first sync; subsequent runs reuse the
  cached wheel.

If `uv sync` fails to build the extension, `run.sh` exits early with a
hint pointing at `uv sync -v` for the full build log.

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `FlashVSR/` resolve to *this* venv, not the surrounding
flashdreams one. The cloned upstream tree itself is registered with
`uv pip install -e ./FlashVSR` so the upstream `diffsynth` / `utils`
packages are importable.

`run.sh` exports `UV_PROJECT_ENVIRONMENT=${SCRIPT_DIR}/.venv` for the
parity-check uv calls. `data_local/docker_interactive.sh` pins
`UV_PROJECT_ENVIRONMENT=/tmp/venv/flashdreams` on the docker session so
every workspace `uv sync` lands in one shared venv; without the
override, `uv sync` from here would manage the shared venv and
uninstall every workspace integration (`flashdreams-omnidreams`,
`flashdreams-flashvsr`, …) that this directory's `pyproject.toml`
doesn't declare.

Both parity tests and the benchmark run from this same parity-check
venv. The candidate side (`flashvsr.transformer.FlashVSRTransformer`,
`flashvsr.decoder.network.FlashVSR_TAEHV`) is made importable by the
`flashdreams-flashvsr = { path = "../.." }` editable source declared
in `pyproject.toml`; the legacy upstream side is registered via
`uv pip install --no-deps -e ./FlashVSR`. `pytest` is a direct
dependency of this venv, so no workspace-venv flip is needed and the
docker-inherited `UV_PROJECT_ENVIRONMENT` stays untouched after we
override it at script start.

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + patch + parity test + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `changes.patch` — local edits on top of the pinned upstream commit
  (`EventProfiler` timing, JSON stats dump)
- `test_tcdecoder_parity.py` — TC decoder parity test against
  upstream's `examples/WanVSR/utils/TCDecoder.py`; run via this
  directory's parity-check venv (see above)
- `test_dit_parity.py` — DiT parity test against upstream's
  `diffsynth.models.wan_video_dit.WanModel` +
  `diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video`; same
  venv as above
- `.gitignore` — ignores the cloned `FlashVSR/` tree and `./.venv/`
