<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Sim Transfer

Turn a control video (segmentation, depth, edge or blur) and a caption into an
832x480 video, one latent at a time (one opening frame, then four frames), with
[nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer).
The package runs on the shared Cosmos3 runtime in
[`cosmos_interactive`](../cosmos_interactive/README.md) and its bounded stream
cache, so history memory stays fixed however long the clip runs. Each session
uses one control kind; simultaneous depth and segmentation conditioning is not
supported. Controls are RGB-encoded maps, not raw metric-depth tensors or
segmentation ID arrays.

See the [model guide](../../docs/source/models/cosmos3_nano_sim_transfer.md)
for the quick start and supported profiles.

It offers two ways in:

- **Batch:** the v2 application `control2v-cosmos3-nano-sim-transfer` for
  `flashdreams-run-v2`: a control clip, a caption, a seed and a history profile
  in, an MP4 out, with stage timings per latent.
- **Streaming:** `TransferStream`, for a live host in the same process:
  `start` a session, `step` it with the control frames of each latent, get the
  generated frames back.

Status: the model path (prompt, control encoding, rollout, decoding) with CPU
tests and a GPU parity probe against cosmos-framework. The quality baseline is
eager inference with the export's four-step schedule and Wan 2.2 decoding.
`accel="graphs"` runs the same math on fixed K/V buffers and CUDA graphs,
and `accel="compile"` compiles the layers too (see
[Speed](#speed-fixed-kv-buffers-cuda-graphs-compile)).

## Install

```bash
uv sync --frozen --inexact --python 3.12 \
  --package flashdreams-cosmos3-nano-sim-transfer --package flashdreams --group cuda13
```

Use Linux, Python 3.12, a CUDA-capable NVIDIA GPU and `ffmpeg` on `PATH`.
The original integration was measured on an RTX PRO 6000 with about 37 GiB
peak allocated GPU memory for the eager 30/3 profile; allow additional device
memory for the CUDA context and other processes.

The weights (30.3 GB bf16) come from the release at a pinned revision
(`config.HF_REVISION`); `config.fetch_release()` downloads only the files the
rollout needs. They load strictly through
`cosmos_interactive.checkpoint_load.load_exported_checkpoint`, without
conversion. Controls go through the Wan 2.2 VAE encoder that FlashDreams already
uses (`Wan-AI/Wan2.2-TI2V-5B`). Writing the MP4 needs `ffmpeg` on `PATH`.

### Preparing a trained checkpoint

An explicit `release=` path must contain compatible metadata, the safetensors
index and all referenced shards. Missing or incompatible local paths fail
before model loading; only `release=None` downloads the pinned public release.
Native distributed training checkpoints (DCP) need an export first.

Use the Framework's
[student export](https://github.com/NVIDIA/cosmos-framework/blob/5950e7b4ea3677b99a2de279a7ed7058b5446e42/cosmos_framework/scripts/export_model.py)
to select the trained **student EMA** weights and preserve the resolved
`config.json` and `checkpoint.json` provenance. Its native Hugging Face export
has supported `model.net.*` weights but does not supply this integration's
Transfer metadata or `text_tokenizer` directory. Prepare those locally with:

```bash
python -m cosmos3_nano_sim_transfer.probes.checkpoint \
  NATIVE_HF_EXPORT FRESH_PREPARED_DIR \
  --tokenizer NATIVE_HF_EXPORT --checkpoint-id 'training-run@iteration' --hint seg
```

The tokenizer argument names the actual Qwen3-VL tokenizer directory, which
Framework exports at the root even with `--no-vit`. Select `--hint` from the
training recipe and reference request; repeat it only for a checkpoint trained
on multiple alternative controls. This does not enable simultaneous controls.
The helper requires explicit EMA provenance and supported resolved native
settings, maps them without public-release defaults, and checks that every
source key can be mapped by the shared loader. It
creates symlinks in a fresh output directory and leaves the source unchanged;
keep the original files available. `preparation.json` records source/config
hashes and the reference Framework revision.

This is artifact preparation, not trained-model parity validation. It does not
validate tensor shapes, tokenizer semantics or numerical agreement on a GPU.
Before selecting the result in a live host, strictly load the trained network
and compare the same prompt, controls, seed, schedule, history and prompt
emphasis against its Framework reference. The helper has CPU fixture coverage;
verification against the actual trained artifact is still pending.

## Batch: a control clip to an MP4

```bash
uv run --no-sync flashdreams-run-v2 control2v-cosmos3-nano-sim-transfer \
  --output-path out.mp4 --timeout unbound -- \
  --control seg.mp4 --hint seg --prompt "A city street seen from behind a car ..." \
  --seed 0 --history long-rollout --timing-path timing.json
```

The application reads the control clip, generates one latent per model step
and finishes after the last. A clip that is not `1 + 4n` frames long is handled
as cosmos-framework handles it: the control is padded to whole latents for the
encoder, the prompt states the clip's own length and the output is trimmed to
it. Its arguments (`flashdreams-run-v2 control2v-cosmos3-nano-sim-transfer -- --help`):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--control` | required | Control video, any size; resized to 832x480 per `--hint` |
| `--hint` | `seg` | `seg`, `depth`, `edge` or `blur` |
| `--prompt` / `--prompt-file` | one required | Plain-text or JSON-object caption, or a file whose first non-empty line is the caption |
| `--emphasize-control-in-prompt` | off | Append Framework's modality-specific control-adherence sentence; match the reference request |
| `--prompt-aspect-ratio` | none | Optional nominal bucket label, such as `16,9`, injected into JSON prompt metadata; no effect on resizing |
| `--seed` | `0` | Noise seed |
| `--history` | `long-rollout` | A preset (below) or `W/S`: `W` latents in view, the first `S` pinned |
| `--fps` | the file's | Frame rate of the control video |
| `--max-frames` | all | Use at most this many control frames |
| `--prompt-frames` | the clip's | Clip length the prompt's duration sentence states |
| `--decoder` | `wan2.2` | `wan2.2` (the decoder cosmos-framework uses) or `taehv-super` (much faster, an approximation) |
| `--release` | download | Release snapshot folder |
| `--vae`, `--taehv` | download | Wan 2.2 VAE (`Wan2.2_VAE.pth`) and TAEHV-super weights |
| `--timing-path` | none | JSON file with the stage timings per latent, their steady-state medians, load times and peak GPU memory |

Every step's result also carries its stage timings, so the runtime's
`--stats-path` records them with its own measurements.

## Streaming: a session in a live host

```python
from cosmos3_nano_sim_transfer import TransferStream

stream = TransferStream.load(
    release_dir, hint="seg", history="long-rollout", fps=30.0,
    decoder="wan2.2", accel="eager",
)
stream.start(prompt="A city street seen from behind a car ...", seed=20261012)
frames = stream.step(first_control)   # [1, H, W, 3] uint8 in -> [1, 480, 832, 3] uint8 out
frames = stream.step(next_controls)   # [4, H, W, 3] uint8 in -> [4, 480, 832, 3] uint8 out
stream.last_timing                    # {"encode_ms", "model_ms", "decode_ms", "to_host_ms", "step_ms"}
stream.start(prompt=..., seed=...)    # next session: cache, encoder and decoder start over
stream.close()
```

`TransferStream.load(..., emphasize_control_in_prompt=True)` appends Framework's
exact modality-specific control-adherence sentence. The default is `False`,
matching the parity references even though the public checkpoint metadata
advertises `True`. Match the reference request explicitly; the preparation
helper's corresponding flag records provenance and does not change the stream
default. JSON-object captions receive structured duration, FPS and resolution
metadata using the pinned public
[Framework formatter](https://github.com/NVIDIA/cosmos-framework/blob/5950e7b4ea3677b99a2de279a7ed7058b5446e42/cosmos_framework/inference/inference.py).
`prompt_aspect_ratio="16,9"` also sets the JSON aspect-ratio field; omit it to
preserve an existing field. This label affects only the prompt, not the fixed
832x480 output. Structured formatting has CPU reference coverage. For a custom checkpoint,
compare it with that checkpoint's inference source during parity validation.

- One latent per `step`: one control frame for a session's first latent, four
  for each after it, as `[T, H, W, 3]` uint8 (numpy or torch, on any device),
  and as many generated frames back. `step_tensor` returns them as
  `[T, 3, 480, 832]` uint8 on the GPU instead.
- `start(prompt, seed, history=None, num_frames=None)` begins a session. It
  returns `new` when the prompt went into a new or changed text cache and
  `reset` when the previous prompt's cache was reused; both draw the same
  frames. `history` changes the window for this session (a new window
  reallocates the cache); `num_frames` is the clip length the prompt states,
  20 s (601 frames at 30 fps) by default for an open-ended session. `reset()`
  starts the current session over.
- `model_ms` is the control pass, the denoising passes (four with the
  export's schedule) and committing the clean latent to the cache. The stages
  are timed with a GPU synchronization between them, so they add up to about
  `step_ms`.
- `accel` is `eager` (the default), `graphs` or `compile` (see
  [Speed](#speed-fixed-kv-buffers-cuda-graphs-compile)); the fast modes need
  CUDA and a prompt of at most `text_cache_max_len` (512) tokens. `warm_up`
  runs throwaway sessions on black control frames that capture every graph,
  so call it once with the prompt the sessions will use, then `start`.
- `load(..., steps=2)` runs fewer denoising passes per latent: the export's
  schedule `(1, 0.9375, 0.8333, 0.625)` with intermediate sigmas dropped, so
  2 steps run `(1, 0.625)` and 3 steps `(1, 0.8333, 0.625)`
  (`config.skip_schedule`). `t_list=` gives a schedule directly (strictly
  decreasing sigmas in (0, 1] from the export's first, 1.0). Each step is one
  network pass; a latent also runs the control pass and the clean replay, so
  2 steps run 4 passes instead of 6. The checkpoint was distilled for four
  steps, so fewer trade quality for speed; truncating the schedule instead
  (`1, 0.9375`, then straight to the clean latent) leaves a color haze, which
  dropping intermediate sigmas avoids. `stream.t_list` is the schedule in use.
- The calls are those a host makes on a control-to-RGB model that generates a
  latent per step (`start(prompt=, seed=)`, `step`, `last_timing`, `close`,
  `width`, `height`), so a host adapter can pass them straight through.

The implementation lives under `impl/`. Its components can be driven by hand:
`impl.prompt.prompt_token_ids`, `impl.control.ControlEncoder`,
`impl.rollout.TransferRollout` (`start(text_ids, seed=)`,
`step(control_latent)` returning the clean `[48, 1, 30, 52]` latent),
`impl.fast.FastTransferRollout` (the same calls on the fast path) and
`impl.decode.LatentDecoder`.

## History presets

| Preset | Window / sinks | Use |
| --- | --- | --- |
| `long-rollout` | 30 / 3 | The release's profile for long clips (default) |
| `export` | 51 / 1 | The checkpoint's own (`checkpoint.json`) |
| `W/S` | `W` / `S` | Any other window, e.g. `20/1` |

A new latent attends to `W` latents, itself and the `S` sinks included; the
first `S` generated latents stay pinned. The cache holds `W` blocks, so memory
does not grow with the clip.

## How it runs

- **One latent per step, no opening frames.** Every RGB latent starts from
  noise. The export's chunk partition must be all ones; anything else is
  rejected, and so is an export with a system prompt other than
  `cosmos3_transfer_v1` or a control kind other than `seg`, `depth`, `edge`
  and `blur`. A session's `--hint` must be one of the export's `hints`.
- **Lazy control.** Control `k` enters the cache only after RGB `k-1` is
  committed, so it reads the prompt, earlier controls and strictly earlier
  clean RGB (`causal_control_with_rgb_history`). A live loop can encode it from
  the engine's frame at that point.
- **One cache block per latent, `[control | RGB]`.** The control pass reads the
  blocks before it; the denoise passes (four with the export's schedule) and
  the clean replay read those plus the latent's own control; the replay writes
  the RGB half. No pass needs an attention mask.
- **Positions and noise.** Base FPS, the text-to-vision margin, the sampler
  schedule and the window come from the export's `checkpoint.json`. Noise
  follows cosmos-framework's Transfer seeding: `seed + k` for the starting noise
  in latent layout, `seed + k * 1_000_003 + (step + 1) * 9_176` for re-noising,
  with `step` counting the steps of the schedule in use.

## Original integration measurements

The results in this section and the speed table below were supplied by Sascha
Sertel with the original integration. They are historical measurements, not
new benchmark results for the current upstream checkout. The recorded inputs
and tensor dumps are not distributed here; use the parity probe below with
your own matched native reference to reproduce the procedure.

The original runs used an RTX PRO 6000 workstation, eager inference, and
cosmos-framework references for 201-frame and 601-frame recorded segmentation
clips (called `long_201` and `long_601` below), seed 20261012. Pixel PSNR is
per frame in RGB, peak 255, median per second, file to file.

These numbers were measured with the workspace's dependency versions from before
FlashDreams 0.2.0. The author also reported a rerun with the 0.2.0 lock: the
first 16 latents of `long_201` pass the same bit-for-bit checks (prompt ids,
positions, initial noise, history limits). Its teacher-forced PSNR is within
0.1 dB of the values below for every latent after the first (first latent
37.4 dB with full history and 34.8 dB at 30/3), and its per-second PSNR is
within 0.42 dB in the two whole seconds it covers.

- **Bit for bit** (both `long_201` runs, full history and 30/3, from tensor
  dumps): prompt ids 458 of 458, the text and vision positions of all 152
  passes and the initial noise; the history limits of every cache write in the
  dumps follow each run's window.
- **One step at a time** (teacher-forced: every latent generated on the
  reference's own history): 47.1-53.3 dB latent PSNR with full history and
  48.4-52.8 dB at 30/3 for every latent after the first (first latent 37.3 and
  35.7 dB), flat through the latents the window evicts. The same check with the
  wrong (full) history on the 30/3 run falls to 37.5 dB.
- **Free rollout, `long_201`, full history:** with cosmos-framework's control
  latents, at or above the spread of two cosmos-framework runs of the same
  request on two GPUs (a B300 and an RTX PRO 6000) in every second; with this
  package's own control encoding, up to 0.83 dB below it (its control latents
  agree with cosmos-framework's at 59.3 dB).
- **Free rollout, `long_601` at 30/3:** two cosmos-framework runs of the same
  request, compiled on an RTX PRO 6000 and on a B300, agree at 32.5 dB in the
  first second and at 20.3-22.8 dB after 7 s. Against the B300 output this
  package is at or above that spread in every second of 0-6 s (32.8 dB in the
  first). Against the RTX PRO 6000 output it is below it by 1.4 dB in the
  first second and by 0.3-3.3 dB from 4 s to 20 s, within 0.01 dB at 1 s and
  above it at 2-3 s (31.1 dB in the first second, 18.9-21.1 dB after 7 s). The
  one-step check on the 30/3 `long_201` dump holds through 21 evicted latents,
  so the long free-rollout differences remain an acceptance limitation.
  Per-second PSNR alone cannot separate numerical drift from implementation
  differences.

| Eager, 30/3, steady state | ms per latent (4 frames) |
| --- | --- |
| Control resize and encode | 49.3 |
| Model (control pass, 4 denoising passes, cache commit) | 398.4 |
| Wan 2.2 decode | 156.7 |
| Total | 605.5 (p90 619.3) |

TAEHV-super decodes a latent in 4.5-14.8 ms (28.9-30.1 dB against the Wan 2.2
decode). Over 151 latents at 30/3 the allocated GPU memory stays at 34.72 GiB
from latent 1 on (peak 37.02 GiB), of which the stream cache is 3.21 GiB
(23,400 tokens).

The author also checked `TransferStream` and the batch application:
on `long_201` at 30/3 the stream's frames equal the probe's frame for frame,
and the application's MP4 (the runtime's H.264 at CRF 18) is 37.9-39.3 dB from
them in every second. The application took 577-586 ms per latent at steady
state (encode about 38, model 389-393, decode 150-152), 6.5 frames per second
over the whole clip and 130-140 s to load the network, encoder and decoder onto
the GPU with the weights already in the page cache; it peaked at 37.0 GiB
allocated.

## Speed: fixed K/V buffers, CUDA graphs, compile

`impl.fast.FastTransferRollout` runs the rollout's passes on the same weights,
positions, noise and history, in three steps that each keep the math:

1. **Fixed K/V buffers.** One buffer per layer holds the prompt's K/V
   right-aligned in `text_cache_max_len` slots, then the bounded cache (the
   `BlockKVCache` storage is a view of the rest, and its window rolls in place
   there). Each pass writes its own K/V straight into its slot (the denoise
   passes use the RGB slot as scratch until the replay fills it) and attends
   over one contiguous slice. Eager concatenates prompt, history and the
   current K/V for every layer of every pass instead (about 3.5 GB of copies
   per pass at 30/3). Same SDPA kernel, same keys, same order: equal frames in the reported runs.
2. **CUDA graphs.** A pass's slices depend only on its kind and its write
   slot: the first `W - 1` latents of a session each have their own, then
   every latent uses the same ones. Each geometry (90 at 30/3) is captured
   after one eager call and replayed from then on, all in one memory pool,
   since passes run one at a time and their outputs are copied out at once.
   `TransferStream.warm_up` runs the two sessions that capture them all.
3. **Compile** (`accel="compile"`). The work around attention (norms,
   projections, RoPE, MLP) is one `torch.compile` function for every layer,
   with Inductor's `emulate_precision_casts`, so fused kernels round to
   bfloat16 where eager does. Fused reductions still round differently, so
   the output is close to eager, not equal to it.

The original speed measurements used an RTX PRO 6000 workstation, 30/3
history, four steps, TAEHV-super and a recorded segmentation clip (151
latents), with one weight load,
every mode after its warm-up; ms per latent (4 frames), steady state from
latent 33, p50 (p99):

| Mode | Encode | Model | Decode | Step | Model, latents 1-29 | Same frames as eager |
| --- | --- | --- | --- | --- | --- | --- |
| `eager` | 37.3 | 391.8 (403.8) | 4.7 | 434.4 (446.4) | 308.0 | (reference) |
| fixed buffers, eager launches | 37.7 | 351.2 (375.6) | 3.7 | 393.8 (420.4) | 282.5 | yes, bit for bit |
| `graphs` | 38.5 | 314.2 (318.2) | 5.0 | 357.6 (363.6) | 242.1 | yes, bit for bit |
| `compile` | 38.5 | 296.2 (299.7) | 5.2 | 340.0 (352.1) | 225.0 | no: see below |

The fixed-buffer row is from a second run of the same bench (its eager
measured 389.8 ms model). Graphs captured only for a full window gave the
same steady state (315.3 ms) and 285.6 ms over latents 1-29. A CUDA graph of
the Wan 2.2 encoder's four-frame calls changed nothing (37.1 against 38.6 ms:
its convolutions keep the GPU busy), so the encoder stays eager.

At 30 fps a latent's four frames last 133.3 ms, so `compile` runs at 0.39
times real time (`graphs` 0.37, `eager` 0.31). A profile of `graphs` at
steady state: the GPU is busy 95 % of the step; attention (flash, about
0.6 ms per layer and pass at 23,900 keys) and the network's GEMMs (about
0.6 ms per layer and pass at 390 tokens) take 256 ms of the model's 306
under the profiler, so what is left is mostly the arithmetic itself.

`compile` against `eager` on the same inputs: a teacher-forced step (each
latent generated on eager's own history, the parity probe's method) is 38.7
dB from eager at latent 0 and 47.7-53.9 dB (median 50.9) after it; without
`emulate_precision_casts` it was 32.8 dB and 39.9 dB at worst. That is the
range of numerical differences seen in the parity work; it does not
establish equivalence for every input. The reported free rollout drifts (pixel PSNR 36.4 dB in the first second, 20-24 dB after 10 s).
Warm-up at load: 21 s with `graphs`, 19 s with `compile` (22 s with an empty
Inductor cache). Peak allocated memory is the same in all three modes (33.5-33.6
GiB over 40 latents): the fast path's K/V buffers take 3.29 GiB against eager's
3.21 GiB cache. Steady-state times vary by about 5 % between runs (the fast
modes measured 299 and 284 ms model in a short run on a cooler GPU); the table
is one run, all modes after the same warm-up.

## Checks

The narrow install above leaves out the workspace's `test` group, so add the
test tools to that environment first (or run from a full workspace sync,
`uv sync --locked --group test`, as CI does):

```bash
uv pip install --python .venv/bin/python pytest pytest-manual-marker tomli
uv run --no-sync pytest integrations_v2/cosmos3_nano_sim_transfer/tests -m ci_cpu
```

The CPU tests cover the export settings and history presets, the prompt ids
(458 of 458 equal to cosmos-framework's for a 201-frame request, when a
tokenizer is available), positions, noise seeds, the control resize, and the
cache: on a tiny network, the stream cache selects exactly the history
cosmos-framework selects, for several windows and sink counts, with fixed
memory, and that the fast rollout's fixed buffers give the eager rollout's
latents through the fill and past it, with teacher forcing, restarts and
prompt changes. On the same tiny network they check that `TransferStream` gives the
frames of its parts driven by hand and keeps its session contract, and run the
application through the v2 runtime into an MP4 file. Set
`COSMOS3_TRANSFER_RELEASE` to a release snapshot to also check its settings,
and `COSMOS3_TRANSFER_TOKENIZER` to its `text_tokenizer` folder to tokenize
offline.

### Streaming lifecycle with real weights (GPU, manual)

Run the lifecycle test against a local release or prepared trained export:

```bash
COSMOS3_TRANSFER_RELEASE=RELEASE_DIR COSMOS3_TRANSFER_VAE=Wan2.2_VAE.pth \
COSMOS3_TRANSFER_ACCEL=eager \
  uv run --no-sync pytest integrations_v2/cosmos3_nano_sim_transfer/tests/test_stream_gpu.py \
  -p no:manual_marker -m manual
```

Set `COSMOS3_TRANSFER_ACCEL=graphs` or `compile` to exercise each accelerated
path. The test checks segmentation and depth controls, finite latents, output
shapes, bounded history after eviction, deterministic reset and session closure.
It uses synthetic controls and checks runtime behavior; reference-model quality
and trained-checkpoint parity need the separate probe below.

### Parity probe (GPU, manual)

`probes/parity.py` reruns a cosmos-framework Transfer sample with this package
and compares the decoded video with it, second by second:

```bash
python -m cosmos3_nano_sim_transfer.probes.parity \
  --release RELEASE_DIR --vae Wan2.2_VAE.pth --out OUT_DIR \
  --ref RUN_DIR/SAMPLE_NAME --floor SAMPLE_NAME_w30s3=OTHER_RUN/SAMPLE_NAME/vision.mp4
```

A reference is a sample folder written by cosmos-framework's inference CLI
(`sample_args.json`, `vision.mp4`, with the run's `setup_args.json` beside it
or one level up, which gives the history) for a request of `1 + 4n` frames
whose control input still exists. It retains the sample's explicit
`emphasize_control_in_prompt` option (off
when absent in older reference files). With a tensor dump of the run
(`ref.pt`, format in the probe's docstring) it also checks prompt ids, the
positions of every pass and the initial noise bit for bit, and that the dump's
history limits follow the run's window; runs every step teacher-forced on the reference's exact history (its
control latents, its own clean latents replayed), so that only one step's
numerics separate each latent from the reference; decodes the dump's latents;
and reruns the rollout on the dump's control latents. `--floor` names
a second cosmos-framework run of the same request, on another GPU: its distance
from the reference is the reference's own spread, which each run is measured
against. Decoded videos are written as cosmos-framework writes its own
(lossless H.264 in YUV 4:2:0), so the two files compare like two
cosmos-framework outputs.

`tests/test_parity_gpu.py` (marker `manual`) runs the probe on one dumped
sample and asserts the bit-level checks, the teacher-forced bar and, given a
floor, the rollout on the dump's controls against it:

```bash
COSMOS3_TRANSFER_RELEASE=RELEASE_DIR COSMOS3_TRANSFER_VAE=Wan2.2_VAE.pth \
COSMOS3_TRANSFER_PARITY_REF=RUN_DIR/SAMPLE_NAME \
COSMOS3_TRANSFER_PARITY_FLOOR=OTHER_RUN/SAMPLE_NAME/vision.mp4 \
  uv run --no-sync pytest integrations_v2/cosmos3_nano_sim_transfer/tests/test_parity_gpu.py \
  -p no:manual_marker -m manual
```

## Not done yet

- **More speed.** Returning a latent's frames before its clean latent is
  committed to the cache (one pass less latency, no throughput improvement),
  and FP8 GEMMs. Reduced sampling schedules are already available through
  `steps=` and `t_list=`, but require separate quality validation. The native
  parity numbers above use the eager four-step baseline.
- **Prompt length on the fast path.** The fast modes hold the prompt in
  `text_cache_max_len` (512) slots and refuse a longer one; the eager path
  takes prompts of any length.
- **Long-clip acceptance.** The free rollout on `long_601` is below the
  cross-GPU spread against one of the two references (see above). A
  cosmos-framework tensor dump of a 601-frame run would extend the one-step
  check from 51 to 151 latents.
- **Host adapters.** A live host that runs Cosmos Transfer in another process
  can call `TransferStream` in process instead; that adapter belongs to the
  host.
