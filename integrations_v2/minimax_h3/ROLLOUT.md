# MiniMax H3 V2 rollout record

Validation was performed on 2026-08-25 against these immutable sources:

- FlashDreams V2 baseline `8fd97fa38f04bc32c288760fa0fbf5da52464cea`;
- source PR #457 `9962455fe0220e726b9d6484c64d8faf19873b14`;
- native-integration head before this record
  `6c8842c8924ea6294151ddee60bc3fe35b4b817b`;
- Diffusers parity oracle
  `175fe6b2419a01db9c2ceabd01ec37d2c0305fc2`;
- MiniMax model revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`.

The rollout host was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with
97,887 MiB VRAM and driver 595.84. The environment used Torch 2.12.1+cu130 and
CUDA 13.0. All model runs set `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, disabled compilation, and loaded only the pinned
snapshot. A separate process occupied about 10 GiB throughout the rollout.

Representative Hugging Face content-addressed weight identities were:

- standard transformer shard 1:
  `2d847200c45c09dd7f973c1b096663068408ef851ee0b3711d059b6dc5dcd028`;
- REF transformer shard 1:
  `7a3fcad885f51560e550b2e84c9a8d8b35e62996cfd9076937e992bd23478df9`;
- video VAE shard 1:
  `72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374`;
- audio VAE:
  `52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb`.

The fully cached component sizes were about 63 GiB for the text encoder,
62 GiB for each transformer variant, 9.8 GiB for the video VAE, and 578 MiB
for the audio VAE. No broken snapshot links or incomplete blobs remained.

## Automated gates

The core, native H3, and V2 H3 CPU gate completed with 764 passed and 105
deselected tests. The separate complete V2 runtime gate completed with 171
passed and 14 host-dependent skips. Production modules passed `ty` 0.0.53.

CUDA parity ran against the exact oracle checkout placed first on
`PYTHONPATH`:

```bash
PYTHONPATH=/tmp/diffusers-h3-175fe6b/src:$PWD/flashdreams:$PWD/integrations/minimax_h3:$PWD/integrations_v2/minimax_h3 \
  /home/jmccaffrey/projects/flashdreams/.venv/bin/python -m pytest \
  integrations/minimax_h3/tests -m ci_gpu -q
```

The native transformer, video VAE, and audio VAE all matched the oracle: three
tests passed. Production-boundary searches found no Diffusers, PyAV, V1 runner,
output-path, video-writer, bundled-FFmpeg, or Python FFmpeg imports in either H3
production package.

## Real-weight RTX results

All successful runs returned contiguous CPU float32 TCHW video in `[-1, 1]`
and finite normalized stereo audio at 32 kHz. Unless stated otherwise, the
canvas was 32x32, duration was 5 seconds, and the schedule had two points (one
Euler update).

| Workflow | Input and schedule | Result | Total | Peak H3 VRAM |
| --- | --- | --- | ---: | ---: |
| T2VA | seed 42, math attention | 124 frames, 165,600 audio samples | 47.18 s | 62.17 GiB |
| FL2VA | seed 43, first image, math attention | 124 frames, 165,600 audio samples | 48.63 s | 62.25 GiB |
| REF2VA | seed 44, image plus stereo audio reference, FlashAttention | 124 frames, 165,600 audio samples | 51.18 s | 66.59 GiB |
| T2VA canonical | seed 47, 30 schedule points, FlashAttention | 124 frames, 165,600 audio samples | 48.97 s | 62.17 GiB |
| T2VA maximum | seed 48, 15.0 seconds, FlashAttention | 362 frames, 482,400 audio samples | 47.25 s | 62.17 GiB |

The maximum-duration run proves that the advertised 15.0-second request aligns
upward to the required 362-frame H3 grid instead of failing or trimming to 360.
Its video timeline is 15.0833 seconds and native decoded audio is 15.075
seconds. A file sink must therefore pad 267 samples of silence when muxing to
the exact written-video duration.

The REF run also established the backend boundary: dense math attention for
the image-plus-audio reference case exhausted the memory left beside the
unrelated 10 GiB workload, requesting an additional 15.71 GiB. The production
default FlashAttention backend completed the same request at 66.59 GiB peak.

A paired checkpoint written after the real update contained exactly `audio`
and `video`, with `next_step=1`. A new engine resumed it without emitting a
denoise update and reproduced decoded video and audio bit-for-bit. The resumed
run took 42.27 seconds, including conditioner, weight staging, and both decodes,
then returned GPU allocation to 31 MiB.

The actual `MiniMaxH3Application` T2VA adapter also completed through
`IApplication`, `ISession`, and `IModelLoop`. It returned one `StepResult` at
the requested step index with 124 TCHW frames, 165,600 audio samples, and a
68,376-byte application-owned paired checkpoint. Application close was
idempotent and GPU allocation again returned to 31 MiB.

## File-output boundary

This host had no `ffmpeg` or `ffprobe` executable on `PATH`, so no real MP4 mux
or probe was attempted and no substitute executable was installed or used.
The transactional video-only MP4 path and exceptional abort behavior are CPU
tested, including preservation of a pre-existing target. The H3 application
declares audio, and the video-only MP4 sink rejects that declaration rather
than dropping audio.

Completing synchronized public MP4 output still requires explicit approval of
the public audio codec. After that decision, the sink must stage private f32le
PCM, pad or trim it to the written video timeline, mux using only a host
`ffmpeg` subprocess, atomically replace the target, and probe the result on an
approved host.
