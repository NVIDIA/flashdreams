# MiniMax H3 V2 rollout record

Real-weight artifact validation was performed on 2026-08-25. Clean
post-review verification was performed on 2026-08-26 against these immutable
sources:

- FlashDreams main and V2 design-document baseline
  `e6e1c002fc6996aa5d53894ccbb342a05a0e0582` (PR #515);
- latest merged-main compatibility baseline
  `6512f042420e5b6fd85dddbfcdb43db0218505e0` (PR #514);
- source PR #457 head
  `9962455fe0220e726b9d6484c64d8faf19873b14`;
- full-quality benchmark runtime and codec head
  `0b3ed9e8b0f4489d29ab23c98da6df3df21897dc`;
- real-weight artifact-validation head
  `4e91a25b4a09339ee2530cbff5d3a41aefc59d22`;
- contribution-audit head (documentation-only after production validation)
  `e49a971d96858c784b674e44cc7f8eb94ae4420c`;
- merged Claude-review-fix head
  `437c4b38b81dce10763c5eae7198053b7e29b96f`;
- Diffusers parity oracle
  `175fe6b2419a01db9c2ceabd01ec37d2c0305fc2`;
- MiniMax model revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`.

The rollout host was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
(`GPU-fc16c63c-42fb-e409-7b05-bfa3c8797b62`) with 97,887 MiB VRAM and
driver 595.84. A separate process occupied about 10 GiB throughout validation,
leaving 87,231 MiB free between runs. The stack used Torch 2.12.1+cu130,
CUDA 13.0, and external host executables `/usr/bin/ffmpeg` and
`/usr/bin/ffprobe` 6.1.1-3ubuntu5.

All model runs set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`CUDA_VISIBLE_DEVICES=0`, `PYTHONHASHSEED=0`, and
`TOKENIZERS_PARALLELISM=false`. They loaded only the pinned snapshot from the
Hugging Face cache and ran sequentially in fresh processes.

## Implementation outcome

The three V2 slugs are `minimax-h3-t2va`, `minimax-h3-fl2va`, and
`minimax-h3-ref2va`. Production inference is native FlashDreams code: it
does not import Diffusers, PyAV, a Python FFmpeg binding, or a V1 runner.
Diffusers is used only as the frozen CUDA parity oracle.

The runtime owns public output. H3 returns contiguous CPU float32 TCHW video
in `[-1, 1]` and finite normalized stereo audio at 32 kHz. MP4 publication
uses an external host `ffmpeg` process and fixed AAC-LC at 192 kbit/s. Before
creating output staging or starting model-loop inference, the sink resolves
that executable to a canonical absolute path and performs a bounded encode of
one exact-format silent AAC frame. Video and interleaved `f32le` audio remain
in a private sibling directory until a successful mux atomically replaces the
target.

The audio stream is padded or trimmed to
`round(written_frames * sample_rate / fps)`. Failed inference, encoding,
muxing, cleanup, or application initialization preserves a pre-existing target
and triggers transaction abort. The runtime gives an interrupted abort one
bounded final retry. A recovered transient removes staging; if the retry also
fails, the error is reported and the sink intentionally retains ownership for
an explicit later retry rather than falsely claiming cleanup.

## Automated gates

The artifact-validation checkout passed:

- the exact repository-wide contribution CPU tier with 2,056 passed, 2 skipped,
  and 346 GPU/manual tests deselected;
- 1,070 CPU tests across FlashDreams core, runtime V2, native H3, and H3 V2,
  with 105 GPU/manual tests deselected;
- the independent complete V2 gate with 296 passed;
- exact full-workspace `ty` 0.0.53;
- repository-pinned Ruff 0.12.7 lint and format checks;
- offline `uv lock --check`;
- DCO on every integration commit.

The merged Claude-review-fix head then passed from a clean detached checkout:

- every full pre-commit hook, including repository-pinned Ruff lint/format,
  full-workspace `ty`, `uv lock --check`, version sync, whitespace, EOF, and
  symlink checks;
- the exact repository-wide `ci_cpu` tier with 2,100 passed, 2 skipped, and
  346 GPU/manual tests deselected in 102.88 seconds;
- focused post-merge and review suites covering 222 and 107 CPU tests,
  respectively, including real external-`ffmpeg`/`ffprobe` media round trips;
- DCO sign-off on the latest-main merge and both review-fix commits.

The repository-wide tier used the same development dependency setup as CPU CI,
followed by the command documented in `CONTRIBUTING.md`:

```bash
uv sync --locked --extra dev --group test \
  --no-install-package transformer-engine-torch
uv run --group test pytest -m ci_cpu
```

CUDA parity ran against the exact oracle checkout placed first on
`PYTHONPATH`:

```bash
PYTHONPATH=/tmp/diffusers-h3-175fe6b/src:$PWD/flashdreams:$PWD/integrations/minimax_h3:$PWD/integrations_v2/minimax_h3 \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync pytest integrations/minimax_h3/tests -m ci_gpu -q
```

The native transformer, video VAE, and audio VAE matched the oracle: 3 passed
and 82 deselected. Production-boundary searches found no Diffusers, PyAV,
Python/bundled FFmpeg, V1 runner, application-owned output path, or video
writer in either H3 production package.

The 2026-08-26 fixes do not change native H3 model math, weights, scheduling,
or GPU placement, so the immutable real-weight artifacts and performance
measurements below remain tied to the artifact-validation head rather than
being relabeled as results from the review-fix head. The affected host-media,
Python 3.10 cleanup, application lifecycle, and final-mux paths were exercised
by the clean CPU tier, including the installed external FFmpeg executables.

## Real-weight validation method

All MP4 runs used:

```text
flashdreams-run-v2 SLUG
  --mode mp4 --pixel-width WIDTH --pixel-height HEIGHT --fps 24
  --layout tchw --backpressure-mode block
  --presentation-mode only_present_new
  --stats-path ARTIFACT.stats.json --output-path ARTIFACT.mp4 --
  --prompt "A brass clockwork bird flying through a rainy neon market with synchronized ambient sound."
  --duration DURATION --steps STEPS --seed SEED
  --attention flash --device cuda:0
  --cache-dir /home/jmccaffrey/.cache/huggingface/hub
  --checkpoint-min-free-gb 150
```

The exact run-specific arguments were:

| Artifact | Workflow and request | Conditioning |
| --- | --- | --- |
| `00-t2va-warmup.mp4` | T2VA, 768x768, 5 s, 2 steps, seed 40 | none; excluded from performance |
| `01/02/03-t2va-canonical.mp4` | T2VA, 768x768, 5 s, 30 steps, seed 47 | none; three identical requests |
| `04-fl2va.mp4` | FL2VA, 768x768, 5 s, 30 steps, seed 47 | first and last frames extracted from artifact 01 |
| `05-ref2va-video-audio.mp4` | REF2VA, 768x768, 5 s, 30 steps, seed 47 | artifact 01 as video reference plus its extracted stereo WAV |
| `06-t2va-max-duration.mp4` | T2VA, 256x256, 15 s, 2 steps, seed 49 | temporal/mux boundary case |
| `07-t2va-final-head-smoke.mp4` | T2VA, 768x768, 5 s, 2 steps, seed 50 | artifact-validation-head AAC smoke; excluded from performance aggregates |

The maximum-duration request intentionally used one denoise update at 256x256.
It proves real-weight temporal alignment and mux behavior, not canonical
768x768, 30-step maximum-duration quality or performance.

The large media artifacts, exact machine-readable manifest, stats, wall-time
records, and probes remain outside Git on the validation host under:

```text
/home/jmccaffrey/projects/minimax-h3-v2-validation/0b3ed9e8
```

The exact request configuration is checked in above and all content identities
are checked in below, so the PR record does not depend on mutable filenames.

## Performance

Times are synchronized model-stage seconds. Wall time comes from external
`/usr/bin/time`. The warmup is excluded. Canonical p90 uses nearest-rank p90;
with three samples that is the maximum.

| Run | Conditioning | Prepare | Transformer load | Denoise compute | Release | Denoise total | Video decode | Audio decode | Model total | Wall | Generated fps | Peak GPU GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA canonical median | 20.430 | 0.037 | 19.647 | 170.640 | 0.262 | 190.621 | 10.794 | 1.225 | 223.094 | 238.46 | 0.5558 | 65.569 |
| T2VA canonical p90 | 20.821 | 0.038 | 20.071 | 170.810 | 0.272 | 190.945 | 10.796 | 1.237 | 223.425 | 238.85 | 0.5560 | 65.569 |
| FL2VA | 24.772 | 0.098 | 19.489 | 198.282 | 0.265 | 218.036 | 11.276 | 1.193 | 255.376 | 271.38 | 0.4856 | 65.956 |
| REF2VA video + audio | 42.346 | 0.115 | 19.771 | 552.644 | 0.286 | 572.700 | 10.628 | 1.144 | 626.933 | 643.43 | 0.1978 | 69.871 |
| T2VA 15 s boundary | 21.037 | 0.021 | 19.657 | 2.118 | 0.252 | 22.027 | 5.621 | 1.266 | 49.971 | 84.15 | 7.2442 | 63.267 |

| Run | Compute / model total | Real-time factor at 24 fps | Wall minus model total | Unattributed denoise residual |
| --- | ---: | ---: | ---: | ---: |
| T2VA canonical median | 76.47% | 0.02316x | 15.425 s | 0.000021 s |
| FL2VA | 77.64% | 0.02023x | 16.004 s | 0.000014 s |
| REF2VA video + audio | 88.15% | 0.00824x | 16.497 s | 0.000014 s |
| T2VA 15 s boundary | 4.24% | 0.30184x | 34.179 s | 0.000013 s |

The boundary row is not comparable throughput because it uses one update and
spends most measured time loading weights. Canonical maximum RSS was
66,148,296 KiB median and 66,150,760 KiB p90.

## Validation identities

| Content | SHA-256 |
| --- | --- |
| FL first image | `817e195defb9c2f4b0005362a66a5bac8841822ba5564d71a990d44232f2234c` |
| FL last image | `3182610f12defb73dbe44a96b183d8a10eea131fe4742845f46e83d6b043dcc7` |
| REF stereo WAV | `05fada16b96b76ad4f4c9451bee3f57f259351ec1c4aad0721ede1a5852cfcc5` |
| Warmup MP4 | `7d558849ff24056e6b32a658c5f848e2c0ec1adf963fe6dc8a9563e7b4e5ea11` |
| Canonical MP4 01 | `d83b470f8d406f912ac046ff20d2b51c23ef415121de1e58aae7f48d9cc1a78b` |
| Canonical MP4 02 | `d83b470f8d406f912ac046ff20d2b51c23ef415121de1e58aae7f48d9cc1a78b` |
| Canonical MP4 03 | `d83b470f8d406f912ac046ff20d2b51c23ef415121de1e58aae7f48d9cc1a78b` |
| FL2VA MP4 | `459237e42a498142fc85191a41e1b3adfe32c9d787d92b6bc86624d2c80eac03` |
| REF2VA MP4 | `13027de768531b63c5da13ecfd8607601e171486b2b50e8e901569dd9ee8a26e` |
| Maximum-duration MP4 | `ee485c9f21dbebc73741e772cd4cd073093563c35f68dfa6495440a5168d9be7` |
| Final-head smoke MP4 | `761d4dd8d4fb3f75a0aaa1af5d8c1798e66d7314454e23f7414c673a930c9c99` |

## Media, determinism, and alignment

Every one of the eight generated MP4 files was decoded end-to-end through both
streams using external `ffmpeg`. External `ffprobe` reported exactly one
H.264 High `yuv420p` video stream at 24 fps and one AAC-LC 32 kHz stereo
audio stream. Correct-threshold black/freeze analysis found no whole-clip
interval, silence analysis found no whole-clip silence, and decoded audio had
no NaN, Inf, or denormal samples.

The three canonical MP4 files were byte-for-byte identical:

```text
d83b470f8d406f912ac046ff20d2b51c23ef415121de1e58aae7f48d9cc1a78b
```

Each canonical, FL2VA, and REF2VA artifact contains 124 frames. The video
duration is 5.166667 seconds; AAC duration is 5.167000 seconds, a 0.333 ms
difference.

The 15-second request aligns upward to 362 frames. Its video stream is
15.083333 seconds. Native decoded audio contains 482,400 samples; the sink pads
267 samples to the frame-derived 482,667-sample target before encoding. AAC
framing reports 482,688 ticks at 32 kHz, or 15.084000 seconds. The externally
probed audio/video difference is therefore 0.667 ms, well below one AAC frame
(32 ms), and both streams decode completely.

The FL2VA artifact SHA-256 is
`459237e42a498142fc85191a41e1b3adfe32c9d787d92b6bc86624d2c80eac03`;
REF2VA is
`13027de768531b63c5da13ecfd8607601e171486b2b50e8e901569dd9ee8a26e`;
the maximum-duration boundary artifact is
`ee485c9f21dbebc73741e772cd4cd073093563c35f68dfa6495440a5168d9be7`.
The artifact-validation-head smoke completed in 59.238 model seconds and 72.87 wall seconds
at 65.561 GiB peak GPU memory.

The validation directory also preserves the complete output of
`ffmpeg -version` (including configure flags) and `ffmpeg -L`. This Ubuntu
build enables GPL components and reports GNU GPL version 2 or later; those
files are evidence for deployment/license review, not a bundled project
dependency.
