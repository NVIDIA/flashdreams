# LTX 2.5 V2 integration plan and evidence

This note is the reproducibility record for the FlashDreams V2 text-to-audio-video
adapter for LTX 2.5. Update outcomes in place as each validation layer completes.

## Scope and design

The integration is a one-shot custom V2 application around Diffusers' distilled
`LTX2Pipeline`. LTX 2.5 jointly denoises separate audio and video streams in an
asymmetric diffusion transformer; it is not a Wan-derived streaming recipe. The
adapter therefore owns only model loading, argument validation, tensor conversion,
and a finite model loop. FlashDreams owns threading, presentation, statistics sinks,
FFmpeg, transactional MP4 publication, and audio/video timeline reconciliation.

The production path uses the tiled convolutional video VAE included in the Diffusers
pipeline. The LTX 2.5 diffusion video decoder is intentionally deferred: adding it
would create a second denoising/offload path and should follow only after the concise
joint audio/video path has complete evidence.

Output is one `StepResult` containing `tchw` uint8 video and one stereo normalized-PCM
`AudioOutput`. The default runtime description is 768x512, 24 fps, 48 kHz stereo,
`BackpressureMode.BLOCK`, and `PresentationMode.ONLY_PRESENT_NEW`. Runtime overrides
may select dimensions divisible by 32 and an integer frame rate. Application arguments
select the prompt, seed, temporal length, offload policy, and local-cache-only mode.
Supported temporal lengths are `8k + 1` frames through 241 frames (about ten seconds at
24 fps).

The Diffusers import remains lazy so `create_app()`, help, session description, and CPU
stand-in tests do not import or load the model. The official checkpoint ID and revision
are immutable constants. Prompt enhancement is disabled because it requires a separate
Gemma checkpoint and is outside the concise baseline.

## Immutable inputs

| Input | Revision |
| --- | --- |
| FlashDreams base, `origin/main` | `24f50da7ef26c6dd7fad44209e918ab700285648` |
| Synchronized audio prerequisite, PR #526 | `264840e2aa8807b76f2c274b3b32266915b13ec6` |
| Post-merge V2 target | `a947768b8a3521feb974286a88af07e24c13850f` |
| Named V2 integration procedure | `integrate-a-model-v2-skill-pr/skills/integrate-a-model-v2/SKILL.md` |
| Lightricks reference source | `a95ab856bf29407b6b066ede0abe1846050db56c` |
| Diffusers `main` with unreleased LTX 2.5 support | `119c339551f68ea523b9f204120b929e56342421` |
| `Lightricks/LTX-2.5` model repository | `6c7e5e573ac1667efc83407806fe9b0b93730e60` |
| `Lightricks/LTX-2.5-Diffusers` checkpoint | `426936f8b22dc28e4def61e515478b0b7e4a53cc` |

The checkpoint is gated by the LTX 2 Community License. No Lightricks source or model
weights are redistributed by FlashDreams. At plan time, the host's Hugging Face account
was authenticated but had not accepted access to the Diffusers checkpoint (HTTP 403).
Access was accepted before GPU validation. The first real load reconstructed 154 GB of
the pinned snapshot from about 134 GB transferred and then reused the local cache.

## Architecture boundary

```text
flashdreams-run-v2 t2av-ltx25
  -> LTX25Application (arguments, one shared backend)
    -> LTX25Session (fresh per-run state)
      -> LTX25ModelLoop (one finite generation step)
        -> DiffusersLTX25Backend (joint video/audio tensors)
          -> StepResult(video, AudioOutput)
            -> PR #526 MP4 sink (encode, reconcile, mux, publish atomically)
```

The backend protocol is injectable. CPU tests use a deterministic stand-in that returns
changing RGB frames and stereo PCM; the shipped zero-argument entry point uses the lazy
Diffusers backend.

## Validation ladder

1. CPU argument, description, lifecycle, state-isolation, reset, shape/range, audio, and
   metrics contracts.
2. CPU stand-in through `ApplicationRunner` and the synchronized MP4 sink, followed by
   FFmpeg/FFprobe decode and exact video frame-count checks.
3. Wheel builds for FlashDreams and this adapter, installation into a clean environment,
   entry-point discovery, and delegated help.
4. Pinned checkpoint download and strict Diffusers component load.
5. Minimal 9-frame eager/offload GPU smoke, then one 25-frame warmup excluded from
   headline measurements.
6. Diverse prompt/duration/resolution matrix below, with fresh sessions sharing one
   loaded application where practical.
7. External stream decode, codec/profile, frame/sample counts, A/V drift, silence,
   finite checks, SHA-256, visual review, HTML gallery, and best-video selection.

## GPU matrix

| Label | Frames | Resolution | Prompt focus | Purpose |
| --- | ---: | ---: | --- | --- |
| `sync_percussion_short` | 25 | 768x512 | visible drum strikes and sharp transients | Short-duration A/V synchronization |
| `dialogue_portrait` | 25 | 960x544 | close dialogue with ambient room tone | Alternate aspect and speech |
| `ocean_wide` | 25 | 1280x736 | breaking waves, gulls, and wind | Largest resolution boundary |
| `train_medium` | 121 | 768x512 | moving train with wheel rhythm and horn | Canonical five-second motion/audio |
| `market_medium` | 121 | 960x544 | busy market with voices and footsteps | Canonical higher-resolution diversity |
| `multishot_max` | 241 | 768x512 | connected ten-second multishot with sound continuity | Maximum-duration boundary |

The reduced 9- and 25-frame runs prove loading, temporal accounting, and mux behavior;
they are not canonical quality or performance claims. Headline performance comes from
post-warmup canonical runs only. Every run records exact command and commit; seed,
dimensions, frames, fps, offload mode, model load and generation time, generation FPS,
peak CUDA memory, output path, media properties, and hashes.

## Acceptance

- All CPU and packaging checks pass under the repository-pinned tools.
- The application is discoverable as `t2av-ltx25` and delegated help imports no model.
- Generated video is finite, nonblank, changing, complete, and matches requested geometry.
- Generated audio is finite stereo PCM in `[-1, 1]`, not silent, and is attached exactly
  once at sample offset zero.
- Final MP4 has H.264 video and AAC-LC stereo audio, with no missing frames and A/V drift
  bounded by one AAC frame after the runtime reconciles the timeline.
- Repeated same-seed canonical runs are compared by SHA-256; fresh sessions do not share
  mutable completion state.
- Performance claims identify this host and do not generalize beyond the measured stack.

## Results

| Layer | Status | Evidence |
| --- | --- | --- |
| PR #526 focused CPU tests | Passed | 112 tests passed in 29.64 s |
| Adapter CPU contracts | Passed | 23 CPU tests passed in 3.66 s |
| Full runtime V2 CPU regression | Passed | 363 tests passed in 38.46 s |
| Synchronized stand-in MP4 | Passed | H.264/AAC probe, exact 25 frames, decoded audible 48 kHz stereo |
| Lint/type checks | Passed | pre-commit Ruff format/import checks and workspace `ty` |
| Wheel/install/entry point | Passed | sdist/wheel built with the Diffusers Git SHA in `METADATA`; entry point and delegated help passed |
| Matrix/gallery harness | Passed | Two distinct stand-in cases, one backend load, inspected MP4s, JSON and HTML |
| Checkpoint access/load | Passed | Gate accepted; pinned snapshot downloaded and loaded from the official mirror |
| Minimal real-model A/V smoke | Passed | 9 frames at 384x256; stereo 48 kHz audio; 4.8 s denoising, 42.41 s cached end to end |
| GPU matrix | Passed | 6/6 real cases; all codec, geometry, frame, signal, motion, and drift checks passed |
| Gallery and selected MP4s | Passed | Portable HTML/JSON plus six MP4s; train, market, and maximum-duration greenhouse clips selected |

### Real GPU matrix

The matrix ran on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with model
offload, BF16 weights, one shared model load, 24 fps output, and the local pinned
checkpoint cache. The exact invocation was:

```bash
HF_HUB_OFFLINE=1 .venv/bin/flashdreams-ltx25-benchmark \
  --output-dir /home/jmccaffrey/projects/ltx25-gallery-20260827 \
  --offload model --local-files-only --continue-on-error
```

| Case | Output | Generation | Throughput | Peak CUDA | Wall | Audio RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sync_percussion_short` | 25 · 768x512 | 48.81 s | 0.512 fps | 41.91 GiB | 53.86 s | 0.2418 |
| `dialogue_portrait` | 25 · 960x544 | 72.73 s | 0.344 fps | 41.98 GiB | 74.22 s | 0.1572 |
| `ocean_wide` | 25 · 1280x736 | 35.00 s | 0.714 fps | 42.22 GiB | 36.62 s | 0.0275 |
| `train_medium` | 121 · 768x512 | 43.14 s | 2.805 fps | 42.57 GiB | 48.78 s | 0.1516 |
| `market_medium` | 121 · 960x544 | 50.23 s | 2.409 fps | 42.85 GiB | 56.07 s | 0.0163 |
| `multishot_max` | 241 · 768x512 | 66.21 s | 3.640 fps | 43.39 GiB | 77.12 s | 0.0265 |

Every artifact contains the exact requested frame count, H.264 video, AAC stereo at
48 kHz, finite audible audio, changing video, and the requested dimensions. Video and
audio duration differ by 0.000667 s in every case, well below one 48 kHz AAC frame.
The shared model loaded in 3.42 s from the warm cache. Visual contact-sheet review found
coherent motion and identity in all cases; `train_medium`, `market_medium`, and
`multishot_max` were selected for their longer duration and visual coherence, while
`sync_percussion_short` remains the focused transient-synchronization stress case.

Artifacts are outside the Git worktree at
`/home/jmccaffrey/projects/ltx25-gallery-20260827`. `manifest.json` records prompts,
seeds, revisions, runtime measurements, media properties, all checks, paths, sizes, and
SHA-256 digests; `gallery.html` embeds all six MP4 files with controls.
