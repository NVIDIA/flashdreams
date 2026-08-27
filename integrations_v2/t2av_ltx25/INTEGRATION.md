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
| Diffusers v0.40.0 source | `d035dcd7cc7c88e0a154609b62887d50bba9fdc2` |
| `Lightricks/LTX-2.5` model repository | `6c7e5e573ac1667efc83407806fe9b0b93730e60` |
| `Lightricks/LTX-2.5-Diffusers` checkpoint | `426936f8b22dc28e4def61e515478b0b7e4a53cc` |

The checkpoint is gated by the LTX 2 Community License. No Lightricks source or model
weights are redistributed by FlashDreams. At plan time, the host's Hugging Face account
was authenticated but had not accepted access to the Diffusers checkpoint (HTTP 403).

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
| Synchronized stand-in MP4 | Passed | H.264/AAC probe, exact 25 frames, decoded audible 48 kHz stereo |
| Lint/type checks | Passed | pre-commit Ruff format/import checks and workspace `ty` |
| Wheel/install/entry point | Passed | sdist/wheel built; `t2av-ltx25` discovered; delegated help passed |
| Matrix/gallery harness | Passed | Two distinct stand-in cases, one backend load, inspected MP4s, JSON and HTML |
| Checkpoint access/load | Blocked | HF account must accept the LTX 2.5 gate |
| GPU matrix | Pending | |
| Gallery and selected MP4s | Pending | |
