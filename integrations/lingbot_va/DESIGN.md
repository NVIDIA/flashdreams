<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-VA V2 design record

## Inputs and references

- Target baseline: FlashDreams `8fd97fa38f04bc32c288760fa0fbf5da52464cea`.
- Draft integration: PR #312,
  `f98cae4a18ddf6c189a6cfa2099265d6d570e337`.
- V2 reference application: `integrations_v2/red_screen` plus the finite
  `color_fade` loop.
- Upstream inference reference:
  `robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
- Checkpoint snapshot:
  `robbyant/lingbot-va-posttrain-robotwin@8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`.

## ADR-1: session-owned destructive engine

Accepted: each session owns one `LingbotVAEngine`; the application owns only an
immutable config and an engine factory.

The VAE decode cannot fit alongside the full DiT/text/cache footprint on the
supported capacity path. Generation therefore releases KV, DiT, tokenizer,
text encoder, and streaming-encoder caches before moving the VAE to the decode
device. That transition is destructive. Application-owned reusable model state
would promise reuse that the implementation cannot honor.

Reset closes the current engine and clears the loop's finished flag. The next
step constructs a fresh engine lazily. Close is idempotent. A failed run closes
partial state while preserving the original inference exception even when
cleanup also fails.

## ADR-2: generic typed action artifacts

Accepted: extend the generic V2 result/session/sink contracts with named tensor
artifacts rather than hiding actions in LingBot-specific files or metadata.

`SessionDesc.tensor_artifact_schemas` declares `actions[step, channel]`.
`StepResult.tensor_artifacts` carries the tensor. The generic
`TensorArtifactOutputSink` concatenates declared chunks and atomically writes
`actions.npy`. LingBot code never imports that sink and never chooses an output
path.

## ADR-3: one honest model step with deferred decode

Accepted: the first V2 version generates N dual-stream chunks, releases
denoising state, decodes accumulated video frame-by-frame, and returns one
`StepResult`.

This keeps the UI thread independent of model execution without claiming
per-chunk presentation. A streaming cadence can be added only after a measured
decode path fits without invalidating cache/model ownership.

## State machine

`NEW -> RUNNING -> FINISHED -> CLOSED` is the successful engine path.
Any exception from `RUNNING` triggers cleanup and transitions to `CLOSED`.
Calling `run` outside `NEW` is an error. Session reset replaces the closed or
finished engine with a new `NEW` instance on the next model step.

## Memory ownership

| Phase | GPU/active | CPU/host | Released at boundary |
| --- | --- | --- | --- |
| load | DiT; optionally VAE/T5 | tokenizer; offloaded components | partial state on failure |
| encode | T5 then VAE as needed | three input PNGs | prompt/observation temporaries |
| denoise | DiT, CFG caches, latent/action state | accumulated completed chunks | per-step temporaries |
| teardown | VAE only after transfer | DiT/T5/tokenizer references | all KV and denoising state |
| decode | VAE plus one decoded frame | accumulated output frames/actions | VAE cache and each GPU frame |
| finished | none | returned video/actions/metrics | all model components |

## Fixed contracts

- Robotwin layout: high camera full resolution above two half-resolution wrists.
- Video: TCHW, 256x320 high-camera crop, 10 FPS, float `[-1, 1]`. The VAE
  decodes `2N` latent frames to `8N - 3` pixel frames.
- Actions: 32 steps per chunk, 16 channels in order
  `0..6, 28, 7..13, 29`.
- Default CFG: video scale 5, action scale 1. Conditional and unconditional
  branches own distinct video KV and both branches advance whenever a CFG cache
  exists. Every action denoise pass attends to committed prior chunks plus the
  matching branch's current video KV before its own fresh action KV.
- Cache attention window: 72, matching pinned upstream Robotwin config.
- Checkpoints: local root or revision-aware Hugging Face snapshot with explicit
  component subfolders.

## Deliberate exclusions

- no legacy V1 runner or `flashdreams.runner_configs` entry point;
- no application-owned files, MP4 encoder, threads, or model components;
- no multi-GPU/FSDP claim;
- no speedup claim without matched-output evidence;
- no root CUDA/Torch policy change.
