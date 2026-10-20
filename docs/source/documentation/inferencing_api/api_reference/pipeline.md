---
title: 'Pipeline API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

The `flashdreams.infra` package defines the swappable abstractions that
every integration plugs into: a config system, an encoder / diffusion-model /
decoder triple, and the streaming inference pipeline that drives them.

See the [Inferencing API overview](../index.md) for public runtime boundaries.

## Config

Infra component configs are mutable `InstantiateConfig` dataclasses
built via `config.setup()`. `PrintableConfig` provides readable
configuration dumps, while `derive_config` deep-copies an existing
config before applying nested overrides.

::: flashdreams.infra.config.PrintableConfig

::: flashdreams.infra.config.InstantiateConfig

::: flashdreams.infra.config.derive_config

## Pipeline

The pipeline composes the model-side streaming inference stages. It
autoregressively generates one chunk at a time by running an optional encoder,
the diffusion model, and an optional decoder, with per-session caches for the
configured streaming components. Call `finalize()` after `generate()` to
advance the diffusion model's autoregressive cache.

::: flashdreams.infra.pipeline.StreamInferencePipelineConfig

::: flashdreams.infra.pipeline.StreamInferencePipeline

::: flashdreams.infra.pipeline.StreamInferencePipelineCache

## Diffusion model

The diffusion model wraps a transformer backbone with a denoising scheduler.
Callers see only noise to clean latent conversion; the per-step flow prediction
and iteration loop remain inside `DiffusionModel.generate`.

::: flashdreams.infra.diffusion.model.DiffusionModelConfig

::: flashdreams.infra.diffusion.model.DiffusionModel

## Transformer

::: flashdreams.infra.diffusion.transformer.TransformerConfig

::: flashdreams.infra.diffusion.transformer.Transformer

::: flashdreams.infra.diffusion.transformer.TransformerAutoregressiveCache

## Schedulers

A scheduler owns the denoising loop. Its internal operations broadcast
per-step scalar sigmas, so the same scheduler works for any latent layout.

::: flashdreams.infra.diffusion.scheduler.Scheduler

::: flashdreams.infra.diffusion.scheduler.SchedulerConfig

::: flashdreams.infra.diffusion.scheduler.FlowPredictor

::: flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig

::: flashdreams.infra.diffusion.scheduler.FlowMatchScheduler

::: flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig

::: flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig

::: flashdreams.infra.diffusion.scheduler.FlowMatchUniPCScheduler

## Encoder

Encoders turn raw conditioning into latent tensors. `Encoder` is stateless and
one-shot, while `StreamingEncoder` carries per-step state. The video variant
adds spatial and temporal size contracts.

::: flashdreams.infra.encoder.EncoderConfig

::: flashdreams.infra.encoder.Encoder

::: flashdreams.infra.encoder.StreamingEncoder

::: flashdreams.infra.encoder.StreamingVideoEncoder

::: flashdreams.infra.encoder.StreamingEncoderCache

::: flashdreams.infra.encoder.NullEncoderConfig

::: flashdreams.infra.encoder.NullEncoder

## Decoder

Decoders turn emitted latents into frames. `StreamingDecoder` carries
per-rollout state; the video variant adds spatial and temporal size contracts.

::: flashdreams.infra.decoder.DecoderConfig

::: flashdreams.infra.decoder.StreamingDecoder

::: flashdreams.infra.decoder.StreamingVideoDecoder

::: flashdreams.infra.decoder.StreamingDecoderCache
