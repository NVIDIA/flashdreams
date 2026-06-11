.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Infra
===================================

The ``flashdreams.infra`` package defines the swappable abstractions that
every integration plugs into: a config system, an encoder / diffusion-model /
decoder triple, and the streaming inference pipeline that drives them.

Config
------

Every component is built from a frozen :class:`InstantiateConfig`
dataclass via ``config.setup()``. This makes the full configuration
tree printable, hashable, and trivially serialisable.

.. currentmodule:: flashdreams.infra.config

.. autoclass:: PrintableConfig
   :members:

.. autoclass:: InstantiateConfig
   :members:

.. autofunction:: derive_config

Pipeline
--------

The pipeline is the top-level streaming inference loop. It autoregressively
generates one chunk of latent video at a time by running the encoder, the
diffusion model, and the decoder back-to-back, threading per-chunk caches
through every component.

.. currentmodule:: flashdreams.infra.pipeline

.. autoclass:: StreamInferencePipelineConfig
   :members:

.. autoclass:: StreamInferencePipeline
   :members:

.. autoclass:: StreamInferencePipelineCache
   :members:

Diffusion model
---------------

Wraps a transformer backbone with a denoising scheduler. Callers see
only ``noise → clean_latent``; the per-step flow prediction and the
iteration loop are hidden inside
:meth:`~flashdreams.infra.diffusion.model.DiffusionModel.generate`.

.. currentmodule:: flashdreams.infra.diffusion.model

.. autoclass:: DiffusionModelConfig
   :members:

.. autoclass:: DiffusionModel
   :members:

Transformer
-----------

.. currentmodule:: flashdreams.infra.diffusion.transformer

.. autoclass:: Transformer
   :members:

.. autoclass:: TransformerAutoregressiveCache
   :members:

Schedulers
----------

A scheduler owns the entire denoising loop. It is shape-agnostic: every
internal op is a broadcast against per-step scalar sigmas, so the same
scheduler works for any latent layout.

.. currentmodule:: flashdreams.infra.diffusion.scheduler

.. autoclass:: Scheduler
   :members:

.. autoclass:: FlowPredictor
   :members:

.. autoclass:: FlowMatchSchedulerConfig
   :members:

.. autoclass:: FlowMatchScheduler
   :members:

.. autoclass:: FlowMatchUniPCSchedulerConfig
   :members:

.. autoclass:: FlowMatchUniPCScheduler
   :members:

Encoder
-------

Encoders turn raw conditioning (text prompts, reference images, per-AR-step
control inputs, …) into latent tensors. Two flavours:

- :class:`Encoder` is stateless and one-shot. ``forward(self, input)``.
  Used as ``transformer.context_encoder`` for text / CLIP-image / identity.
- :class:`StreamingEncoder` is stateful and per-AR-step. ``forward(self,
  input, autoregressive_index, cache)`` with an
  :class:`StreamingEncoderCache`. Used as ``pipeline.encoder`` for
  per-step control (HDMap, camera trajectory, I2V first-frame VAE).
- :class:`StreamingVideoEncoder` extends :class:`StreamingEncoder` with
  the contracts a streaming pixel-video encoder always needs: spatial /
  temporal compression ratios plus AR-step-aware temporal size mappers
  between pixel and latent space.

.. currentmodule:: flashdreams.infra.encoder

.. autoclass:: Encoder
   :members:

.. autoclass:: StreamingEncoder
   :members:

.. autoclass:: StreamingVideoEncoder
   :members:

.. autoclass:: StreamingEncoderCache
   :members:

.. autoclass:: NullEncoderConfig
   :members:

.. autoclass:: NullEncoder
   :members:

Decoder
-------

Decoders turn the latents emitted by the diffusion model back into pixel
frames. Single base class with two specialisations:

- :class:`StreamingDecoder` is stateful. ``forward(self, input,
  autoregressive_index, cache)`` with a :class:`StreamingDecoderCache`.
  Use for chunk-by-chunk streaming decoders (e.g. WAN VAE that maintains
  a temporal cache across AR steps); stateless decoders just return an
  empty :class:`StreamingDecoderCache` from
  :meth:`StreamingDecoder.initialize_autoregressive_cache` and ignore
  ``autoregressive_index`` / ``cache`` in ``forward``.
- :class:`StreamingVideoDecoder` extends :class:`StreamingDecoder` with
  the contracts a streaming pixel-video decoder always needs: spatial /
  temporal compression ratios plus AR-step-aware temporal size mappers
  between latent and pixel space.

.. currentmodule:: flashdreams.infra.decoder

.. autoclass:: StreamingDecoder
   :members:

.. autoclass:: StreamingVideoDecoder
   :members:

.. autoclass:: StreamingDecoderCache
   :members:
