Infra
===================================

The ``flashdreams.infra`` package defines the swappable abstractions that
every recipe plugs into: a config system, an encoder / diffusion-model /
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

.. autoclass:: TransformerConfig
   :members:

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

.. autoclass:: SchedulerConfig
   :members:

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

Encoders turn raw conditioning (text prompts, reference images, …) into
the latent tensors consumed by the transformer. Like every other
component, they are stateful across AR steps via an
:class:`EncoderAutoregressiveCache`.

.. currentmodule:: flashdreams.infra.encoder

.. autoclass:: EncoderConfig
   :members:

.. autoclass:: Encoder
   :members:

.. autoclass:: EncoderAutoregressiveCache
   :members:

.. autoclass:: NullEncoderConfig
   :members:

.. autoclass:: NullEncoder
   :members:

Decoder
-------

Decoders turn the latents emitted by the diffusion model back into pixel
frames. They keep a :class:`DecoderAutoregressiveCache` so the streaming
pipeline can decode one chunk at a time with the correct temporal
context.

.. currentmodule:: flashdreams.infra.decoder

.. autoclass:: DecoderConfig
   :members:

.. autoclass:: Decoder
   :members:

.. autoclass:: DecoderAutoregressiveCache
   :members:
