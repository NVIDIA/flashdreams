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

Inference pipeline overview
===================================

This page outlines the major computation flow in the FlashDreams inference pipeline.
It outlines the core concepts and APIs for building custom model integrations,
or modifying existing ones.

.. Figure creation trace: https://chatgpt.com/share/6a14ab0f-90bc-83e8-8145-c1a03b64f43a

.. image:: /_static/diagrams/flashdreams-inference-pipeline-overview.jpg
   :alt: FlashDreams autoregressive inference pipeline overview.
   :class: zoomable

The key entry point for the inference pipeline is the
:class:`~flashdreams.infra.pipeline.StreamInferencePipeline` class, which defines the
autoregressive generation loop shown in the figure. The persistent state is held in
:class:`~flashdreams.infra.pipeline.StreamInferencePipelineCache` as a cache object,
which is shared within the pipeline and updated in each autoregressive step.

.. code-block:: python

   from flashdreams.infra.pipeline import (
       StreamInferencePipeline,
       StreamInferencePipelineCache,
   )

   pipeline: StreamInferencePipeline = ...

   # One-shot encoding on global conditions, then initialize cache/state
   cache: StreamInferencePipelineCache = pipeline.initialize_cache(
       text=["a beautiful beach scene"],
       image=first_frame,
       ...,
   )

   # Autoregressive generation loop
   for autoregressive_index, control in enumerate(controls):
       current_output = pipeline.generate(autoregressive_index, cache, input=control)
       yield current_output
       pipeline.finalize(autoregressive_index, cache)

The code snippet above shows the basic execution loop. The initial call :meth:`~flashdreams.infra.pipeline.StreamInferencePipeline.initialize_cache`
consumes global conditions, such as text prompts and the first frame. Then, for each autoregressive step,
:meth:`~flashdreams.infra.pipeline.StreamInferencePipeline.generate` is called to produce the current output chunk,
followed by :meth:`~flashdreams.infra.pipeline.StreamInferencePipeline.finalize`. This split exists because
:meth:`~flashdreams.infra.pipeline.StreamInferencePipeline.finalize` typically handles additional KV
cache updates that are not in the hot path, which can be offloaded to a background thread in many
cases to hide latency.

Inside :meth:`~flashdreams.infra.pipeline.StreamInferencePipeline.generate`, the pipeline encodes the
per-step control input, runs the diffusion model's denoising loop, and decodes the latent chunk into
the final output. The following snippet illustrates this internal flow:

.. code-block:: python

   # class StreamInferencePipeline
   def generate(
       autoregressive_index: int, cache, input=None,
   ) -> torch.Tensor:
       # 1. Convert per-step control into model conditioning
       if input is not None:
           input = pipeline.encoder(
               input=input,
               autoregressive_index=autoregressive_index,
               cache=cache.encoder_cache,
           )

       # 2. Run scheduler loop + DiT flow prediction
       clean_latent, final_state = diffusion_model.generate(
           autoregressive_index=autoregressive_index,
           cache=cache.transformer_cache,
           input=input,
       )
       cache.final_state = final_state

       # 3. Convert latent chunk to output chunk
       if pipeline.decoder is None:
           return clean_latent

       return pipeline.decoder(
           input=clean_latent,
           autoregressive_index=autoregressive_index,
           cache=cache.decoder_cache,
       )

In FlashDreams, these components are composed using a configuration system. This allows building
customized pipelines by supplying different configurations for the encoder, diffusion model, and decoder.
A typical :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig` is instantiated as follows:

.. code-block:: python

   from flashdreams.infra.diffusion.model import DiffusionModelConfig
   from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
   from flashdreams.infra.pipeline import StreamInferencePipelineConfig

   # Define your own configs for the encoder, transformer, and decoder
   CustomizedStreamingEncoderConfig = ...
   CustomizedTransformerConfig = ...
   CustomizedStreamingDecoderConfig = ...

   # create a pipeline config
   pipeline_config = StreamInferencePipelineConfig(
       name="customized-method-name",
       encoder=MyStreamingEncoderConfig(),
       diffusion_model=DiffusionModelConfig(
           transformer=MyTransformerConfig(),
           scheduler=FlowMatchSchedulerConfig(),
       ),
       decoder=MyStreamingDecoderConfig(),
   )

   # then a pipeline can be simply instantiated as follows:
   pipeline = pipeline_config.setup().to("cuda").eval()

More details on the config system can be found in :doc:`/developer_guides/config_system`.

Examples
--------

Samples on how existing models use this structure:

- `LingBot-World config <https://github.com/NVIDIA/flashdreams/blob/main/integrations/lingbot/lingbot/config.py>`_:
  A camera-controlled I2V model that uses the per-step camera encoder.
- `Self-Forcing config <https://github.com/NVIDIA/flashdreams/blob/main/integrations/self_forcing/self_forcing/config.py>`_:
  A pure T2V model that sets ``encoder=None``, so each rollout starts from noise.
- `NVIDIA OmniDreams config <https://github.com/NVIDIA/flashdreams/blob/main/integrations/omnidreams/omnidreams/config.py>`_:
  An I2V video model with a VAE-based causal encoder for HDMap control.
- `Wan2.1 config <https://github.com/NVIDIA/flashdreams/blob/main/integrations/wan21/wan21/config.py>`_:
  Treats a bidirectional video model as a single-rollout autoregressive model.

For the detailed API documentation, please reference :doc:`/api/infra`. To integrate a new model,
please refer to :doc:`/developer_guides/new_integration`.
