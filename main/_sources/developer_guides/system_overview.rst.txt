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

System overview
===================================

FlashDreams separates reusable runtime orchestration from model-specific
implementation. The infra layer defines the contracts for runners, pipelines,
encoders, diffusion models, transformers, schedulers, decoders, and caches.
Recipes and integrations fill those contracts with concrete model code.

.. Figure creation trace: https://chatgpt.com/share/6a12a13f-19f0-83e8-bf02-f02e6c236fb5

.. image:: /_static/diagrams/system_overview.jpg
   :alt: FlashDreams serving session lifecycle with initialize, generate, finalize, and persistent cache state.

The serving session runs one ``initialize_cache`` phase, then repeats
``generate`` and ``finalize``. ``generate`` produces the next output chunk;
``finalize`` advances the world state so the next chunk continues from the same
session instead of starting over.

Computation To Infra Map
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Stage
     - Infra class / contract
     - Integration responsibility
   * - Launch and I/O
     - ``Runner`` + ``RunnerConfig``; see :doc:`/api/recipes`
     - Define slugged runner configs, usually with ``runner_name`` matching
       ``pipeline.recipe_name``. ``Runner.run`` resolves prompts, images,
       controls, output paths, device/rank behavior, and rank-zero persistence.
   * - Runtime container
     - ``StreamInferencePipeline`` + ``StreamInferencePipelineConfig``; see
       :doc:`/api/infra`
     - Provide config literals that wire the optional per-step encoder,
       ``DiffusionModel``, optional decoder, and recipe slug. Most integrations
       use the base pipeline; custom pipelines adapt signatures such as T2V/I2V
       ``height``/``width`` or first-frame inputs.
   * - Session initialization
     - ``StreamInferencePipeline.initialize_cache``
     - Build the per-rollout cache tree once. Integration pipelines prepare
       context such as text embeddings, image embeddings, latent height/width,
       first-frame state, encoder cache, decoder cache, and transformer AR/KV
       cache.
   * - One-shot context
     - ``Encoder`` used as ``transformer.context_encoder``
     - Encode global conditioning once per rollout, such as text prompts,
       negative prompts for CFG, CLIP image embeddings, or identity
       passthroughs. This belongs to transformer context, not per-step control.
   * - Per-step control
     - ``StreamingEncoder`` / ``StreamingVideoEncoder`` used as
       ``pipeline.encoder``
     - Convert each AR step's live input into model conditioning. Examples
       include camera controls, HDMap videos, image/video control chunks, and
       I2V first-frame VAE encoding with an encoder cache.
   * - Denoising loop
     - ``DiffusionModel.generate`` + ``Scheduler``
     - Choose the scheduler config and inference-step schedule. The infra model
       samples noise, runs the scheduler loop, calls transformer flow
       prediction, and returns a clean latent chunk plus ``final_state``.
   * - Model forward
     - ``Transformer.predict_flow`` and ``TransformerAutoregressiveCache``
     - Implement the concrete recipe transformer and DiT network. This is where
       patchify/unpatchify, context-parallel split/gather, RoPE, AR/KV cache,
       CFG cond/uncond branches, ``torch.compile``, and optional CUDA Graph
       replay are connected to the model.
   * - Output chunk
     - ``StreamingDecoder`` / ``StreamingVideoDecoder`` used as
       ``pipeline.decoder``
     - Convert clean latent chunks into frames or application-facing outputs.
       Video decoders expose temporal/spatial compression contracts so pipelines
       can reason about chunk sizes.
   * - State advance
     - ``DiffusionModel.finalize`` + ``Transformer.finalize_kv_cache``
     - Consume ``final_state`` from ``generate``, optionally re-noise the clean
       latent, advance transformer AR/KV cache state, and run post-update hooks.
       The next ``generate`` continues the same evolving world state.

Code Ownership
--------------

- :doc:`/api/core` owns reusable primitives such as attention, KV cache,
  context-parallel utilities, checkpoint loading, and IO helpers.
- :doc:`/api/infra` owns abstract contracts and orchestration: config,
  pipeline, diffusion model, scheduler, transformer, encoder, decoder, compile,
  CUDA graph, and serving contracts.
- :doc:`/api/recipes` documents the public pipeline/runner surface and the
  current integration map.
- :doc:`/developer_guides/usage_patterns` explains how to run existing models or
  call FlashDreams programmatically.
- :doc:`/developer_guides/new_recipes` explains how to implement and register a
  new model integration.
