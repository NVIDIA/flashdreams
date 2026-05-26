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

Config system
===================================

FlashDreams configuration is built around simple, strongly-typed Python ``dataclass`` objects.
This configuration system is similar to the one employed in `nerfstudio <https://github.com/nerfstudio-project/nerfstudio>`_. It allows you to easily plug in different permutations of model components and nest configurations to define the complete inference pipeline.

Base components
---------------

All configurable components in FlashDreams—such as the encoder, transformer, scheduler, and decoder—have a corresponding configuration dataclass and can be found under ``flashdreams.infra``. As outlined in the :doc:`/developer_guides/inference_pipeline_overview`, the main entry point for defining an integration is the :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`.

These config objects are modular and nestable. A typical pipeline config defines the architecture by composing other config dataclasses:

.. code-block:: python

   from flashdreams.infra.diffusion.model import DiffusionModelConfig
   from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
   from flashdreams.infra.pipeline import StreamInferencePipelineConfig

   # Define your own configs for the encoder, transformer, and decoder.
   MyStreamingEncoderConfig = ...
   MyTransformerConfig = ...
   MyStreamingDecoderConfig = ...

   # Compose them into a pipeline config
   pipeline_config = StreamInferencePipelineConfig(
       name="customized-method-name",
       encoder=MyStreamingEncoderConfig(),
       diffusion_model=DiffusionModelConfig(
           transformer=MyTransformerConfig(),
           scheduler=FlowMatchSchedulerConfig(),
       ),
       decoder=MyStreamingDecoderConfig(),
   )

Creating new configs
--------------------

If you are interested in creating a brand new model component, you will need to create a corresponding config with the associated parameters you want to expose.

Let's say you want to create a new encoder called ``MyEncoder``. You can create a new ``Encoder`` class that extends the base class. Before the model definition, you define the actual ``MyEncoderConfig`` which points to the ``MyEncoder`` class using the ``_target`` field.

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Type
   from flashdreams.infra.encoder.base import EncoderConfig, Encoder

   @dataclass(kw_only=True)
   class MyEncoderConfig(EncoderConfig):
       """My custom encoder config."""

       # Point to the class that will be instantiated by this config
       _target: Type = field(default_factory=lambda: MyEncoder)

       # Expose your configurable parameters
       embedding_dim: int = 512
       num_layers: int = 6

   class MyEncoder(Encoder):
       """My custom encoder model.

       Args:
           config: Configuration to instantiate the encoder.
       """

       # Enable type checking
       config: MyEncoderConfig

       def __init__(self, config: MyEncoderConfig) -> None:
           super().__init__(config)
           # Build your layers using self.config.embedding_dim, etc.
           ...

       def forward(self, input):
           ...

Alternatively, you do not always have to write a complete configuration from scratch. You can use :func:`flashdreams.infra.config.derive_config` to create concise variants from existing configs. This allows you to inherit the base settings and only override the specific fields you want to change:

.. code-block:: python

   from flashdreams.infra.config import derive_config
   from my_project.configs import MyBasePipelineConfig

   # Create a variant that inherits everything from MyBasePipelineConfig
   # but overrides the encoder's embedding dimension.
   my_variant_config = derive_config(
       MyBasePipelineConfig,
       encoder=dict(embedding_dim=1024),
   )

Modifying from CLI
------------------

Often times, you just want to play with the parameters of an existing model without having to specify a new one. You can easily do so via the CLI, which is powered by `tyro <https://github.com/brentyi/tyro>`_.

Because our configurations are strongly typed dataclasses, ``tyro`` automatically generates a comprehensive command-line interface. You can use the ``flashdreams-run`` (The CLI carried in FlashDreams) command to dynamically override any nested dataclass field.

For example, to list out all existing configurable parameters for a model:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --help

.. image:: /_static/diagrams/cli-screen-shot.png
   :alt: CLI helptext showing tyro dynamically parsing nested configuration arguments.


To run the model with a modified configuration:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash \
       --pipeline.diffusion-model.transformer.use-cuda-graph True \
       --total-blocks 7

For full details on the available commands, see the :doc:`/api/cli` reference.
For end-to-end examples of defining custom pipeline configurations, see :doc:`/developer_guides/new_integration`.
