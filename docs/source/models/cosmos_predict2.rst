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

Cosmos-Predict2.5
===================================
(TODO: To be updated)

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/nvidia-cosmos/cosmos-predict2.5" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://huggingface.co/nvidia/Cosmos-Predict2.5-2B" target="_blank" rel="noopener noreferrer">Model page</a>
   </div>

Cosmos-Predict2.5 is integrated as a bidirectional video model family in
FlashDreams, with text-to-video and image-to-video runner presets backed by the
shared FlashDreams pipeline and scheduler contracts.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/cosmos_predict2

Running the method
------------------

To run Cosmos-Predict2.5, launch one of the registered runner slugs via
``flashdreams-run``:

.. code-block:: bash

   uv run flashdreams-run \
       cosmos2-t2v-2b-720p \
       --prompt "A robot arm welding in a clean industrial lab."

Image-to-video:

.. code-block:: bash

   uv run flashdreams-run \
       cosmos2-i2v-2b-720p \
       --prompt "A robot arm welding in a clean industrial lab." \
       --image-path /path/to/first_frame.jpg

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/cosmos_predict2 flashdreams-run \
       cosmos2-t2v-2b-720p \
       --help

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``cosmos2-t2v-2b-720p``
     - Cosmos-Predict2.5 2B T2V at 720p, prompt-only.
   * - ``cosmos2-i2v-2b-720p``
     - Cosmos-Predict2.5 2B I2V at 720p, prompt plus first-frame image.

What FlashDreams accelerates
----------------------------

FlashDreams runs Cosmos-Predict2.5 through the same config, scheduler,
transformer, and decoder contracts used by other integrations. That makes the
model available through the unified CLI and programmatic pipeline path while
keeping the model-specific runner logic isolated in the integration package.
