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

Causal-Forcing
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://thu-ml.github.io/CausalForcing.github.io/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2602.02214" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/thu-ml/Causal-Forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Causal-Forcing uses Causal ODE or Causal Consistency Distillation to drive
asymmetric DMD as a theoretically correct initialization for real-time
interactive video generation.

.. image:: https://thu-ml.github.io/CausalForcing.github.io/images/overview.png
   :alt: Causal-Forcing overview figure.
   :width: 100%

.. raw:: html

   <p class="model-footnote">
     Teaser image source:
     <a href="https://thu-ml.github.io/CausalForcing.github.io/">Causal-Forcing project page</a>.
   </p>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/causal_forcing

Running the method
------------------

To run Causal-Forcing, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/causal_forcing \
       flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --prompt "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

For multi-GPU inference, use ``torchrun`` on top of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/causal_forcing \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --prompt "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

For I2V, run with the following command:

.. code-block:: bash

   uv run --project integrations/causal_forcing \
       flashdreams-run \
       causal-forcing-wan2.1-i2v-1.3b-framewise \
       --prompt "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle." \
       --image-path https://raw.githubusercontent.com/thu-ml/Causal-Forcing/refs/heads/main/prompts/i2v/26-15/000001.png \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``causal-forcing-wan2.1-t2v-1.3b-chunkwise``
     - Causal-Forcing chunkwise Wan 2.1 1.3B T2V (``len_t=3``).
   * - ``causal-forcing-wan2.1-t2v-1.3b-framewise``
     - Causal-Forcing framewise Wan 2.1 1.3B T2V (``len_t=1``).
   * - ``causal-forcing-wan2.1-i2v-1.3b-framewise``
     - Causal-Forcing framewise Wan 2.1 1.3B I2V (``len_t=1``).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/causal_forcing \
       flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid zoomable">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle."
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-i2v-1.3b-framewise.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle."
         <br/>
         image: https://raw.githubusercontent.com/thu-ml/Causal-Forcing/refs/heads/main/prompts/i2v/26-15/000001.png
       </div>
     </div>
   </div>

Citation
--------

If you use Causal-Forcing, please cite the original work:

.. code-block:: bibtex

   @article{zhu2026causal,
     title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
     author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
     journal={arXiv preprint arXiv:2602.02214},
     year={2026}
   }
