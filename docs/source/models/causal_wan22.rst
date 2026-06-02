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

Causal Wan2.2
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_t2v.py" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

CausalWan2.2 is a `FastVideo <https://github.com/hao-ai-lab/FastVideo>`_-released
14B MoE causal-diffusion variant of Wan 2.2 with 8-step inference.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/fastvideo_causal_wan22

Running the method
------------------

To run Causal Wan2.2, launch the registered runner slug via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/fastvideo_causal_wan22 \
       flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

   uv run --project integrations/fastvideo_causal_wan22 \
       flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --prompt "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

For multi-GPU inference, use ``torchrun`` on top of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/fastvideo_causal_wan22 \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

We provide the following variant:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``fastvideo-causal-wan2.2-t2v-14b``
     - FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/fastvideo_causal_wan22 \
       flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid zoomable">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_1.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_2.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
       </div>
     </div>
   </div>

Citation
--------

If you use Causal Wan2.2, please cite the original work:

.. code-block:: bibtex

   @article{zhang2025fast,
     title={Fast video generation with sliding tile attention},
     author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
     journal={arXiv preprint arXiv:2502.04507},
     year={2025}
   }
