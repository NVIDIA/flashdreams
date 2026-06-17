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

Wan2.1
===================================

.. container:: fd-cta-row

   .. button-link:: https://wan.video/
      :color: primary

      Project page

   .. button-link:: https://arxiv.org/abs/2503.20314
      :color: primary

      arXiv paper

   .. button-link:: https://github.com/Wan-Video/Wan2.1
      :color: primary

      Official code

Wan2.1 is a bidirectional video generation model, supporting both
text-to-video (T2V) and image-to-video (I2V) tasks.

Requirements
------------

- **Minimum VRAM**: ~46 GB.
- **PyTorch**: >= 2.9.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/wan21

Running the method
------------------

To run Wan2.1, launch one of the registered runner slugs. For example:

.. code-block:: bash

   uv run --project integrations/wan21 \
       flashdreams-run \
       wan21-t2v-1.3b-480p \
       --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
       --pixel-height 832 --pixel-width 480

For multi-GPU inference, run the same command under ``torchrun`` (taking
4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/wan21 \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       wan21-t2v-1.3b-480p \
       --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
       --pixel-height 832 --pixel-width 480

For I2V, run with the following command:

.. code-block:: bash

   uv run --project integrations/wan21 \
       flashdreams-run \
       wan21-i2v-14b-480p \
       --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
       --image-path https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG \
       --pixel-height 832 --pixel-width 480

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``wan21-t2v-1.3b-480p``
     - Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only).
   * - ``wan21-i2v-14b-480p``
     - Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/wan21 \
       flashdreams-run \
       wan21-t2v-1.3b-480p \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid zoomable">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-t2v-1.3b-480p.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-i2v-14b-480p.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
         <br/>
         image: https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG
       </div>
     </div>
   </div>

Profiling benchmark
-------------------

Here is the profiling benchmark on DiT per-step runtime for FlashDreams Wan2.1
compared to the `official Wan2.1 implementation <https://github.com/Wan-Video/Wan2.1>`_
and the `FastVideo <https://github.com/hao-ai-lab/FastVideo>`_ baseline under
matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="wan21-benchmark-chart"
       class="benchmark-figure"
      data-benchmark-md-url="../_static/performance/wan21/perf-0521.md"
       data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="Wan2.1 benchmark chart"
     ></div>
     <figcaption>
      <p class="model-footnote">
         This chart shows per-diffusion-step DiT runtime in milliseconds with CFG at 480p (81 frames) on a single GPU.
         For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings.
         For the official Wan2.1 implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21/tests/parity_check">this instruction</a>.
         For the FastVideo baseline, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21/tests/baseline_fastvideo">this instruction</a>.
       </p>
     </figcaption>
   </figure>
  <script src="../_static/js/benchmark_chart.js"></script>

Citation
--------

If you use Wan2.1, please cite the original work:

.. code-block:: bibtex

   @article{wan2025,
         title={Wan: Open and Advanced Large-Scale Video Generative Models},
         author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
         journal={arXiv preprint arXiv:2503.20314},
         year={2025}
   }
