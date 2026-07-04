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

LingBot-World
===================================

.. container:: fd-cta-row

   .. button-link:: https://technology.robbyant.com/lingbot-world
      :color: primary

      Project page

   .. button-link:: https://github.com/robbyant/lingbot-world
      :color: primary

      Official code

Introduced by `Robbyant <https://technology.robbyant.com/>`_, LingBot-World is a camera-controllable image-to-video
(I2V) world model with streaming inference and context-parallel runtime support.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://gw.alipayobjects.com/v/huamei_u94ywh/afts/video/XQk7Rb44qJwAAAAAgfAAAAgAfoeUAQBr" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>
   <p class="model-footnote">
     Teaser video source:
     <a href="https://technology.robbyant.com/lingbot-world">LingBot-World project page</a>.
   </p>

Requirements
------------

- **Minimum VRAM**: ~120 GB.
- **PyTorch**: >= 2.9.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/lingbot

Running the method
------------------

To run LingBot-World, launch one of the registered runner slugs. For
example:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --example-idx 0 \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

Sample data is downloaded from the
`LingBot-World repository <https://github.com/Robbyant/lingbot-world/tree/main/examples>`_.
Valid ``--example-idx`` values are ``0, 1, 2, 5``. Note the single GPU command might run
out of memory for large ``--total-blocks`` values.

For multi-GPU inference, run the same command under ``torchrun`` (taking
4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/lingbot \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --example-idx 0 \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``lingbot-world-fast``
     - Official camera-control I2V (Wan VAE decoder, full KV-cache).
   * - ``lingbot-world-fast-taehv-window15-sink3``
     - Efficient streaming configuration: TAEHV decoder, ``window_size_t=15``
       + ``sink_size_t=3`` streaming KV-cache.

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run \
       lingbot-world-fast \
       --help

What to expect
--------------

- **Example data**: ``--example-data True`` downloads ``image.jpg``,
  ``intrinsics.npy``, ``poses.npy``, ``prompt.txt`` from the
  `upstream examples folder <https://github.com/Robbyant/lingbot-world/tree/main/examples>`_
  into ``assets/example_data/lingbot_world/<NN>/`` (``<NN>`` matches
  ``--example-idx``). Cached after first run; no credentials needed.
- **Model checkpoint**: ~70 GB pulled from
  ``huggingface.co/robbyant/lingbot-world-fast`` on first run, cached
  under ``$HF_HOME``. Export ``HF_TOKEN`` first.
- **Disk**: keep ~200 GB free for the model + HF cache. Hosts under
  ~100 GB have been seen to run out mid-load.
- **First launch**: a few minutes (download + Triton autotuning +
  CUDA-graph warmup). Subsequent launches reuse the caches.
- **Outputs**: ``outputs/<runner-slug>.mp4`` (16 FPS, 464×832 by
  default) and ``outputs/stats_<runner-slug>.json``. Override with
  ``--output-dir`` / ``--pixel-height`` / ``--pixel-width`` / ``--fps``.

See :doc:`/developer_guides/inference_pipeline_overview` for what one
autoregressive chunk does end-to-end.

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid zoomable">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <video autoplay muted loop playsinline preload="metadata" style="position: absolute; right: 10px; bottom: 10px; width: 33.3333%; opacity: 0.7; border-radius: 8px; pointer-events: none;">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-01.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_idx: 01
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-02.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <video autoplay muted loop playsinline preload="metadata" style="position: absolute; right: 10px; bottom: 10px; width: 33.3333%; opacity: 0.7; border-radius: 8px; pointer-events: none;">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-02.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_idx: 02
       </div>
     </div>
   </div>


Launch the interactive server
-----------------------------

Spin up the interactive LingBot-World server via WebRTC:

.. code-block:: bash

   # from the repo root
   uv run --package flashdreams-lingbot torchrun --nproc_per_node 4 \
       -m lingbot.webrtc.server \
       --host 0.0.0.0 --port 8089 \
       --config_name lingbot-world-fast-taehv-window15-sink3 \
       --example-idx 0

``--example-idx`` selects which example to download
(``0``, ``1``, ``2``, ``5``); assets auto-download on first launch.
The HTTP port opens only after model load + warmup — a few minutes on
first launch, much faster afterwards. When ready the server prints
``Connect via http://<server-ip>:8089/request_session`` (use
``localhost`` when running locally).

.. note::

   On a remote or cloud GPU instance (e.g. `Brev <https://www.brev.dev/>`_),
   the HTTP server port is usually not reachable at the host IP directly.
   Forward or expose the HTTP port for the viewer page and signaling, and open
   ``http://localhost:8089/request_session`` when using a local forward:

   .. code-block:: bash

      # Brev
      brev port-forward <instance> -p 8089:8089
      # or plain SSH
      ssh -L 8089:localhost:8089 <user>@<host>

   These commands expose only the HTTP/signaling path; they do not carry the
   WebRTC media path. LingBot-World does not deploy TURN by default, so remote
   deployments that need relay or UDP media connectivity must provide it
   separately. See :ref:`webrtc-troubleshooting`.

When successfully connected, the browser-based UI looks like this:

.. raw:: html

  <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-webrtc-recording-0529.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

Profiling benchmark
-------------------

Here is the profiling benchmark on total DiT runtime for FlashDreams LingBot-World
compared to the `official LingBot-World implementation <https://github.com/robbyant/lingbot-world>`_
and `LightX2V <https://github.com/ModelTC/lightx2v>`_ under
matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="lingbot-world-benchmark-chart"
       class="benchmark-figure"
      data-benchmark-md-url="../_static/performance/lingbot_world/perf-0521.md"
      data-benchmark-series="official:Official Impl:#3b82f6;lightx2v:LightX2V:#f59e0b;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="LingBot-World benchmark chart"
     ></div>
     <figcaption>
       <p class="model-footnote">
         This chart shows total DiT runtime (4 diffusion steps) in milliseconds at the 6th autoregressive rollout on 4x GPUs.
         For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings,
         and all runs use Ulysses sequence parallelism for multi-GPU inference.
         For the official LingBot-World implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot/tests/parity_check">this instruction</a>.
         For the LightX2V baseline, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot/tests/baseline_lightx2v">this instruction</a>.
       </p>
     </figcaption>
   </figure>
  <script src="../_static/js/benchmark_chart.js"></script>

Citation
--------

If you use LingBot-World, please cite the original work:

.. code-block:: bibtex

   @article{lingbot-world,
         title={Advancing Open-source World Models},
         author={Robbyant Team and Zelin Gao and Qiuyu Wang and Yanhong Zeng and Jiapeng Zhu and Ka Leong Cheng and Yixuan Li and Hanlin Wang and Yinghao Xu and Shuailei Ma and Yihang Chen and Jie Liu and Yansong Cheng and Yao Yao and Jiayi Zhu and Yihao Meng and Kecheng Zheng and Qingyan Bai and Jingye Chen and Zehong Shen and Yue Yu and Xing Zhu and Yujun Shen and Hao Ouyang},
         journal={arXiv preprint arXiv:2601.20540},
         year={2026}
   }
