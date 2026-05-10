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

FastVideo CausalWan 2.2
===================================

FastVideo CausalWan 2.2 14B MoE distilled streaming T2V, shipped as the
out-of-tree ``flashdreams-fastvideo-causal-wan22`` plugin (see
``integrations/fastvideo_causal_wan22/``) and driven by the unified
``flashdreams-run`` CLI under the slug
``fastvideo-causal-wan2.2-t2v-14b``. Reference: `FastVideo
basic_self_forcing_causal_wan2_2_i2v.py
<https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py>`_.

T2V only for now: the FastVideo Wan 2.2 checkpoint's I2V protocol
(one-shot first-frame VAE-seed warmup) does not fit the unified
streaming pipeline's per-AR-step mask-injection I2V and is not wired
here.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b --total-blocks 21
