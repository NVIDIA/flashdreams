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

Overview
--------

This integration brings FastVideo's causal Wan2.2 T2V variant into the
FlashDreams streaming runtime for consistent CLI and benchmarking workflows.

Links
-----

- Project page: `FastVideo GitHub <https://github.com/hao-ai-lab/FastVideo>`_
- Reference script: `FastVideo Wan2.2 inference example <https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py>`_
- Integration package: `flashdreams/integrations/fastvideo_causal_wan22 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/fastvideo_causal_wan22>`_

Run this model
--------------

.. code-block:: bash

   uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b --total-blocks 21
