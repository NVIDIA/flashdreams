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

Self-forcing T2V (Wan2.1)
===================================

Self-forcing T2V variant of Wan2.1, shipped as the
``flashdreams-self-forcing`` integration plugin and exposed on the
unified ``flashdreams-run`` CLI under two slugs:

- ``self-forcing-wan2.1-t2v-1.3b`` — Wan VAE decoder, 4-step.
- ``self-forcing-wan2.1-t2v-1.3b-flash`` — same DiT, swapped to the
  TAEHV (LightTAE) decoder for faster decoding.

Checkpoints are auto-downloaded from HuggingFace on first run.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface  # optional

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b --total-blocks 7
