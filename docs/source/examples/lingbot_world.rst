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

Lingbot-World
===================================

Camera-controlled image-to-video with the Lingbot-World recipe.
Reference:
`lingbot-world fast inference <https://github.com/robbyant/lingbot-world?tab=readme-ov-file#fast-inference>`_.

Shipped as the out-of-tree ``flashdreams-lingbot`` plugin under
``integrations/lingbot``. It registers two runner slugs with the
unified ``flashdreams-run`` CLI: ``lingbot-world-fast`` (Wan VAE) and
``lingbot-world-fast-flash`` (LightTAE decoder, tighter streaming
window). Pass ``--example-data`` to lazy-sync the bundled prompt +
first-frame + camera arrays from S3 into
``assets/example_data/lingbot_world/`` and fill the path defaults.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

Single GPU
----------

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

Multi GPU
---------

Wan 2.1 context parallel assumes ``cp_size == world_size``; launch via
``torchrun --no-python``:

.. code-block:: bash

   uv run torchrun --nproc_per_node=2 --no-python flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21
