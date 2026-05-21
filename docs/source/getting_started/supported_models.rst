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

Supported models
===================================

FlashDreams currently supports the following model families.

Autoregressive / streaming models
---------------------------------

- `Self-Forcing <https://github.com/liruilong940607/Self-Forcing>`_
- `Causal-Forcing <https://github.com/LiRunyi2001/causal-forcing>`_
- `Causal Wan2.2 <https://github.com/hao-ai-lab/FastVideo>`_
- `Lingbot-World <https://github.com/robbyant/lingbot-world>`_
- `OmniDreams <https://huggingface.co/nvidia/omni-dreams-models>`_

Bidirectional models
--------------------

- `Wan2.1 <https://github.com/Wan-Video/Wan2.1>`_
- Cosmos-Predict2

In FlashDreams, bidirectional models are executed through the same pipeline
interface and are treated as a single-rollout autoregressive run.

Contributing a new method
-------------------------

Want to add a new model or variant? Start from
:doc:`/developer_guides/new_recipes`.
