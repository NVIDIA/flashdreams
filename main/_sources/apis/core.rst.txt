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

Core
===================================

The ``flashdreams.core`` package collects the low-level kernels and
process-group utilities that recipes share.

Attention
---------

The attention package provides the kernels used by the transformer and
the block-structured KV cache that backs streaming inference.

.. currentmodule:: flashdreams.core.attention

.. autoclass:: NativeAttention
   :members:

.. autoclass:: RingAttention
   :members:

.. autoclass:: BlockKVCache
   :members:

Distributed
-----------

Helpers for multi-GPU / multi-node inference. ``init`` boots the NCCL
process group with sensible defaults (NVML-derived CPU affinity,
heartbeat timeout, larger L2 fetch granularity) and is a drop-in for
the boilerplate at the top of the example launchers.

.. currentmodule:: flashdreams.core.distributed

.. autofunction:: init

.. autoclass:: Device
   :members:
