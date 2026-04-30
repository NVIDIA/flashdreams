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
