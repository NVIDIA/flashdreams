---
title: 'Core API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

The `flashdreams.core` package collects the low-level attention primitives,
KV-cache implementations, and process-group utilities that integrations share.

## Attention

The attention package provides the kernels used by the transformer and
the block-structured KV cache that backs streaming inference.

<a id="flashdreams.core.attention.NativeAttention"></a>
### `NativeAttention`

```text
class NativeAttention ( * args : Any , ** kwargs : Any )
```

Bases: `Module`

Native attention module with configurable QKV layout and SDPA backend.

<a id="flashdreams.core.attention.NativeAttention.set_context_parallel_group"></a>
#### `set_context_parallel_group`

```text
set_context_parallel_group ( cp_group : ProcessGroup | None ) → None
```

Enable or disable context parallelism for ring attention.

##### ``

```text
Parameters :
```

**cp\_group** – Process group for context parallel; use None to disable.
<a id="flashdreams.core.attention.NativeAttention.is_context_parallel_enabled"></a>
#### `is_context_parallel_enabled`

```text
is_context_parallel_enabled ( ) → bool
```

Return True if context parallelism is active.
<a id="flashdreams.core.attention.NativeAttention.context_parallel_size"></a>
#### `context_parallel_size`

```text
context_parallel_size ( ) → int
```

Return the context parallel world size, or 1 if disabled.
<a id="flashdreams.core.attention.NativeAttention.forward"></a>
#### `forward`

```text
forward ( query : torch.Tensor , key : torch.Tensor , value : torch.Tensor ) → torch.Tensor
```

Run context-parallel SDPA (or single-rank SDPA when CP is disabled).

##### ``

```text
Parameters :
```

* **query** – Query tensor in configured `qkv_format`.
* **key** – Key tensor in configured `qkv_format`.
* **value** – Value tensor in configured `qkv_format`.

<a id="flashdreams.core.attention.ContextParallelAttention"></a>
### `ContextParallelAttention`

```text
class ContextParallelAttention ( * args : Any , ** kwargs : Any )
```

Bases: `NativeAttention`

Context-parallel attention with selectable method and SDPA backend.

<a id="flashdreams.core.attention.BlockKVCache"></a>
### `BlockKVCache`

```text
class BlockKVCache ( k_shape : tuple [ int , ... ] , v_shape : tuple [ int , ... ] , seq_dim : int , chunk_size : int , window_size : int , sink_size : int = 0 , device : torch.device | str = torch.device , dtype : torch.dtype = torch.float16 , _prev_chunk_idx : int = -1 , _curr_chunk_idx : int | None = None , _n_cached : int = 0 )
```

Bases: `object`

KV cache for causal attention with a fixed-size local window, CUDA-graph compatible.

Keys and values can have arbitrary shape `[..., total_size, ...]`; the sequence
(rolling) dimension is given by `seq_dim` (dimension index, can be negative).
Layout along that dimension: [sink tokens | local window tokens]. Sink tokens are
never evicted; the local window rolls left as new chunks are added if full. Chunks are
non-overlapping: each update adds one chunk of `chunk_size` tokens at the
next logical position in the full sequence.

#### ``

```text
Phases :
```

* **- Filling** – cache not yet full; tokens are written contiguously;
  `cached_k()` / `cached_v()` return only the valid prefix.
* **- Steady-state** – if adding a chunk would exceed the fixed cache size, the
  local window rolls left by the overflow amount and the new chunk
  overwrites the rightmost positions; `cached_k()` / `cached_v()`
  return the full buffer.
The argument `chunk_idx` (0, 1, 2, …) is the index of the new chunk in the full
sequence (not an index into the cache). If `chunk_idx` is greater than
the previous one, the chunk is appended (or, in steady-state, written after
the roll). If `chunk_idx` equals the previous one, the same cache positions
are overwritten.

#### ``

```text
Per-step usage:
```

1. before\_update(chunk\_idx) — prepare (roll local window if steady-state).
2. update(k, v) — write the new chunk’s keys/values into the cache.
3. cached\_k() / cached\_v() — get cached keys/values for attention.
4. after\_update(chunk\_idx) — update internal bookkeeping.
<a id="flashdreams.core.attention.BlockKVCache.k_shape"></a>
#### `k_shape`

```text
k_shape : tuple [ int , ... ]
```

Shape of the keys. Must be the same as the values shape except for the last dimension.
<a id="flashdreams.core.attention.BlockKVCache.v_shape"></a>
#### `v_shape`

```text
v_shape : tuple [ int , ... ]
```

Shape of the values. Must be the same as the keys shape except for the last dimension.
<a id="flashdreams.core.attention.BlockKVCache.seq_dim"></a>
#### `seq_dim`

```text
seq_dim : int
```

Sequence dimension that will be rolled. Can be negative.
<a id="flashdreams.core.attention.BlockKVCache.chunk_size"></a>
#### `chunk_size`

```text
chunk_size : int
```

Number of tokens processed each time.
<a id="flashdreams.core.attention.BlockKVCache.window_size"></a>
#### `window_size`

```text
window_size : int
```

Size of the local attention window (excluding sink tokens).
<a id="flashdreams.core.attention.BlockKVCache.sink_size"></a>
#### `sink_size`

```text
sink_size : int = 0
```

Number of sink tokens at the start of the cache that are never evicted. Defaults to 0.
<a id="flashdreams.core.attention.BlockKVCache.device"></a>
#### `device`

```text
device : torch.device | str = 'cuda'
```

Device to store the cache on.
<a id="flashdreams.core.attention.BlockKVCache.dtype"></a>
#### `dtype`

```text
dtype : torch.dtype
```

Data type to store the cache in.
<a id="flashdreams.core.attention.BlockKVCache.size"></a>
#### `size`

```text
property size : int
```

Number of valid cached tokens visible to attention.
<a id="flashdreams.core.attention.BlockKVCache.write_end"></a>
#### `write_end`

```text
property write_end : int
```

Right edge of the current chunk in the physical cache layout.
<a id="flashdreams.core.attention.BlockKVCache.from_tensor"></a>
#### `from_tensor`

```text
classmethod from_tensor ( k : torch.Tensor , v : torch.Tensor , seq_dim : int ) → Self
```

Build a single-chunk cache pre-filled with the given key and value tensors.
<a id="flashdreams.core.attention.BlockKVCache.is_steady_state"></a>
#### `is_steady_state`

```text
is_steady_state ( ) → bool
```

Return True if the cache is full (steady-state phase).
<a id="flashdreams.core.attention.BlockKVCache.before_update"></a>
#### `before_update`

```text
before_update ( chunk_idx : int ) → None
```

Prepare the cache before writing new tokens.

If `chunk_idx` equals the previous chunk index, this is a no-op. Otherwise,
we expect the `chunk_idx` to be +1 from the previous chunk index. In this case,
we will roll the local window left if the cache is in steady-state, or no op
if the cache is in filling phase.

##### ``

```text
Parameters :
```

**chunk\_idx** – Chunk index of the new chunk in the full sequence.
<a id="flashdreams.core.attention.BlockKVCache.update"></a>
#### `update`

```text
update ( k : torch.Tensor , v : torch.Tensor ) → None
```

Write the new chunk’s keys and values into the cache.

Must be called after `before_update()` and before `after_update()`.

##### ``

```text
Parameters :
```

* **k** – Keys; shape must match cached keys except at seq\_dim, where length must be chunk\_size.
* **v** – Values; shape must match cached values except at seq\_dim, where length must be chunk\_size.
<a id="flashdreams.core.attention.BlockKVCache.after_update"></a>
#### `after_update`

```text
after_update ( chunk_idx : int ) → None
```

Finalize bookkeeping after writing new tokens.

Updates `_prev_chunk_idx` and, in filling phase, `_n_cached`.

##### ``

```text
Parameters :
```

**chunk\_idx** – The index of the new chunk in the full sequence.
<a id="flashdreams.core.attention.BlockKVCache.cached_k"></a>
#### `cached_k`

```text
cached_k ( ) → torch.Tensor
```

Return cached keys for attention (valid prefix in filling phase, full buffer in steady-state).
<a id="flashdreams.core.attention.BlockKVCache.cached_v"></a>
#### `cached_v`

```text
cached_v ( ) → torch.Tensor
```

Return cached values for attention (valid prefix in filling phase, full buffer in steady-state).
<a id="flashdreams.core.attention.BlockKVCache.reset"></a>
#### `reset`

```text
reset ( ) → None
```

Reset bookkeeping while preserving the allocated tensor storage.
<a id="flashdreams.core.attention.BlockKVCache.clone_kv"></a>
#### `clone_kv`

```text
clone_kv ( ) → tuple [ torch.Tensor , torch.Tensor ]
```

Return clones of the full physical K/V buffers.
<a id="flashdreams.core.attention.BlockKVCache.overwrite_kv_"></a>
#### `overwrite_kv_`

```text
overwrite_kv_ ( k : torch.Tensor , v : torch.Tensor ) → None
```

Overwrite the full K/V buffers without changing their addresses.

##### ``

```text
Parameters :
```

* **k** – Replacement keys with the exact cache shape.
* **v** – Replacement values with the exact cache shape.

## Distributed

Helpers for multi-GPU and multi-node inference. `init` boots the NCCL
process group with NVML-derived CPU affinity, a configurable heartbeat timeout,
and a larger L2 fetch granularity.

<a id="flashdreams.core.distributed.init"></a>
### `init`

```text
init ( ) → int | None
```

Initialize distributed training.

<a id="flashdreams.core.distributed.Device"></a>
### `Device`

```text
class Device ( device_idx : int )
```

Bases: `object`

Lightweight wrapper around an NVML device handle for CPU-affinity queries.

<a id="flashdreams.core.distributed.Device.get_name"></a>
#### `get_name`

```text
get_name ( ) → str
```

Return the marketing name reported by NVML for this device.
<a id="flashdreams.core.distributed.Device.get_cpu_affinity"></a>
#### `get_cpu_affinity`

```text
get_cpu_affinity ( ) → list [ int ]
```

Return the indices of CPUs ideally affined to this GPU per NVML.
