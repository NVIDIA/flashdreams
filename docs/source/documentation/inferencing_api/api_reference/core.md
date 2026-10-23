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

::: flashdreams.core.attention.NativeAttention

::: flashdreams.core.attention.ContextParallelAttention

::: flashdreams.core.attention.BlockKVCache

## Distributed

Helpers for multi-GPU / multi-node inference. `init` boots the NCCL process
group with sensible defaults and replaces the boilerplate at the top of
example launchers.

::: flashdreams.core.distributed.init

::: flashdreams.core.distributed.Device
