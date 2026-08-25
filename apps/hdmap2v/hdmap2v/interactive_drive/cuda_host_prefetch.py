# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the shared CUDA host frame-prefetch helper."""

from __future__ import annotations

from flashdreams.infra.acceleration.frame_prefetch import CudaHostPrefetch

__all__ = ["CudaHostPrefetch"]
