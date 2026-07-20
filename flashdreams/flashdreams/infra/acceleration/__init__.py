# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional acceleration policy helpers."""

from flashdreams.infra.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)

__all__ = [
    "CUDAGraphDispatch",
    "cuda_graph_capture_ar_index",
]
