# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional acceleration policy helpers."""

from flashdreams.infra.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.acceleration.prewarm import (
    PrewarmDeadline,
    PrewarmSequenceTiming,
    PrewarmTimeoutError,
    PrewarmTiming,
    cuda_graph_prewarm_steps,
    is_warmup_index,
    run_prewarm_sequence,
    run_timed_prewarm,
)

__all__ = [
    "CUDAGraphDispatch",
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "cuda_graph_capture_ar_index",
    "cuda_graph_prewarm_steps",
    "is_warmup_index",
    "run_prewarm_sequence",
    "run_timed_prewarm",
]
