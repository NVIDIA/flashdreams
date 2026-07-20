# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional acceleration policy helpers."""

from flashdreams.infra.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.acceleration.encoder_lifecycle import (
    collect_and_release_cuda_memory,
    ensure_one_shot_encoder,
    move_tensors_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
    setup_one_shot_encoder,
)
from flashdreams.infra.acceleration.frame_prefetch import (
    CudaHostPrefetch,
    LazyCudaFrame,
    prefetch_to_numpy,
)
from flashdreams.infra.acceleration.overlap import (
    CudaStreamOverlap,
    HostThreadOverlap,
    SynchronousOverlap,
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
    "CudaHostPrefetch",
    "CudaStreamOverlap",
    "HostThreadOverlap",
    "LazyCudaFrame",
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "SynchronousOverlap",
    "collect_and_release_cuda_memory",
    "cuda_graph_capture_ar_index",
    "cuda_graph_prewarm_steps",
    "ensure_one_shot_encoder",
    "is_warmup_index",
    "move_tensors_to_cpu",
    "prefetch_to_numpy",
    "release_one_shot_encoder_references",
    "run_prewarm_sequence",
    "run_one_shot_encoder_stage",
    "run_timed_prewarm",
    "setup_one_shot_encoder",
]
