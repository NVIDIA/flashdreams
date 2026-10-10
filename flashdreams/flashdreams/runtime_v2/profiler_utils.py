# SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Choosing and binding the profiler the inference API runs under."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

import torch

from flashdreams.api_v2.profiler import IProfiler
from flashdreams.runtime_v2.profiler import (
    CompositeProfiler,
    CudaEventProfiler,
    FrameRateProfiler,
    InputLatencyProfiler,
    NullProfiler,
    NVTXProfiler,
)

_INFERENCE_PROFILER: ContextVar[IProfiler | None] = ContextVar(
    "flashdreams_inference_profiler", default=None
)
_DEFAULT_PROFILER: IProfiler | None = None
_DEFAULT_LOCK = threading.Lock()


def get_inference_profiler() -> IProfiler:
    """Return the profiler the inference API runs under.

    An application sets one for its sessions. Code that runs a pipeline without
    a session -- the batch runner, serving, benchmarks -- falls back to what the
    environment asks for, which is what it got before this interface existed.
    """
    profiler = _INFERENCE_PROFILER.get()
    if profiler is not None:
        return profiler
    global _DEFAULT_PROFILER
    if _DEFAULT_PROFILER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_PROFILER is None:
                _DEFAULT_PROFILER = create_profiler()
    return _DEFAULT_PROFILER


def _reset_default_profiler() -> None:
    """Drop the cached environment default. For tests that change the env."""
    global _DEFAULT_PROFILER
    _DEFAULT_PROFILER = None


@contextmanager
def set_flashdreams_inference_profiler(profiler: IProfiler) -> Iterator[IProfiler]:
    """Run the block with ``profiler`` as the inference API's profiler."""
    token = _INFERENCE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _INFERENCE_PROFILER.reset(token)


def bind_inference_profiler(profiler: IProfiler) -> Token[IProfiler | None]:
    """Bind ``profiler`` for the calling thread, which keeps its own context.

    A ``ContextVar`` set on one thread is invisible to another, so the model
    thread binds the profiler the session injected into its loop. Pass the token
    back to :func:`unbind_inference_profiler`: a loop driven on a caller's thread
    would otherwise leave the binding behind.
    """
    return _INFERENCE_PROFILER.set(profiler)


def unbind_inference_profiler(token: Token[IProfiler | None]) -> None:
    """Restore whatever the context held before the matching bind."""
    _INFERENCE_PROFILER.reset(token)


def create_profiler() -> IProfiler:
    """Build the profiler a session runs with, from the environment.

    ``FLASHDREAMS_NVTX`` adds Nsight Systems ranges,
    ``FLASHDREAMS_SYNC_AND_PROFILE`` adds the CUDA-event stage timings that
    ``finalize`` returns, ``FLASHDREAMS_FPS`` adds the frame rates the model loop
    reports, and ``FLASHDREAMS_INPUT_LATENCY`` adds input latency beside them.
    None set means nothing is recorded.
    """
    profilers: list[IProfiler] = []
    if os.environ.get("FLASHDREAMS_NVTX") == "1" and torch.cuda.is_available():
        profilers.append(NVTXProfiler())
    if os.environ.get("FLASHDREAMS_SYNC_AND_PROFILE") == "1":
        profilers.append(CudaEventProfiler())
    if os.environ.get("FLASHDREAMS_FPS") == "1":
        profilers.append(FrameRateProfiler())
    if os.environ.get("FLASHDREAMS_INPUT_LATENCY") == "1":
        profilers.append(InputLatencyProfiler())
    if not profilers:
        return NullProfiler()
    if len(profilers) == 1:
        return profilers[0]
    return CompositeProfiler(tuple(profilers))
