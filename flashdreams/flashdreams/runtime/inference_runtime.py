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

"""Distributed inference runtime with shared pipeline ownership."""

import os
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime.inference_session import InferenceSession


def _is_torchrun_env() -> bool:
    """Return whether ``torchrun`` set the distributed rendezvous variables."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


SessionT = TypeVar("SessionT", bound=InferenceSession)
"""Session type parameter for :class:`InferenceRuntime`."""


class InferenceRuntime(ABC, Generic[SessionT]):
    """Shared pipeline runtime for distributed inference sessions.

    Construction initializes PyTorch distributed when launched by ``torchrun``,
    records rank metadata, and constructs one pipeline shared by every session.
    The concrete session type associates the runtime with its pipeline type, so
    callers only parameterize the runtime with ``SessionT``.
    Subclasses implement :meth:`warmup` for integration-specific model execution.
    """

    # ---------------- PyTorch Distributed State  ---------------- #

    local_rank: int
    """Process-local rank; ``0`` outside distributed runs."""

    global_rank: int
    """Global process rank; ``0`` outside distributed runs."""

    world_size: int
    """Number of distributed processes; ``1`` outside distributed runs."""

    is_rank_zero: bool
    """Whether this process is the global rank-zero process."""

    pipeline: StreamInferencePipeline
    """Pipeline constructed once and shared by all sessions."""

    session_type: type[SessionT]
    """Concrete session type created by :meth:`create_session`."""

    def __init__(
        self,
        pipeline_config: StreamInferencePipelineConfig,
        session_type: type[SessionT],
    ) -> None:
        """Initialize distributed state and construct the shared pipeline.

        Args:
            pipeline_config: Pipeline configuration to instantiate.
            session_type: Concrete session type to create.
        """
        # Initialize before pipeline construction so context-parallel components
        # observe torchrun's world size while allocating their runtime state.
        if _is_torchrun_env() and not torch.distributed.is_initialized():
            init_distributed()

        # Snapshot launch metadata for rank-gated runtime work while preserving
        # stable single-process defaults for ordinary Python processes.
        if torch.distributed.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
        self.is_rank_zero = self.global_rank == 0

        self.pipeline = pipeline_config.setup()
        self.session_type = session_type

    def create_session(self) -> SessionT:
        """Create a session backed by the shared pipeline.

        Returns:
            Fresh session with its own pipeline cache.
        """
        return self.session_type(self.pipeline)

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the pipeline for inference."""
