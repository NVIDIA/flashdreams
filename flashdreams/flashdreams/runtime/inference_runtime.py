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
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import torch
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import InstantiateConfig
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

RuntimeT = TypeVar("RuntimeT", bound="InferenceRuntime")
"""Runtime type constructed by :class:`InferenceRuntimeConfig`."""


@dataclass(kw_only=True)
class InferenceRuntimeConfig(InstantiateConfig, Generic[RuntimeT]):
    """Configuration for constructing an inference runtime."""

    _target: type[RuntimeT]

    pipeline: StreamInferencePipelineConfig
    """Pipeline configuration instantiated and shared by runtime sessions."""

    session_type: type[InferenceSession]
    """Concrete session type created by the runtime."""

    def setup(self, **kwargs: Any) -> RuntimeT:
        """Construct the configured inference runtime.

        Args:
            **kwargs: Additional constructor arguments for the runtime.

        Returns:
            Configured inference runtime.
        """
        return self._target(self, **kwargs)


class InferenceRuntime(ABC, Generic[SessionT]):
    """Shared pipeline runtime for distributed inference sessions.

    Construction initializes PyTorch distributed when launched by ``torchrun``,
    records rank metadata, and constructs one pipeline shared by every session.
    The concrete session type associates the runtime with its pipeline type, so
    callers only parameterize the runtime with ``SessionT``.
    Subclasses implement :meth:`warmup` for integration-specific model execution.
    """

    ## Distributed state

    _local_rank: int
    """Process-local rank; ``0`` outside distributed runs."""

    _global_rank: int
    """Global process rank; ``0`` outside distributed runs."""

    _world_size: int
    """Number of distributed processes; ``1`` outside distributed runs."""

    _is_rank_zero: bool
    """Whether this process is the global rank-zero process."""

    _pipeline: StreamInferencePipeline
    """Pipeline constructed once and shared by all sessions."""

    _session_type: type[SessionT]
    """Concrete session type created by :meth:`create_session`."""

    def __init__(
        self,
        config: InferenceRuntimeConfig,
    ) -> None:
        """Initialize distributed state and construct the shared pipeline.

        Args:
            config: Runtime construction configuration.
        """
        # Initialize before pipeline construction so context-parallel components
        # observe torchrun's world size while allocating their runtime state.
        if _is_torchrun_env() and not torch.distributed.is_initialized():
            init_distributed()

        # Snapshot launch metadata for rank-gated runtime work while preserving
        # stable single-process defaults for ordinary Python processes.
        if torch.distributed.is_initialized():
            self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self._global_rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._local_rank = 0
            self._global_rank = 0
            self._world_size = 1
        self._is_rank_zero = self._global_rank == 0

        self._pipeline = config.pipeline.setup()
        self._session_type = cast(type[SessionT], config.session_type)

    def create_session(self) -> SessionT:
        """Create a session backed by the shared pipeline.

        Returns:
            Fresh session with its own pipeline cache.
        """
        return self._session_type(self._pipeline)

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the pipeline for inference."""
