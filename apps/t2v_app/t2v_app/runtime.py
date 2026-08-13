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

"""Text-to-video model runtime and one-time pipeline state."""

from __future__ import annotations

from typing import Any

import torch

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime import InferenceInput
from flashdreams_runner import AppConfig, IOHandler, Runtime, Session

from .session import T2VSession, T2VSessionDefaults


class T2VRuntime(Runtime):
    """Own T2V model weights and create isolated generation sessions."""

    def __init__(
        self,
        *,
        pipeline_config: StreamInferencePipelineConfig,
        session_defaults: T2VSessionDefaults,
        config: AppConfig,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._session_defaults = session_defaults
        self._config = config
        self._pipeline: StreamInferencePipeline[Any, Any, Any] | None = None
        self._io_handler: IOHandler | None = None

    @property
    def config(self) -> AppConfig:
        """Return T2V configuration for runner-owned presentation."""
        return self._config

    def initialize(self, *, device: str, io_handler: IOHandler) -> None:
        """Construct model weights once for the selected device and I/O mode."""
        if self._pipeline is not None:
            raise RuntimeError("T2VRuntime is already initialized.")
        pipeline = self._pipeline_config.setup()
        if not isinstance(pipeline, StreamInferencePipeline):
            raise TypeError(
                "T2V pipeline config must construct StreamInferencePipeline, got "
                f"{type(pipeline).__name__}."
            )
        self._pipeline = pipeline.to(device).eval()
        self._io_handler = io_handler

    def create_session(self, initial_input: InferenceInput | None = None) -> Session:
        """Create a T2V session with its own prompt and autoregressive cache."""
        if self._pipeline is None:
            raise RuntimeError("T2VRuntime must be initialized before use.")
        return T2VSession(
            pipeline=self._pipeline,
            defaults=self._session_defaults,
            initial_input=initial_input or InferenceInput(),
            output_layout=self._config.output_layout,
        )

    def destroy(self) -> None:
        """Release pipeline weights and accelerator allocator state."""
        pipeline = self._pipeline
        self._pipeline = None
        self._io_handler = None
        if pipeline is None:
            return
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["T2VRuntime"]
