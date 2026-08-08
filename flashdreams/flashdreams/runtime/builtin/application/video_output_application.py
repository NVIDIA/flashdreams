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


"""Application base with video-artifact output handling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from flashdreams.runtime.application import Application, ApplicationConfig
from flashdreams.runtime.builtin.inference_output.handler.video_output_handler import (
    VideoOutputHandler,
)
from flashdreams.runtime.inference_runtime import InferenceRuntime

RuntimeT = TypeVar("RuntimeT", bound=InferenceRuntime)
"""Inference-runtime type owned by the video-output application."""


@dataclass(kw_only=True)
class VideoOutputApplicationConfig(ApplicationConfig[RuntimeT], Generic[RuntimeT]):
    """Configuration for an application that writes video output."""

    _target: type["VideoOutputApplication"] = field(
        default_factory=lambda: VideoOutputApplication
    )

    artifact_path: str | Path
    """Destination written by the video output handler."""


class VideoOutputApplication(Application[RuntimeT], Generic[RuntimeT]):
    """Application base that collects inference frames into a video artifact."""

    _inference_output_handler: VideoOutputHandler
    """Video handler constructed by the output initialization hook."""

    def _initialize_inference_output_handler(
        self, config: ApplicationConfig[RuntimeT]
    ) -> VideoOutputHandler:
        """Construct the video output handler from application configuration.

        Args:
            config: Application configuration accepted for the base hook contract.

        Returns:
            Handler configured with the artifact destination.

        Raises:
            TypeError: The application was not given video-output configuration.
        """
        if not isinstance(config, VideoOutputApplicationConfig):
            raise TypeError(
                "VideoOutputApplication requires VideoOutputApplicationConfig; "
                f"got {type(config).__name__}"
            )
        return VideoOutputHandler(config.artifact_path)

    def run(self) -> None:
        """Run inference and finish the video artifact after input exhaustion."""
        super().run()
        self._inference_output_handler.finish()


__all__ = ["VideoOutputApplication", "VideoOutputApplicationConfig"]
