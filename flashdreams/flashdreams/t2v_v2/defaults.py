# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one text-to-video integration contributes to the shared application."""

from dataclasses import dataclass
from typing import Any

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_REQUIRED_RUNNER_FIELDS = ("pipeline", "pixel_height", "pixel_width")
"""Runner config fields :meth:`T2VApplicationDefaults.from_runner_config` reads."""


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VApplicationDefaults:
    """Defaults one integration supplies to the shared text-to-video application.

    Every value here is a default rather than a fixed setting: the shared
    application declares a command line flag for each, so a run can ask for
    something else. What an integration is saying is what its model generates
    when nobody asks for anything in particular.
    """

    pipeline_config: Any
    """Model to load. Owned by the integration, which knows what it is."""

    total_blocks: int
    """Steps one run generates unless it is asked for a different number.

    The session does not stop itself, so this bounds nothing on its own. It is
    what a runner uses when its caller did not say how long to generate for.
    """

    pixel_width: int
    """Frame width the model was trained at."""

    pixel_height: int
    """Frame height it was trained at."""

    device: str = "cuda"
    """Device the pipeline is built on."""

    fps: int = 16
    """Rate the generated frames are meant to play at."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Layout the pipeline emits."""

    @classmethod
    def from_runner_config(
        cls,
        runner_config: Any,
        *,
        total_blocks: int | None = None,
    ) -> "T2VApplicationDefaults":
        """Read an integration's defaults off the runner config it already has.

        A model's frame size and rate are already written down in its runner
        config, so an integration that has one says nothing twice.

        Args:
            runner_config: Runner config owned by the integration.
            total_blocks: Rollout length, when it differs from the runner's.

        Returns:
            Defaults for the shared application.

        Raises:
            TypeError: The runner config does not carry what this reads. It is
                duck-typed rather than a declared type, so this is where a
                config that cannot drive a text-to-video application is caught.
        """
        required = list(_REQUIRED_RUNNER_FIELDS)
        if total_blocks is None:
            required.append("total_blocks")
        missing = [name for name in required if not hasattr(runner_config, name)]
        if missing:
            raise TypeError(
                f"Runner config {type(runner_config).__name__} is missing "
                f"text-to-video application defaults: {missing}."
            )
        return cls(
            pipeline_config=runner_config.pipeline,
            total_blocks=(
                int(runner_config.total_blocks)
                if total_blocks is None
                else int(total_blocks)
            ),
            pixel_width=int(runner_config.pixel_width),
            pixel_height=int(runner_config.pixel_height),
            fps=int(getattr(runner_config, "fps", 16)),
            output_layout=_layout_of(runner_config),
        )


def _layout_of(runner_config: Any) -> VideoTensorLayout:
    """Return the layout a runner config declares, as a v2 layout.

    A v1 runner config spells its layout as a string, and this API uses an
    enum, so the two have to be reconciled somewhere.
    """
    declared = getattr(runner_config, "postprocess_output_layout", None)
    if declared is None:
        return VideoTensorLayout.tchw
    if isinstance(declared, VideoTensorLayout):
        return declared
    return VideoTensorLayout(str(declared))
