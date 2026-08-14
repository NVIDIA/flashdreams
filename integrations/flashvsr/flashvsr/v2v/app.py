# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashVSR video-to-video application factory."""

from typing import Any

from v2v import V2VApplication, V2VApplicationDefaults, V2VApplicationSession

from flashdreams.demo import IFlashDreamsApplication
from flashdreams.infra.config import derive_config
from flashvsr.config import RUNNER_FLASHVSR_V1_1_SPARSE_2_0


def _pipeline_config_for_video(config: Any, height: int, width: int) -> Any:
    """Adapt FlashVSR's placeholder pipeline dimensions to the source video."""
    target_height = ((height * config.encoder.scale) // 128) * 128
    target_width = ((width * config.encoder.scale) // 128) * 128
    if target_height == 0 or target_width == 0:
        raise ValueError("FlashVSR input dimensions must scale to at least 128 pixels.")
    topk_ratio = 2.0 * 768 * 1280 / (target_height * target_width)
    return derive_config(
        config,
        encoder={"input_H": height, "input_W": width},
        diffusion_model={"transformer": {"topk_ratio": topk_ratio}},
    )


class FlashVSRV2VApplication(V2VApplication):
    """FlashVSR video super-resolution application."""

    session_type = V2VApplicationSession

    def __init__(self) -> None:
        super().__init__(
            defaults=V2VApplicationDefaults(
                pipeline_config=RUNNER_FLASHVSR_V1_1_SPARSE_2_0.pipeline,
                pipeline_config_for_video=_pipeline_config_for_video,
                first_chunk_frames=13,
                chunk_frames=16,
                default_input_height=704,
                default_input_width=1280,
                model_name="FlashVSR-v1.1",
            )
        )


def create_app() -> IFlashDreamsApplication:
    """Create the FlashVSR video-to-video application."""
    return FlashVSRV2VApplication()


__all__ = ["FlashVSRV2VApplication", "create_app"]
