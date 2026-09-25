# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import torch
from cam2v import Cam2VApplication, Cam2VApplicationDefaults

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
    PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF,
    PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF_TAEHV,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.impl.conditioning import resolve_lingbot_conditioning

LINGBOT_CAM2V_DEFAULTS = Cam2VApplicationDefaults(
    pipeline_config=derive_config(
        PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
    ),
    input_resolver=resolve_lingbot_conditioning,
    total_blocks=20,
    pixel_width=832,
    pixel_height=464,
    first_frame_dtype=torch.float32,
    first_frame_interpolation="torch_bicubic",
    fps=16,
    log_model_timing=True,
    install_hint="Install the Lingbot integration: pip install flashdreams-lingbot.",
    input_defaults={"example_data": False, "example_idx": 0},
)
"""Lingbot defaults for the reusable Cam2V application."""


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World specialization of the shared Cam2V application."""

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """Select the pipeline config used by this Cam2V application.

        Args:
            pipeline_config: Model variant to run; ``None`` uses the application
                default.
        """
        defaults = LINGBOT_CAM2V_DEFAULTS
        selected_pipeline_config = defaults.pipeline_config
        if pipeline_config is not None:
            selected_pipeline_config = derive_config(pipeline_config)
        defaults = dataclasses.replace(
            defaults,
            pipeline_config=selected_pipeline_config,
            input_resolver=partial(
                resolve_lingbot_conditioning,
                transformer_len_t=selected_pipeline_config.diffusion_model.transformer.len_t,
            ),
        )
        super().__init__(defaults=defaults)

    def _apply_parsed_arguments(self, args: Any) -> None:
        """Size an enabled reference-noise bank to this rollout."""
        transformer = self._pipeline_config.diffusion_model.transformer
        if getattr(transformer, "reference_noise_steps", None) is None:
            return
        self._pipeline_config = derive_config(
            self._pipeline_config,
            diffusion_model={
                "transformer": {"reference_noise_steps": args.total_blocks}
            },
        )


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


def create_app_fast() -> IApplication:
    """Return the Lingbot World Fast application."""
    return LingbotCam2VApplication(pipeline_config=PIPELINE_LINGBOT_WORLD_FAST)


def create_app_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World Fast bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3
    )


def create_app_v2_14b_causal_fast() -> IApplication:
    """Return the Lingbot World v2 14B Causal Fast application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST
    )


def create_app_v2_14b_causal_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World v2 bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=(PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3)
    )


def create_app_v2_1p3b_causal_fast_max_perf() -> IApplication:
    """Return the maximum-throughput Lingbot World v2 1.3B application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF
    )


def create_app_v2_1p3b_causal_fast_max_perf_taehv() -> IApplication:
    """Return the maximum-throughput 1.3B application with TAEHV decoding."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF_TAEHV
    )


__all__ = [
    "LingbotCam2VApplication",
    "create_app",
    "create_app_fast",
    "create_app_fast_taehv_window15_sink3",
    "create_app_v2_1p3b_causal_fast_max_perf",
    "create_app_v2_1p3b_causal_fast_max_perf_taehv",
    "create_app_v2_14b_causal_fast",
    "create_app_v2_14b_causal_fast_taehv_window15_sink3",
]
