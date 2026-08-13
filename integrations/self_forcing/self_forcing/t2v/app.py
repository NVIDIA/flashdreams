# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default Self-Forcing model."""

from flashdreams.demo import Application
from t2v import T2VModelConfig, create_t2v_application

from self_forcing.config import PIPELINE_WAN21_T2V_1PT3B
from self_forcing.runner import DEFAULT_T2V_PROMPT

MODEL = T2VModelConfig(
    model_id="self-forcing-t2v",
    preset_id=PIPELINE_WAN21_T2V_1PT3B.name,
    pipeline=PIPELINE_WAN21_T2V_1PT3B,
    prompt=DEFAULT_T2V_PROMPT,
    total_blocks=60,
    pixel_height=480,
    pixel_width=832,
    fps=16,
)


def create_app() -> Application:
    """Create the default Self-Forcing T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
