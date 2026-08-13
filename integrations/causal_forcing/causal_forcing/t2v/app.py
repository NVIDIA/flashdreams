# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default Causal-Forcing model."""

from flashdreams.demo import Application
from t2v import T2VModelConfig, create_t2v_application

from causal_forcing.config import PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE
from causal_forcing.runner import DEFAULT_T2V_PROMPT

MODEL = T2VModelConfig(
    model_id="causal-forcing-t2v",
    preset_id=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE.name,
    pipeline=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
    prompt=DEFAULT_T2V_PROMPT,
    total_blocks=60,
    pixel_height=480,
    pixel_width=832,
    fps=16,
)


def create_app() -> Application:
    """Create the default Causal-Forcing T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
