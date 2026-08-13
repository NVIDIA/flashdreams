# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default Cosmos Predict2 model."""

from flashdreams.demo import Application
from t2v import T2VModelConfig, create_t2v_application

from cosmos_predict2.config import PIPELINE_COSMOS2_T2V_2B_720P
from cosmos_predict2.runner import DEFAULT_PROMPT

MODEL = T2VModelConfig(
    model_id="cosmos-predict2-t2v",
    preset_id=PIPELINE_COSMOS2_T2V_2B_720P.name,
    pipeline=PIPELINE_COSMOS2_T2V_2B_720P,
    prompt=DEFAULT_PROMPT,
    total_blocks=1,
    pixel_height=720,
    pixel_width=1280,
    fps=16,
)


def create_app() -> Application:
    """Create the default Cosmos Predict2 T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
