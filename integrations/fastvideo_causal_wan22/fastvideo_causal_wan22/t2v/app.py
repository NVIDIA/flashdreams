# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default FastVideo CausalWan 2.2 model."""

from t2v import create_t2v_application, model_config_from_runner

from fastvideo_causal_wan22.config import RUNNER_WAN22_T2V_14B
from flashdreams.demo import Application

MODEL = model_config_from_runner(
    model_id="fastvideo-causal-wan22-t2v",
    runner=RUNNER_WAN22_T2V_14B,
)


def create_app() -> Application:
    """Create the default FastVideo CausalWan 2.2 T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
