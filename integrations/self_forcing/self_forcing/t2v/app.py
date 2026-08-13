# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default Self-Forcing model."""

from t2v import create_t2v_application, model_config_from_runner

from flashdreams.demo import Application
from self_forcing.config import RUNNER_WAN21_T2V_1PT3B

MODEL = model_config_from_runner(
    model_id="self-forcing-t2v",
    runner=RUNNER_WAN21_T2V_1PT3B,
)


def create_app() -> Application:
    """Create the default Self-Forcing T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
