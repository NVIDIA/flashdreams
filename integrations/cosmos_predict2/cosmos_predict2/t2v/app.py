# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public T2V app entry for the default Cosmos Predict2 model."""

from t2v import create_t2v_application, model_config_from_runner

from cosmos_predict2.config import RUNNER_COSMOS2_T2V_2B_720P
from flashdreams.demo import Application

MODEL = model_config_from_runner(
    model_id="cosmos-predict2-t2v",
    runner=RUNNER_COSMOS2_T2V_2B_720P,
)


def create_app() -> Application:
    """Create the default Cosmos Predict2 T2V application."""
    return create_t2v_application(model=MODEL)


createApp = create_app

__all__ = ["MODEL", "createApp", "create_app"]
