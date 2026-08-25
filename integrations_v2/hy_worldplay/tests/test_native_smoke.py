# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU construction smoke test for the native HY-WorldPlay pipeline."""

import pytest
import torch

from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B

pytestmark = pytest.mark.ci_gpu


def test_native_pipeline_setup() -> None:
    """Expose the native model config without relying on a runner."""
    if not torch.cuda.is_available():
        pytest.skip("native HY-WorldPlay smoke requires CUDA")
    assert PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name == "hy-worldplay-wan-i2v-5b"
