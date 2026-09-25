# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-isolation tests for encoder configuration modules."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.ci_cpu


def test_encoder_configs_do_not_require_unused_model_implementations() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["transformers.models.clip.modeling_clip"] = None
sys.modules["transformers.models.umt5.modeling_umt5"] = None
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
assert CLIPImageEncoderConfig()._target.__name__ == "CLIPImageEncoder"
assert UMT5TextEncoderConfig()._target.__name__ == "UMT5TextEncoder"
""",
        ],
        check=True,
        timeout=60,
    )
