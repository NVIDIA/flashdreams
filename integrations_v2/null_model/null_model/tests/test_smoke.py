# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


def test_smoke() -> None:
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()
    output = pipeline.generate(0, cache, input=torch.tensor([[1]]))
    assert output.shape == (1, 3, 1, 1, 1)
    assert torch.all(output == 1)
