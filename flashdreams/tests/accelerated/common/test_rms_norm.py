# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the shared RMSNorm inference primitive."""

import pytest
import torch

from flashdreams.accelerated.common.rms_norm import rms_norm

pytestmark = pytest.mark.ci_cpu


def test_rms_norm_matches_pytorch_module() -> None:
    torch.manual_seed(0)
    module = torch.nn.RMSNorm(16, eps=1e-6)
    input = torch.randn(2, 7, 16)

    assert torch.allclose(rms_norm(input, module.weight, module.eps), module(input))


def test_rms_norm_rejects_a_mismatched_weight() -> None:
    with pytest.raises(ValueError, match="does not match"):
        rms_norm(torch.randn(2, 7, 16), torch.ones(8), 1e-6)
