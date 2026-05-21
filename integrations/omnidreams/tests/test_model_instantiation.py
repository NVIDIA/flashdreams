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

"""Omnidreams-specific model instantiation tests.

Split out from ``flashdreams/tests/test_model_instantiation.py`` when
Omnidreams moved out of the in-tree recipe set; the wan/cosmos/umt5
tests stay in flashdreams.
"""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestVideoVAE:
    """Tests for omnidreams-shipped video VAE models."""

    @pytest.mark.ci_gpu
    def test_pixel_shuffle_vae_instantiation(self, device):
        """Test PixelShuffleVAEInterface can be instantiated."""
        from omnidreams.encoder.pixel_shuffle import (
            PixelShuffleVAEEncoderConfig,
        )

        model = PixelShuffleVAEEncoderConfig().setup().to(device)

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8
