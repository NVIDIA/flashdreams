# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omnidreams-specific model instantiation tests.

Split out from ``flashdreams/tests/test_model_instantiation.py`` when
Omnidreams moved out of the in-tree integration set; the wan/cosmos/umt5
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
        from omnidreams.impl.encoder.pixel_shuffle import (
            PixelShuffleVAEEncoderConfig,
        )

        model = PixelShuffleVAEEncoderConfig().setup().to(device)

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8
