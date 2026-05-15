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

"""
Model instantiation and checkpoint loading tests.

Tests marked ``ci_gpu`` run on the GPU CI runner automatically.
Tests marked ``manual`` require large downloads or high VRAM and must
be invoked explicitly: ``pytest tests/test_model_instantiation.py -v -m manual``
"""

import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.bfloat16


class TestImageEncoder:
    """Tests for image encoders."""

    @pytest.mark.ci_gpu
    def test_wan_image_encoder_instantiation(self, device, dtype):
        """Test CLIPImageEncoder can be instantiated and encode images."""
        from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig

        image_encoder = CLIPImageEncoderConfig().setup().to(device)

        image = torch.rand(1, 2, 3, 224, 224, device=device, dtype=dtype) * 2.0 - 1.0
        image_embeds = image_encoder(image)

        assert image_embeds.shape == (1, 2, 257, 1280)
        assert image_embeds.dtype == dtype
        assert image_embeds.device.type == "cuda"


class TestTextEncoders:
    """Tests for text encoders."""

    @pytest.mark.ci_gpu
    def test_wan_text_encoder_instantiation(self, device):
        """Test UMT5TextEncoder can be instantiated and encode text."""
        from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig

        text_encoder = UMT5TextEncoderConfig().setup().to(device)

        text = ["hello world"]
        text_embeddings = text_encoder(text)

        assert text_embeddings.shape == (1, 512, 4096)
        assert text_embeddings.dtype == torch.bfloat16

    @pytest.mark.manual
    def test_cosmos_reason1_text_encoder_instantiation(self, device):
        """Test CosmosReason1TextEncoder can be instantiated and encode text.

        Kept as manual: ~20-32 GB VRAM, causes OOM on shared CI runners.
        """
        from flashdreams.infra.encoder.text.cosmos_reason1 import (
            CosmosReason1TextEncoderConfig,
        )

        text_encoder = CosmosReason1TextEncoderConfig().setup().to(device)

        text = ["A beautiful sunset over a calm ocean."]
        text_embeddings = text_encoder(text)

        # full_concat strategy: 28 layers * 3584 hidden_size = 100352
        assert text_embeddings.shape == (1, 512, 100352)
        assert text_embeddings.dtype == torch.bfloat16
        assert text_embeddings.device.type == "cuda"


class TestVideoVAE:
    """Tests for video VAE models."""

    @pytest.mark.ci_gpu
    def test_pixel_shuffle_vae_instantiation(self, device):
        """Test PixelShuffleVAEInterface can be instantiated."""
        from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
            PixelShuffleVAEEncoderConfig,
        )

        model = PixelShuffleVAEEncoderConfig().setup().to(device)

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8

    @pytest.mark.manual
    def test_teahv_vae_instantiation(self, device):
        """Test TeahvInterface can be instantiated.

        Kept as manual: default checkpoint path is on S3 (needs credentials).
        """
        from flashdreams.recipes.taehv import TeahvVAEDecoderConfig

        model = TeahvVAEDecoderConfig().setup().to(device)

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8

    @pytest.mark.manual
    def test_wan_vae_instantiation(self, device):
        """Test WanVAEInterface can be instantiated.

        Kept as manual: default checkpoint path is on S3 (needs credentials).
        """
        from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoderConfig

        model = WanVAEEncoderConfig().setup().to(device)

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8


class TestDiTNetwork:
    """Tests for DiT network models."""

    @pytest.mark.ci_gpu
    def test_wan_dit_t2v_1_3b_instantiation_and_checkpoint_loading(self, device):
        """Test WanDiTNetwork 1.3B T2V can be instantiated and load checkpoint."""
        from flashdreams.core.checkpoint.load import load_checkpoint
        from flashdreams.recipes.wan.transformer.impl.network import (
            WanDiTNetwork1pt3BConfig,
        )

        network_config = WanDiTNetwork1pt3BConfig()
        network = network_config.setup().to(device)

        state_dict = load_checkpoint(
            "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors"
        )
        network.load_state_dict(state_dict)

        assert network is not None

    @pytest.mark.manual
    def test_wan_dit_i2v_14b_instantiation_and_checkpoint_loading(self, device):
        """Test WanDiTNetwork 14B I2V can be instantiated and load checkpoint.

        Kept as manual: ~28 GB download, ~28-40 GB peak VRAM during checkpoint
        loading causes OOM on shared CI runners.
        """
        from flashdreams.core.checkpoint.load import load_checkpoint
        from flashdreams.recipes.wan.transformer.impl.network import (
            WanDiTNetwork14BConfig,
        )

        network_config = WanDiTNetwork14BConfig(
            cross_attn_enable_img=True, in_dim=16 + 20
        )
        network = network_config.setup().to(device)

        state_dict = load_checkpoint(
            "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/blob/main/diffusion_pytorch_model.safetensors.index.json"
        )
        network.load_state_dict(state_dict)

        assert network is not None
