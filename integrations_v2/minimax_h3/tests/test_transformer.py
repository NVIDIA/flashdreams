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

"""CPU contracts for the native H3 joint transformer and acceleration policy."""

import pytest
import torch
from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    OptimizedMultiHeadAttention,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from minimax_h3.impl.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_cpu


def _config(**kwargs):
    return MiniMaxH3TransformerConfig(
        checkpoint_path=None,
        dtype=torch.float32,
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=10,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=2,
        **kwargs,
    )


def test_joint_transformer_preserves_both_modalities() -> None:
    model = _config(attention_backend="torch").setup()
    inputs = {
        "hidden_states": torch.randn(1, 4, 2),
        "audio_hidden_states": torch.randn(1, 2, 4),
        "encoder_hidden_states": torch.randn(1, 3, 10),
        "timestep": torch.tensor([0.1, 0.5, 0.999]),
        "timestep_indices": torch.tensor([1, 1, 1, 2, 0, 1, 1, 0, 2]),
        "token_tags": torch.tensor([1, 1, 1, 0, 0, 2, 2, 0, 0]),
        "position_ids": torch.randn(9, 3),
        "video_indices": torch.tensor([3, 4, 7, 8]),
        "audio_indices": torch.tensor([5, 6]),
        "text_indices": torch.tensor([0, 1, 2]),
    }
    with torch.no_grad():
        video, audio = model(**inputs)
        again = model(**inputs)
    assert video.shape == (1, 4, 2) and audio.shape == (1, 2, 4)
    assert video.isfinite().all() and audio.isfinite().all()
    torch.testing.assert_close(video, again[0])
    torch.testing.assert_close(audio, again[1])
    assert "transformer_blocks.0.attn.to_q.weight" in model.state_dict()


def test_optimized_default_and_lora_refresh() -> None:
    config = _config(
        optimized_impl=OptimizedImplConfig(
            qkv_fusion_option=QKVFusionOption.FULL,
            sdpa_backend=SDPABackend.TORCH,
            use_tma=False,
        )
    )
    model = config.setup()
    attention = model.transformer_blocks[0].attn
    assert isinstance(attention, OptimizedMultiHeadAttention)
    before = attention.fused_qkv.weight.clone()
    with torch.no_grad():
        attention.to_q.weight.add_(1)
    model.refresh_derived_weights()
    assert not torch.equal(before, attention.fused_qkv.weight)
    torch.testing.assert_close(attention.fused_qkv.weight[:32], attention.to_q.weight)
    default = MiniMaxH3TransformerConfig().optimized_impl
    assert default.qkv_fusion_option is QKVFusionOption.NONE
    assert default.sdpa_backend is SDPABackend.TORCH
    assert not default.use_tma and default.quantization == QuantizationOption()


def test_ampere_native_policy_and_fp8_guard(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    native = _config().setup().transformer_blocks[0].attn
    native._validate_cuda_device("cuda:0")
    fp8 = (
        _config(
            optimized_impl=OptimizedImplConfig(
                quantization=QuantizationOption(quantized_sdpa=True),
            )
        )
        .setup()
        .transformer_blocks[0]
        .attn
    )
    with pytest.raises(RuntimeError, match="9.0 for FP8"):
        fp8._validate_cuda_device("cuda:0")
