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

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from omnidreams.impl._drift_corrector import (
    DriftCorrectorDispatch,
    apply_drift_corrector,
)
from omnidreams.impl.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.impl.transformer import modules as transformer_modules
from omnidreams.impl.transformer.modules import AttentionBackend, MultiHeadAttention
from omnidreams.impl.transformer.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_transformer_layerwise_offload_disables_whole_network_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = CosmosTransformerConfig(
        network=_small_network_config(),
        dtype=torch.float32,
        compile_network=True,
        use_cuda_graph=True,
        enable_layerwise_offload=True,
    )

    transformer = CosmosTransformer(config)

    assert isinstance(transformer.network, CosmosDiTNetwork)
    assert transformer.network.layerwise_offloader is not None
    assert transformer._use_cuda_graph is False
    with pytest.raises(ValueError, match="text-edit LoRA"):
        transformer.set_text_edit_lora(object())


def test_layerwise_offload_rejects_drift_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    transformer = CosmosTransformer(
        CosmosTransformerConfig(
            network=_small_network_config(),
            dtype=torch.float32,
            compile_network=False,
            use_cuda_graph=False,
            enable_layerwise_offload=True,
        )
    )
    runner = SimpleNamespace(
        pipeline=SimpleNamespace(
            diffusion_model=SimpleNamespace(transformer=transformer)
        )
    )
    message = "drift correction is not compatible with layer-wise offload"

    for mode in ("premerged", "fused", "unfused"):
        with pytest.raises(ValueError, match=message):
            apply_drift_corrector(
                runner,
                Path("unused-corrector.pt"),
                0.25,
                mode=mode,
            )
    with pytest.raises(ValueError, match=message):
        DriftCorrectorDispatch(runner)


@torch.inference_mode()
def test_layerwise_offload_covers_cache_initialization_and_text_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    network = CosmosDiTNetwork(_small_network_config()).to(dtype=torch.float32)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    network.enable_layerwise_offload()
    context = torch.randn(1, 1, 3, 16)

    cache = network.initialize_cache(
        chunk_size=2,
        window_size=4,
        sink_size=0,
        context=context,
    )
    network.replace_text_embeddings(cache, torch.randn_like(context))

    assert len(cache.block_caches) == 2
    assert network.layerwise_offloader is not None
    assert network.layerwise_offloader.resident_layer_indices == (0,)
    assert cache[0].cross_attn.cached_k().shape == (1, 3, 4, 8)


def test_layerwise_offload_matches_multiview_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match a resident network across the multi-view block path on CPU."""
    torch.manual_seed(0)
    config = _small_network_config(enable_cross_view_attn=True)
    resident = CosmosDiTNetwork(config).to(dtype=torch.float32).eval()
    resident.update_parameters_after_loading_checkpoint()
    offloaded = copy.deepcopy(resident)
    offloaded.enable_layerwise_offload()

    monkeypatch.setattr(
        transformer_modules, "apply_rope_freqs", lambda tensor, _freqs: tensor
    )

    def cpu_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    for network in (resident, offloaded):
        for block in network.blocks:
            for attention in (
                block.self_attn,
                block.cross_attn,
                block.cross_view_attn,
            ):
                assert isinstance(attention, MultiHeadAttention)
                monkeypatch.setattr(attention.attn_op, "_impl", cpu_sdpa)

    context = torch.randn(1, 2, 3, 16)
    value = torch.randn(1, 2, 1, 2, 2)
    condition_mask = torch.zeros(1, 2, 1, 2, 1)
    forward_kwargs: dict[str, Any] = {
        "x": value,
        "timesteps": torch.tensor(1000.0),
        "rope_freqs": torch.zeros(2, 1, 1, 8),
        "condition_video_input_mask": condition_mask,
        "current_chunk_idx": 0,
        "view_indices": torch.tensor([[0, 1]]),
        "eager_mode": False,
    }

    with torch.no_grad():
        expected_cache = resident.initialize_cache(2, 4, 0, context)
        actual_cache = offloaded.initialize_cache(2, 4, 0, context)
        expected_cache.before_update(0)
        actual_cache.before_update(0)
        expected = resident(cache=expected_cache, **forward_kwargs)
        actual = offloaded(cache=actual_cache, **forward_kwargs)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 2, 1, 2, 2)
    for expected_block, actual_block in zip(
        expected_cache.block_caches,
        actual_cache.block_caches,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_block.self_attn.cached_k(), expected_block.self_attn.cached_k()
        )
        torch.testing.assert_close(
            actual_block.self_attn.cached_v(), expected_block.self_attn.cached_v()
        )


@pytest.mark.parametrize(
    (
        "native_acceleration",
        "self_attention_backend",
        "cross_attention_backend",
        "message",
    ),
    [
        (
            "required",
            AttentionBackend.OMNIDREAMS,
            AttentionBackend.OMNIDREAMS,
            "native DiT acceleration",
        ),
        (
            "auto",
            AttentionBackend.OMNIDREAMS,
            AttentionBackend.OMNIDREAMS,
            "native DiT acceleration",
        ),
        (
            "disabled",
            AttentionBackend.OPTIMIZED,
            AttentionBackend.OMNIDREAMS,
            "OmniDreams self- and cross-attention backends",
        ),
        (
            "disabled",
            AttentionBackend.OMNIDREAMS,
            AttentionBackend.OPTIMIZED,
            "OmniDreams self- and cross-attention backends",
        ),
    ],
)
def test_layerwise_offload_rejects_incompatible_dit_paths(
    native_acceleration: Literal["auto", "disabled", "required"],
    self_attention_backend: AttentionBackend,
    cross_attention_backend: AttentionBackend,
    message: str,
) -> None:
    config = CosmosTransformerConfig(
        network=_small_network_config(
            self_attention_backend=self_attention_backend,
            cross_attention_backend=cross_attention_backend,
        ),
        dtype=torch.float32,
        compile_network=False,
        use_cuda_graph=False,
        enable_layerwise_offload=True,
        native_dit_acceleration=native_acceleration,
    )

    with pytest.raises(ValueError, match=message):
        CosmosTransformer(config)


def _small_network_config(**overrides: Any) -> CosmosDiTNetworkConfig:
    values: dict[str, Any] = {
        "in_channels": 2,
        "out_channels": 2,
        "patch_spatial": 1,
        "model_channels": 32,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "adaln_lora_dim": 8,
        "crossattn_proj_in_channels": 16,
        "crossattn_emb_channels": 16,
    }
    values.update(overrides)
    return CosmosDiTNetworkConfig(**values)
