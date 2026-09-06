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

"""CPU-only unit tests for the pre-merged text-edit LoRA deploy hook.

Covers the deploy invariants: ``TextEditLoRA`` merges ``W + B @ A``
correctly, toggles by in-place ``copy_`` (stable storage addresses),
restores the base bit-exactly, is idempotent, and ``release_targets``
hands projections over to the fused drift-corrector dispatch.

The transformer-level window tests (``replace_text_embeddings`` building
a ``use_lora`` window, ``predict_flow`` expiry) live on the branch that
carries the text-edit transformer machinery.
"""

from __future__ import annotations

import pytest
import torch
from omnidreams._edit_lora import TextEditLoRA, _target_linears
from omnidreams.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.transformer.impl.network import CosmosDiTNetworkConfig

pytestmark = pytest.mark.ci_cpu


def _tiny_transformer(seed: int = 0) -> CosmosTransformer:
    torch.manual_seed(seed)
    config = CosmosTransformerConfig(
        network=CosmosDiTNetworkConfig(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            adaln_lora_dim=8,
            crossattn_proj_in_channels=32,
            crossattn_emb_channels=16,
            additional_concat_ch=0,
            enable_cross_view_attn=False,
        ),
        checkpoint_path=None,
        batch_shape=(1,),
        num_views=1,
        len_t=2,
        window_size_t=6,
        sink_size_t=0,
        compile_network=False,
        use_cuda_graph=False,
        guidance_scale=1.0,
    )
    return CosmosTransformer(config)


def _fake_checkpoint(network, rank: int = 4, path=None):
    torch.manual_seed(3)
    linears = _target_linears(network)
    sd = {}
    for i, lin in enumerate(linears):
        sd[2 * i] = torch.randn(rank, lin.in_features) * 0.02  # A
        sd[2 * i + 1] = torch.randn(lin.out_features, rank) * 0.02  # B
    torch.save({"lora": sd}, path)
    return linears, sd


def test_merge_toggle_and_bit_exact_restore(tmp_path):
    transformer = _tiny_transformer()
    ckpt = tmp_path / "edit_lora.pt"
    linears, sd = _fake_checkpoint(transformer.network, path=ckpt)
    base = [lin.weight.detach().clone() for lin in linears]
    ptrs = [lin.weight.data_ptr() for lin in linears]

    edit_lora = TextEditLoRA(transformer.network, ckpt)
    assert edit_lora.rank == 4
    assert len(linears) == 2 * 8  # 2 tiny blocks x 8 projections

    edit_lora.set_active(True)
    for i, lin in enumerate(linears):
        expected = (
            base[i].to(torch.float32) + sd[2 * i + 1].float() @ sd[2 * i].float()
        ).to(base[i].dtype)
        assert torch.equal(lin.weight, expected)
        assert lin.weight.data_ptr() == ptrs[i]  # in place: CUDA-graph safe
    edit_lora.set_active(True)  # idempotent

    edit_lora.set_active(False)
    for i, lin in enumerate(linears):
        assert torch.equal(lin.weight, base[i])
        assert lin.weight.data_ptr() == ptrs[i]


def test_checkpoint_shape_mismatch_rejected(tmp_path):
    transformer = _tiny_transformer()
    ckpt = tmp_path / "bad.pt"
    torch.save({"lora": {0: torch.zeros(4, 8), 1: torch.zeros(8, 4)}}, ckpt)
    with pytest.raises(AssertionError, match="target-list mismatch"):
        TextEditLoRA(transformer.network, ckpt)


def test_release_targets_returns_deltas_and_stops_toggling_them(tmp_path):
    transformer = _tiny_transformer()
    ckpt = tmp_path / "edit_lora.pt"
    linears, sd = _fake_checkpoint(transformer.network, path=ckpt)
    base = [lin.weight.detach().clone() for lin in linears]
    edit_lora = TextEditLoRA(transformer.network, ckpt)

    released = [lin for i, lin in enumerate(linears) if i % 2 == 0]
    kept = [(i, lin) for i, lin in enumerate(linears) if i % 2 == 1]
    deltas = edit_lora.release_targets(released)

    # Deltas are the merged edit minus base (== B @ A here, up to the base
    # dtype's rounding — the tiny network is bf16), in released order.
    for delta, lin in zip(deltas, released):
        i = linears.index(lin)
        expected = sd[2 * i + 1].float() @ sd[2 * i].float()
        assert torch.allclose(delta, expected, atol=1e-3)
        assert delta.abs().max() > 0

    # Toggling now touches only the kept projections.
    edit_lora.set_active(True)
    for lin in released:
        assert torch.equal(lin.weight, base[linears.index(lin)])
    for i, lin in kept:
        assert not torch.equal(lin.weight, base[i])
    edit_lora.set_active(False)
    for i, lin in enumerate(linears):
        assert torch.equal(lin.weight, base[i])

    # Releasing a foreign linear or releasing while active is rejected.
    import torch.nn as nn

    with pytest.raises(AssertionError, match="not one of"):
        edit_lora.release_targets([nn.Linear(4, 4)])
    edit_lora.set_active(True)
    with pytest.raises(AssertionError, match="base weights"):
        edit_lora.release_targets([kept[0][1]])


def test_fp32_network_restores_base_bit_exactly(tmp_path):
    """Regression: ``base.to(float32)`` on an fp32 network aliased ``base``,
    so the in-place merge corrupted the cached base set and deactivating
    the edit could not restore the original weights."""
    import torch.nn as nn

    torch.manual_seed(0)
    network = nn.ModuleDict(
        {
            "self_attn": nn.ModuleDict(
                {n: nn.Linear(8, 8, bias=False) for n in ("q_proj", "k_proj")}
            )
        }
    )
    assert all(lin.weight.dtype == torch.float32 for lin in _target_linears(network))
    ckpt = tmp_path / "edit_lora.pt"
    linears, _ = _fake_checkpoint(network, path=ckpt)
    base = [lin.weight.detach().clone() for lin in linears]

    edit_lora = TextEditLoRA(network, ckpt)
    edit_lora.set_active(True)
    for lin, w in zip(linears, base):
        assert not torch.equal(lin.weight, w)  # the merge actually applied
    edit_lora.set_active(False)
    for lin, w in zip(linears, base):
        assert torch.equal(lin.weight, w)
