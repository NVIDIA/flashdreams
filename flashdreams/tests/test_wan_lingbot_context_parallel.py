# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from dataclasses import dataclass
from typing import cast

import pytest
import torch

from flashdreams.recipes.lingbot_world.encoder.camctrl import I2VCamCtrlEmbeddings
from flashdreams.recipes.lingbot_world.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)
from flashdreams.recipes.lingbot_world.transformer.impl.network import (
    LingbotWorldDiTNetworkConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)


class _FakeProcessGroup:
    def __init__(self, world_size: int, rank: int) -> None:
        self._world_size = world_size
        self._rank = rank

    def size(self) -> int:
        return self._world_size

    def rank(self) -> int:
        return self._rank


class _DummyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cp_group = None
        self.parameters_updated = False

    def set_context_parallel_group(self, cp_group=None) -> None:
        self.cp_group = cp_group

    def update_parameters_after_loading_checkpoint(self) -> None:
        self.parameters_updated = True


@dataclass
class _DummyNetworkConfig:
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 16

    def setup(self) -> _DummyNetwork:
        return _DummyNetwork()


def _mock_distributed(
    monkeypatch, world_size: int = 2, rank: int = 0
) -> _FakeProcessGroup:
    fake_group = _FakeProcessGroup(world_size=world_size, rank=rank)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed,
        "group",
        types.SimpleNamespace(WORLD=fake_group),
        raising=False,
    )
    return fake_group


def test_wan21_uses_world_cp_group_when_distributed(monkeypatch) -> None:
    fake_group = _mock_distributed(monkeypatch, world_size=2, rank=0)
    transformer = Wan21Transformer(
        Wan21TransformerConfig(
            network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
            batch_shape=(1,),
            len_t=2,
            height=4,
            width=4,
            cp_size=2,
            window_size_t=2,
            sink_size_t=0,
        )
    )

    assert transformer.cp_group is fake_group
    assert transformer.cp_size == 2
    assert isinstance(transformer.network, _DummyNetwork)
    assert transformer.network.cp_group is fake_group
    assert transformer.network.parameters_updated


def test_wan21_requires_cp_size_one_without_distributed(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(
        AssertionError, match="cp_size must be 1 in non-distributed mode"
    ):
        Wan21Transformer(
            Wan21TransformerConfig(
                network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
                batch_shape=(1,),
                len_t=2,
                height=4,
                width=4,
                cp_size=2,
            )
        )


def test_wan21_requires_tokens_divisible_by_cp_size(monkeypatch) -> None:
    _mock_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(AssertionError, match="must be divisible by cp_size=2"):
        Wan21Transformer(
            Wan21TransformerConfig(
                network=cast(WanDiTNetworkConfig, _DummyNetworkConfig()),
                batch_shape=(1,),
                len_t=1,
                height=2,
                width=2,
                cp_size=2,
                window_size_t=1,
                sink_size_t=0,
            )
        )


def test_wan_patchify_unpatchify_round_trip_without_cp() -> None:
    network = WanDiTNetworkConfig(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=1,
        patch_embedding_type="linear",
    ).setup()
    latent = torch.randn(1, 2, 16, 4, 4)

    patched = network.patchify_and_maybe_split_cp(
        latent,
        process_groups=[None],
        cp_dims=[-2],
    )
    restored = network.unpatchify_and_maybe_gather_cp(
        pH=2,
        pW=2,
        x=patched,
        process_groups=[None],
        cp_dims=[-2],
    )

    assert patched.shape == (1, 8, 64)
    torch.testing.assert_close(restored, latent)


def test_lingbot_patchify_marks_i2v_and_plucker_as_patchified() -> None:
    transformer = LingbotWorldTransformer(
        LingbotWorldTransformerConfig(
            network=LingbotWorldDiTNetworkConfig(
                dim=64,
                ffn_dim=128,
                num_heads=4,
                num_layers=1,
                patch_embedding_type="linear",
                control_type="cam",
            ),
            batch_shape=(1, 1),
            len_t=2,
            height=4,
            width=4,
            cp_size=1,
            window_size_t=2,
            sink_size_t=0,
            compile_network=False,
        )
    )

    camctrl_embeddings = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.randn(1, 1, 2, 16, 4, 4),
            mask=torch.randn(1, 1, 2, 16, 4, 4),
        ),
        plucker=torch.randn(1, 1, 2, 6 * 64, 4, 4),
    )

    patched = transformer.patchify_and_maybe_split_cp(camctrl_embeddings)
    assert isinstance(patched, I2VCamCtrlEmbeddings)
    assert patched._is_patchified
    assert patched.i2v._is_patchified
    assert patched.i2v.latent.shape == (1, 1, 8, 64)
    assert patched.i2v.mask.shape == (1, 1, 8, 64)
    assert patched.plucker.shape == (1, 1, 8, 1536)

    # Idempotent once marked patchified.
    assert transformer.patchify_and_maybe_split_cp(patched) is patched
