# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for LingBot-VA conditional and unconditional KV ownership."""

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from lingbot_va.transformer import (
    LingbotVATransformer,
    LingbotVATransformerCache,
    LingbotVATransformerConfig,
)
from lingbot_va.transformer.impl.network import (
    VideoKV,
    WanVADiTNetworkCache,
    WanVADiTNetworkConfig,
)

pytestmark = pytest.mark.ci_cpu


class _BranchRecordingNetwork(nn.Module):
    """Record the video KV supplied to each action branch."""

    def __init__(
        self,
        cond_cache: WanVADiTNetworkCache,
        uncond_cache: WanVADiTNetworkCache,
    ) -> None:
        super().__init__()
        self._cond_cache = cond_cache
        self._uncond_cache = uncond_cache
        self.action_video_kv: dict[str, VideoKV | None] = {}

    def _branch(self, cache: WanVADiTNetworkCache) -> tuple[str, float]:
        if cache is self._cond_cache:
            return "cond", 1.0
        assert cache is self._uncond_cache
        return "uncond", -1.0

    def forward_video(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanVADiTNetworkCache,
        rope_freqs: Tensor,
        persist: bool = False,
    ) -> tuple[Tensor, VideoKV | None]:
        del timesteps, rope_freqs
        _, value = self._branch(cache)
        video_kv = ((torch.tensor([value]), torch.tensor([value * 10])),)
        return torch.full_like(x, value), video_kv if persist else None

    def forward_action(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanVADiTNetworkCache,
        rope_freqs: Tensor,
        video_kv: VideoKV | None = None,
        persist: bool = False,
    ) -> Tensor:
        del timesteps, rope_freqs
        branch, value = self._branch(cache)
        if persist:
            self.action_video_kv[branch] = video_kv
        return torch.full_like(x, value * 3)


def test_cfg_action_branches_consume_their_matching_video_kv() -> None:
    cond_cache = WanVADiTNetworkCache(block_caches=[])
    uncond_cache = WanVADiTNetworkCache(block_caches=[])
    config = LingbotVATransformerConfig(
        network=WanVADiTNetworkConfig(dim=12, num_heads=1, num_layers=1),
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        compile_network=False,
    )
    transformer = LingbotVATransformer(config)
    network = _BranchRecordingNetwork(cond_cache, uncond_cache)
    object.__setattr__(transformer, "_network", network)
    cache = LingbotVATransformerCache(
        network_cache=cond_cache,
        network_cache_uncond=uncond_cache,
    )
    noisy = torch.zeros(1, 1, 1)
    timestep = torch.zeros(1, 1)
    model_input: dict[str, Any] = {"grid_id": torch.zeros(3, 1)}

    video_flow = transformer.predict_flow(
        noisy,
        timestep,
        cache,
        input=model_input,
        persist=True,
    )
    action_flow = transformer.predict_action_flow(
        noisy,
        timestep,
        cache,
        input=model_input,
        persist=True,
    )

    assert torch.equal(video_flow, torch.full_like(noisy, 9.0))
    assert torch.equal(action_flow, torch.full_like(noisy, 3.0))
    assert network.action_video_kv["cond"] is not None
    assert network.action_video_kv["uncond"] is not None
    assert network.action_video_kv["cond"][0][0].item() == 1.0
    assert network.action_video_kv["uncond"][0][0].item() == -1.0
    assert cache.video_kv_cond is None
    assert cache.video_kv_uncond is None
