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

"""CPU regressions for LingBot-VA conditional and unconditional KV ownership."""

from typing import Any

import pytest
import torch
from lingbot_va.transformer import (
    LingbotVATransformer,
    LingbotVATransformerCache,
    LingbotVATransformerConfig,
)
from lingbot_va.transformer.impl.kvcache import VAKVCache
from lingbot_va.transformer.impl.modules import VABlockCache
from lingbot_va.transformer.impl.network import (
    VideoKV,
    WanVADiTNetwork,
    WanVADiTNetworkCache,
    WanVADiTNetworkConfig,
)
from torch import Tensor, nn

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.recipes.wan.transformer.impl.modules import CrossAttnCache

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
        self.video_branches: list[str] = []
        self.action_branches: list[str] = []

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
        branch, value = self._branch(cache)
        self.video_branches.append(branch)
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
        self.action_branches.append(branch)
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


def test_inactive_action_cfg_branch_only_runs_when_committing_cache() -> None:
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

    transformer.predict_flow(noisy, timestep, cache, input=model_input)
    transformer.predict_action_flow(noisy, timestep, cache, input=model_input)

    assert network.video_branches == ["cond", "uncond"]
    assert network.action_branches == ["cond"]

    transformer.predict_flow(noisy, timestep, cache, input=model_input, persist=True)
    transformer.predict_action_flow(
        noisy,
        timestep,
        cache,
        input=model_input,
        persist=True,
    )

    assert network.video_branches == ["cond", "uncond", "cond", "uncond"]
    assert network.action_branches == ["cond", "cond", "uncond"]


def test_action_block_loop_attends_to_committed_then_current_video_kv() -> None:
    config = WanVADiTNetworkConfig(
        dim=12,
        ffn_dim=24,
        num_heads=1,
        num_layers=1,
        text_dim=8,
        freq_dim=4,
    )
    network = WanVADiTNetwork(config)
    network._parameters_updated_after_loading_checkpoint = True
    prior_k = torch.tensor([[[[1.0] * 12]]])
    prior_v = torch.tensor([[[[2.0] * 12]]])
    current_video_k = torch.tensor([[[[3.0] * 12]]])
    current_video_v = torch.tensor([[[[4.0] * 12]]])
    text_k = torch.zeros(1, 1, 1, 12)
    text_v = torch.zeros_like(text_k)
    self_attn = VAKVCache.create(
        video_chunk=1,
        action_chunk=0,
        window_slots=2,
        batch_size=1,
        num_heads=1,
        head_dim=12,
        device="cpu",
        dtype=torch.float32,
    )
    self_attn.before_update(0)
    self_attn.write_video(prior_k, prior_v)
    self_attn.after_update(0)
    self_attn.before_update(1)
    block_cache = VABlockCache(
        self_attn=self_attn,
        cross_attn=CrossAttnCache(
            text=BlockKVCache.from_tensor(text_k, text_v, seq_dim=1)
        ),
    )
    cache = WanVADiTNetworkCache(block_caches=[block_cache])
    captured: dict[str, Tensor] = {}

    def record_action_inputs(
        x: Tensor,
        timesteps: Tensor,
        committed_k: Tensor,
        committed_v: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        del timesteps, cross_k, cross_v, rope_freqs
        captured["k"] = committed_k
        captured["v"] = committed_v
        fresh = torch.zeros(1, 1, 1, 12)
        return x, [fresh], [fresh]

    object.__setattr__(network, "_forward_blocks_action", record_action_inputs)

    output = network.forward_action(
        torch.zeros(1, 1, 12),
        torch.zeros(1, 1),
        cache,
        torch.zeros(1, 1, 1, 12),
        video_kv=((current_video_k, current_video_v),),
    )

    assert output.shape == (1, 1, 12)
    assert captured["k"].shape == (1, 1, 2, 1, 12)
    assert captured["v"].shape == (1, 1, 2, 1, 12)
    torch.testing.assert_close(captured["k"][0, 0, 0], prior_k[0, 0])
    torch.testing.assert_close(captured["k"][0, 0, 1], current_video_k[0, 0])
    torch.testing.assert_close(captured["v"][0, 0, 0], prior_v[0, 0])
    torch.testing.assert_close(captured["v"][0, 0, 1], current_video_v[0, 0])


def test_action_block_loop_requires_current_video_kv() -> None:
    network = WanVADiTNetwork(
        WanVADiTNetworkConfig(
            dim=12,
            ffn_dim=24,
            num_heads=1,
            num_layers=1,
            text_dim=8,
            freq_dim=4,
        )
    )
    network._parameters_updated_after_loading_checkpoint = True

    with pytest.raises(ValueError, match="current-chunk video KV"):
        network.forward_action(
            torch.zeros(1, 1, 12),
            torch.zeros(1, 1),
            WanVADiTNetworkCache(block_caches=[]),
            torch.zeros(1, 1, 1, 12),
        )


def test_rolling_window_excludes_stale_trailing_chunk() -> None:
    cache = VAKVCache.create(
        video_chunk=2,
        action_chunk=1,
        window_slots=3,
        batch_size=1,
        num_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )

    for chunk_idx in range(3):
        value = float(chunk_idx + 1)
        video_k = torch.full((1, 2, 1, 1), value)
        video_v = torch.full((1, 2, 1, 1), value + 10)
        action_k = torch.full((1, 1, 1, 1), value)
        action_v = torch.full((1, 1, 1, 1), value + 10)
        cache.before_update(chunk_idx)
        cache.write_video(video_k, video_v)
        cache.write_action(
            action_k,
            action_v,
            video_k,
            video_v,
        )
        cache.after_update(chunk_idx)

    cache.before_update(3)
    committed_k, committed_v = cache.committed_kv()

    expected_k = torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).view(1, 6, 1, 1)
    expected_v = expected_k + 10
    torch.testing.assert_close(committed_k, expected_k)
    torch.testing.assert_close(committed_v, expected_v)
    assert cache.n_committed_tokens == 6
    assert committed_k.shape == (1, 6, 1, 1)
