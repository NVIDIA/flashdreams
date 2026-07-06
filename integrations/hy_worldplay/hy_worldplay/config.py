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

"""Static pipeline + runner configs for the HY-WorldPlay WAN-5B I2V integration."""

from __future__ import annotations

import copy

from wan22.config import PIPELINE_WAN22_TI2V_5B

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchEulerDiscreteSchedulerConfig,
)
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from hy_worldplay._action import (
    HyWorldPlayWan21TransformerConfig,
    HyWorldPlayWanCtrlEncoderConfig,
    HyWorldPlayWanDiTNetworkConfig,
)
from hy_worldplay._checkpoint import (
    HY_WORLDPLAY_DISTILLED_CKPT_PATH,
    hy_worldplay_distilled_state_dict_transform,
)
from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

__all__ = [
    "PIPELINE_HY_WORLDPLAY_WAN_I2V_5B",
    "RUNNER_CONFIGS",
    "RUNNER_HY_WORLDPLAY_WAN_I2V_5B",
]


def _build_hy_worldplay_pipeline() -> WanInferencePipelineConfig:
    """Deep-copy the Wan 2.2 TI2V-5B recipe and layer the full HY-WorldPlay stack on top.

    Swaps in the distilled 4-step Euler scheduler, the action / camera
    HY encoder, and the HY transformer + DiT network (PRoPE blocks
    enabled). The transformer is copied field-by-field rather than
    derived: :func:`derive_config` can't change the dataclass type, and
    an explicit copy fails loudly if a base field is added.
    """
    pipeline = copy.deepcopy(PIPELINE_WAN22_TI2V_5B)
    pipeline.name = "hy-worldplay-wan-i2v-5b"

    # Distilled WAN-5B fixed-timestep schedule (base recipe stays on UniPC).
    pipeline.diffusion_model.scheduler = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )

    assert isinstance(pipeline.encoder, WanI2VCtrlEncoderConfig)
    pipeline.encoder = HyWorldPlayWanCtrlEncoderConfig(
        encoder=pipeline.encoder.encoder,
    )

    base_t = pipeline.diffusion_model.transformer
    # Narrow to the concrete config so the subclass-only attributes
    # copied below resolve.
    assert isinstance(base_t, Wan21TransformerConfig)
    base_n = base_t.network
    assert isinstance(base_n, WanDiTNetworkConfig)
    pipeline.diffusion_model.transformer = HyWorldPlayWan21TransformerConfig(
        network=HyWorldPlayWanDiTNetworkConfig(
            patch_size=base_n.patch_size,
            text_len=base_n.text_len,
            in_dim=base_n.in_dim,
            dim=base_n.dim,
            ffn_dim=base_n.ffn_dim,
            freq_dim=base_n.freq_dim,
            text_dim=base_n.text_dim,
            out_dim=base_n.out_dim,
            num_heads=base_n.num_heads,
            num_layers=base_n.num_layers,
            cross_attn_norm=base_n.cross_attn_norm,
            cross_attn_enable_img=base_n.cross_attn_enable_img,
            eps=base_n.eps,
            concat_padding_mask=base_n.concat_padding_mask,
            patch_embedding_type=base_n.patch_embedding_type,
            apply_rope_before_kvcache=base_n.apply_rope_before_kvcache,
            use_prope_blocks=True,
        ),
        dtype=base_t.dtype,
        # Inference loads HY-WorldPlay's distilled WAN-5B weights by default;
        # ``--ckpt-path`` overrides with a local ``model.pt``.
        checkpoint_path=HY_WORLDPLAY_DISTILLED_CKPT_PATH,
        state_dict_transform=hy_worldplay_distilled_state_dict_transform,
        batch_shape=base_t.batch_shape,
        # 4-latent AR chunks (not the base recipe's 21); sets total
        # frame counts and RoPE positions.
        len_t=4,
        # CFG is baked into the distilled checkpoint; ``1.0`` skips the
        # uncond branch.
        guidance_scale=1.0,
        # Match the rolling KV window to a single chunk.
        window_size_t=4,
        sink_size_t=base_t.sink_size_t,
        h_extrapolation_ratio=base_t.h_extrapolation_ratio,
        w_extrapolation_ratio=base_t.w_extrapolation_ratio,
        compile_network=base_t.compile_network,
        # CUDA-graph capture is unsafe on the HY-WorldPlay memory-prefill
        # path. The ``CUDAGraphWrapper`` captures pointers into the KV
        # cache, but HY re-runs ``prefill_memory_kv_cache`` every chunk: it
        # resets+repopulates each PRoPE block's memory KV from a *different*
        # FOV-selected frame set (``select_mem_frames_wan``), reallocating
        # the underlying storage. A graph captured on one chunk then replays
        # against another chunk's stale/freed memory-KV slots, decoding to a
        # "shatter" of speckle corruption (deterministic across seeds and
        # prompts; only the captured/replayed chunks are hit, so it presents
        # as one or two garbled chunks mid-rollout). Disabling capture keeps
        # ``compile_network`` (Inductor) -- still ~4x faster diffuse than
        # vendor -- without the unsafe replay. See ``HY_DEBUG_DISABLE_CUDA_GRAPH``.
        use_cuda_graph=False,
        cuda_graph_warmup_iters=base_t.cuda_graph_warmup_iters,
        stamp_image_latent=base_t.stamp_image_latent,
        concat_image_mask_to_latent=base_t.concat_image_mask_to_latent,
        ti2v_first_frame_per_token_timestep=(
            base_t.ti2v_first_frame_per_token_timestep
        ),
        # First-frame context runs at the stabilisation sigma 14, which
        # the distilled checkpoint's AdaLN table is fitted to.
        first_frame_timestep_value=14.0,
    )
    return pipeline


PIPELINE_HY_WORLDPLAY_WAN_I2V_5B = _build_hy_worldplay_pipeline()
"""Wan 2.2 TI2V-5B + HY-WorldPlay distilled stack: HY encoder /
transformer / network with PRoPE blocks and the 4-step Euler schedule.
Production target for the ``hy-worldplay-wan-i2v-5b`` runner."""


RUNNER_HY_WORLDPLAY_WAN_I2V_5B = HyWorldPlayWanI2VRunnerConfig(
    runner_name=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name,
    description=(
        "HY-WorldPlay WAN-5B I2V (Wan 2.2 TI2V backbone, action + camera "
        "trajectory conditioning, reconstituted-context memory)."
    ),
    pipeline=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg for cfg in (RUNNER_HY_WORLDPLAY_WAN_I2V_5B,)
}
"""Shipped HY-WorldPlay runner configs keyed by ``runner_name``."""
