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

"""CPU contract tests for native MiniMax H3 model components."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from minimax_h3.constants import align_num_frames, validate_canvas
from minimax_h3.lora import convert_musubi_lora
from minimax_h3.model import MiniMaxH3DenoiseState, MiniMaxH3DiffusionModel
from minimax_h3.scheduler import MiniMaxH3SchedulerConfig
from minimax_h3.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_cpu


def _block_shapes(
    prefix: str, config: MiniMaxH3TransformerConfig
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    inner = config.num_attention_heads * config.attention_head_dim
    return {
        f"{prefix}.norm1.weight": (hidden,),
        f"{prefix}.attn.to_q.weight": (inner, hidden),
        f"{prefix}.attn.to_k.weight": (inner, hidden),
        f"{prefix}.attn.to_v.weight": (inner, hidden),
        f"{prefix}.attn.norm_q.weight": (config.attention_head_dim,),
        f"{prefix}.attn.norm_k.weight": (config.attention_head_dim,),
        f"{prefix}.attn.to_out.0.weight": (hidden, inner),
        f"{prefix}.norm2.weight": (hidden,),
        f"{prefix}.ff.net.0.proj.weight": (2 * config.ffn_dim, hidden),
        f"{prefix}.ff.net.2.weight": (hidden, config.ffn_dim),
    }


def _expected_checkpoint_shapes(
    config: MiniMaxH3TransformerConfig,
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    video_dim = config.in_channels * math.prod(config.patch_size)
    shapes = {
        "proj_in.weight": (hidden, video_dim),
        "proj_in.bias": (hidden,),
        "audio_proj_in.weight": (hidden, config.audio_in_channels),
        "audio_proj_in.bias": (hidden,),
        "context_embedder.weight": (hidden, config.text_dim),
        "context_embedder.bias": (hidden,),
        "time_embedder.linear_1.weight": (
            config.time_embed_hidden_dim,
            config.freq_dim,
        ),
        "time_embedder.linear_1.bias": (config.time_embed_hidden_dim,),
        "time_embedder.linear_2.weight": (
            config.time_embed_dim,
            config.time_embed_hidden_dim,
        ),
        "time_embedder.linear_2.bias": (config.time_embed_dim,),
        "token_refiner.final_norm.weight": (hidden,),
        "norm_out.norm.weight": (hidden,),
        "norm_out.linear.weight": (2 * hidden, config.time_embed_dim),
        "norm_out.linear.bias": (2 * hidden,),
        "proj_out.weight": (video_dim, hidden),
        "proj_out.bias": (video_dim,),
        "audio_proj_out.weight": (config.audio_in_channels, hidden),
        "audio_proj_out.bias": (config.audio_in_channels,),
    }
    for index in range(config.num_refiner_layers):
        shapes.update(
            _block_shapes(f"token_refiner.refiner_blocks.{index}", config)
        )
    for index in range(config.num_layers):
        prefix = f"transformer_blocks.{index}"
        shapes.update(_block_shapes(prefix, config))
        shapes[f"{prefix}.adaln_proj.linear.weight"] = (
            18 * hidden,
            config.time_embed_dim,
        )
        shapes[f"{prefix}.adaln_proj.linear.bias"] = (18 * hidden,)
    return shapes


def test_duration_and_canvas_contracts() -> None:
    """Align both duration boundaries and reject invalid requests."""
    assert align_num_frames(5.0) == 124
    assert align_num_frames(15.0) == 362
    validate_canvas(576, 768)
    with pytest.raises(ValueError, match="between 5 and 15"):
        align_num_frames(15.01)
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_canvas(577, 768)


def test_native_checkpoint_keys_and_shapes_are_bijective() -> None:
    """Match every key and shape in the pinned H3 transformer index."""
    config = MiniMaxH3TransformerConfig(
        checkpoint_path=None,
        device="meta",
        execution_device="cpu",
        attention_backend="math",
    )
    assert (
        config.num_attention_heads,
        config.attention_head_dim,
        config.hidden_size,
        config.num_layers,
        config.num_refiner_layers,
        config.ffn_dim,
        config.in_channels,
        config.audio_in_channels,
        config.patch_size,
        config.text_dim,
        config.freq_dim,
        config.time_embed_hidden_dim,
        config.time_embed_dim,
        config.rope_freq_dim,
    ) == (
        56,
        128,
        5376,
        50,
        2,
        14336,
        24,
        32,
        (1, 2, 2),
        5120,
        256,
        5376,
        2688,
        16,
    )
    actual = {
        key: tuple(value.shape) for key, value in config.setup().state_dict().items()
    }
    expected = _expected_checkpoint_shapes(config)
    assert len(actual) == len(expected) == 638
    assert actual == expected


def test_musubi_lora_conversion_targets_native_layers(tmp_path: Path) -> None:
    """Convert all H3 block targets without introducing a default adapter."""
    from safetensors.torch import load_file, save_file

    source = tmp_path / "adapter.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    for block in range(50):
        for module in ("attn_qkv_proj", "attn_out_proj", "mlp_fc1", "mlp_fc2"):
            prefix = f"lora_unet_blocks_{block}_{module}"
            tensors[f"{prefix}.alpha"] = torch.tensor(2.0)
            tensors[f"{prefix}.lora_down.weight"] = torch.ones(2, 3)
            out_features = 6 if module == "attn_qkv_proj" else 4
            tensors[f"{prefix}.lora_up.weight"] = torch.ones(out_features, 2)
    save_file(tensors, source)

    converted = load_file(
        convert_musubi_lora(source, tmp_path / "converted.safetensors")
    )
    assert len(converted) == 600
    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in converted
    assert "transformer.transformer_blocks.0.attn.to_v.lora_B.weight" in converted
    assert "transformer.transformer_blocks.49.ff.net.2.lora_B.weight" in converted


def test_native_scheduler_matches_pinned_h3_oracle() -> None:
    """Match frozen vectors from Diffusers commit ``175fe6b2419a``."""
    native = MiniMaxH3SchedulerConfig(num_inference_steps=7, shift=12.0).setup()
    sigmas, timesteps = native.schedule("cpu")
    torch.testing.assert_close(
        sigmas,
        torch.tensor(
            [
                1.0,
                0.9836066365242004,
                0.9599999785423279,
                0.9230769276618958,
                0.8571428060531616,
                0.7058823108673096,
                0.0,
            ]
        ),
    )
    torch.testing.assert_close(
        timesteps,
        torch.tensor(
            [
                0.0,
                0.01639336347579956,
                0.04000002145767212,
                0.07692307233810425,
                0.14285719394683838,
                0.29411768913269043,
            ]
        ),
    )

    sample = torch.tensor([[0.25, -0.5, 1.0], [-1.5, 2.0, 0.0]])
    flow = torch.tensor([[0.5, 0.25, -0.75], [1.0, -0.5, 0.125]])
    actual = native.step(sample, flow, timesteps[0], sigmas[0], sigmas[1])
    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [0.2581966817378998, -0.4959016442298889, 0.9877049922943115],
                [-1.4836066961288452, 1.9918032884597778, 0.002049170434474945],
            ]
        ),
    )


def test_scheduler_rejects_invalid_configuration() -> None:
    """Reject schedules that cannot produce a valid Euler grid."""
    with pytest.raises(ValueError, match="at least 2"):
        MiniMaxH3SchedulerConfig(num_inference_steps=1).setup()
    with pytest.raises(ValueError, match="positive"):
        MiniMaxH3SchedulerConfig(shift=0).setup()


def test_row_timestep_plan_preserves_conditioning_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build packed row timesteps without materializing scalar tensors."""
    state = MiniMaxH3DenoiseState(
        latents=torch.empty(0),
        audio_latents=torch.empty(0),
        prompt_embeds=torch.empty(0),
        position_ids=torch.empty(0),
        token_tags=torch.empty(0),
        video_indices=torch.tensor([0, 1, 4]),
        audio_indices=torch.tensor([2, 3]),
        text_indices=torch.tensor([5]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_latent_frames=0,
        latent_height=0,
        latent_width=0,
    )

    def reject_scalar_conversion(_tensor: torch.Tensor) -> float:
        raise AssertionError("row timestep construction converted a tensor to float")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__float__", reject_scalar_conversion)
        timesteps, indices = MiniMaxH3DiffusionModel._row_timesteps(
            state, torch.tensor(0.5), torch.tensor(0.25)
        )

    assert timesteps.device == state.video_indices.device
    torch.testing.assert_close(timesteps, torch.tensor([0.25, 0.5, 0.999, 1.0]))
    torch.testing.assert_close(indices, torch.tensor([2, 1, 3, 0, 1, 1]))
