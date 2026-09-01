# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU parity checks for native Qwen Image Edit components."""

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)
from qwen_image_edit_v2.editor import pack_latents, unpack_latents
from qwen_image_edit_v2.transformer import QwenImageTransformer
from qwen_image_edit_v2.vae import QwenImageVAE

pytestmark = pytest.mark.ci_cpu


def test_transformer_state_dict_and_forward_match_diffusers() -> None:
    reference = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        guidance_embeds=False,
        zero_cond_t=True,
        use_layer3d_rope=True,
    ).eval()
    native = QwenImageTransformer(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    assert native.state_dict().keys() == reference.state_dict().keys()
    native.load_state_dict(reference.state_dict())
    torch.manual_seed(3)
    hidden = torch.randn(1, 8, 64)
    text = torch.randn(1, 5, 12)
    timestep = torch.tensor([0.4])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    shapes = [(1, 2, 2), (1, 2, 2)]

    actual = native(hidden, text, timestep, shapes, mask)
    expected = reference(
        hidden_states=hidden,
        encoder_hidden_states=text,
        timestep=timestep,
        img_shapes=[shapes],
        encoder_hidden_states_mask=mask,
        return_dict=False,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_image_vae_state_dict_and_forward_match_diffusers() -> None:
    reference = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
    ).eval()
    native = QwenImageVAE(
        base_dim=4,
        z_dim=16,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temporal_downsample=(False,),
    ).eval()
    assert native.state_dict().keys() == reference.state_dict().keys()
    native.load_state_dict(reference.state_dict())
    torch.manual_seed(4)
    image = torch.randn(1, 3, 16, 16)

    actual_moments = native.quant_conv(native.encoder(image.unsqueeze(2)))
    expected_moments = reference.quant_conv(
        reference.encoder(image.unsqueeze(2), feat_cache=None, feat_idx=[0])
    )
    actual = native.decoder(native.post_quant_conv(actual_moments.chunk(2, 1)[0]))
    expected = reference.decoder(
        reference.post_quant_conv(expected_moments.chunk(2, 1)[0]),
        feat_cache=None,
        feat_idx=[0],
    )

    torch.testing.assert_close(actual_moments, expected_moments, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_latent_pack_round_trip() -> None:
    latents = torch.arange(16 * 8 * 12).reshape(1, 16, 8, 12)
    torch.testing.assert_close(unpack_latents(pack_latents(latents), 64, 96), latents)


def test_bfloat16_cast_preserves_complex_rope() -> None:
    model = QwenImageTransformer(
        num_layers=0,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
    ).to(torch.bfloat16)

    assert model.pos_embed.pos_freqs.dtype == torch.complex64
    assert model.pos_embed.neg_freqs.dtype == torch.complex64
