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

from typing import Literal

import mediapy
import pytest
import torch

from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoder,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoder,
    WanVAEDecoderConfig,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)


@torch.no_grad()
@pytest.mark.manual
@pytest.mark.parametrize("tokenizer_choice", ["lightvae", "vae"])
@pytest.mark.parametrize("detokenizer_choice", ["lighttae", "lightvae", "vae"])
def test_tokenizer(
    tokenizer_choice: Literal["lightvae", "vae"],
    detokenizer_choice: Literal["lighttae", "lightvae", "vae"],
) -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda")

    tokenizer: WanVAEEncoder
    if tokenizer_choice == "lightvae":
        tokenizer = (
            WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
                dtype=dtype,
                use_cuda_graph=False,
            )
            .setup()
            .to(device)
        )
    elif tokenizer_choice == "vae":
        tokenizer = (
            WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
                dtype=dtype,
                use_cuda_graph=False,
            )
            .setup()
            .to(device)
        )
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer_choice}")

    detokenizer: WanVAEDecoder | TeahvVAEDecoder
    if detokenizer_choice == "lighttae":
        detokenizer = (
            TeahvVAEDecoderConfig(
                checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
                dtype=dtype,
                use_cuda_graph=False,
                use_compile=False,
            )
            .setup()
            .to(device)
        )
    elif detokenizer_choice == "lightvae":
        detokenizer = (
            WanVAEDecoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
                dtype=dtype,
                use_cuda_graph=False,
            )
            .setup()
            .to(device)
        )
    elif detokenizer_choice == "vae":
        detokenizer = (
            WanVAEDecoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
                dtype=dtype,
                use_cuda_graph=False,
            )
            .setup()
            .to(device)
        )
    else:
        raise ValueError(f"Invalid detokenizer: {detokenizer_choice}")

    tokenizer_cache = tokenizer.initialize_autoregressive_cache()
    detokenizer_cache = detokenizer.initialize_autoregressive_cache()

    video_path = "./assets/example_data/omnidreams/camera_front_wide_120fov.mp4"
    video = mediapy.read_video(video_path)[:81]  # [T, H, W, 3]
    video = (
        torch.from_numpy(video).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]

    video = video.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
    encoded_video = tokenizer(video, cache=tokenizer_cache)
    decoded_video = detokenizer(encoded_video, cache=detokenizer_cache)

    l1_loss = torch.nn.functional.l1_loss(video, decoded_video)
    print(
        f"tokenizer: {tokenizer_choice}, detokenizer: {detokenizer_choice}, L1 loss: {l1_loss.item()}"
    )


@pytest.mark.ci_cpu
def test_wan22_vae_pth_remap_is_full_bijection() -> None:
    """The native ``.pth`` remap covers every ``WanVAE`` param with no leftovers.

    ``WanVAE`` builds its modules on ``meta`` then
    ``load_state_dict(strict=False, assign=True)``; any model key the
    checkpoint's transform does not supply stays on ``meta`` and the
    later ``.to(device)`` raises "Cannot copy out of meta tensor". This
    test reconstructs the Wan 2.2 TI2V-5B ``WanVAE`` module tree on
    ``meta`` (no checkpoint download) and proves
    :func:`wan22_ti2v_5b_vae_pth_state_dict_transform` maps upstream's
    native layout onto exactly that key set -- a 1:1 bijection with
    matching shapes, so a real load leaves nothing on meta.
    """
    import re

    from flashdreams.recipes.wan.autoencoder.vae import (
        CausalConv3d,
        Decoder3d,
        Encoder3d,
        wan22_ti2v_5b_vae_pth_state_dict_transform,
    )

    # Wan 2.2 TI2V-5B knobs (mirrors ``Wan22TI2V5BVAE*Config``).
    td = (False, True, True)
    with torch.device("meta"):
        enc = Encoder3d(
            dim=160,
            z_dim=96,
            temperal_downsample=td,
            in_channels=12,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attn_scales=(),
            dropout=0.0,
            pruning_rate=0.0,
            is_residual=True,
        )
        conv1 = CausalConv3d(96, 96, 1)
        dec = Decoder3d(
            dim=256,
            z_dim=48,
            temperal_upsample=tuple(reversed(td)),
            out_channels=12,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attn_scales=(),
            dropout=0.0,
            pruning_rate=0.0,
            is_residual=True,
        )
        conv2 = CausalConv3d(48, 48, 1)

    model_sd: dict[str, tuple[int, ...]] = {}
    for name, mod in (
        ("encoder", enc),
        ("conv1", conv1),
        ("decoder", dec),
        ("conv2", conv2),
    ):
        for k, v in mod.state_dict().items():
            model_sd[f"{name}.{k}"] = tuple(v.shape)

    # Synthesise upstream's native ``.pth`` layout from the model keys
    # (inverse of the production remap, authored independently here):
    # our grouped ``resnets.{j}`` / ``downsampler`` / ``upsampler`` become
    # the flat ``downsamples.{j}`` / ``upsamples.{j}`` sequential, with the
    # resample appended after the residual blocks (index = num_res_blocks
    # for the encoder, num_res_blocks + 1 for the decoder).
    n_enc_res, n_dec_res = 2, 3

    def to_native(key: str) -> str:
        m = re.match(r"^encoder\.downsamples\.(\d+)\.resnets\.(\d+)\.(.*)$", key)
        if m:
            return f"encoder.downsamples.{m[1]}.downsamples.{m[2]}.{m[3]}"
        m = re.match(r"^encoder\.downsamples\.(\d+)\.downsampler\.(.*)$", key)
        if m:
            return f"encoder.downsamples.{m[1]}.downsamples.{n_enc_res}.{m[2]}"
        m = re.match(r"^decoder\.upsamples\.(\d+)\.resnets\.(\d+)\.(.*)$", key)
        if m:
            return f"decoder.upsamples.{m[1]}.upsamples.{m[2]}.{m[3]}"
        m = re.match(r"^decoder\.upsamples\.(\d+)\.upsampler\.(.*)$", key)
        if m:
            return f"decoder.upsamples.{m[1]}.upsamples.{n_dec_res}.{m[2]}"
        return key  # middle / head / conv1 / conv2 are identical upstream

    native_sd = {
        to_native(k): torch.empty(shape, device="meta") for k, shape in model_sd.items()
    }
    # The synthetic native dict must itself be a bijection (no two model
    # keys collapsing to one native key).
    assert len(native_sd) == len(model_sd), "native-layout synthesis collided keys"

    remapped = wan22_ti2v_5b_vae_pth_state_dict_transform(native_sd)
    remapped_shapes = {k: tuple(v.shape) for k, v in remapped.items()}

    missing = set(model_sd) - set(remapped_shapes)  # would stay on meta
    extra = set(remapped_shapes) - set(model_sd)  # unexpected keys
    assert not missing, (
        f"remap leaves {len(missing)} model params uncovered: {sorted(missing)[:5]}"
    )
    assert not extra, (
        f"remap emits {len(extra)} keys not in the model: {sorted(extra)[:5]}"
    )
    assert remapped_shapes == model_sd, "remap changed a tensor shape"


@pytest.mark.ci_cpu
def test_wan22_vae_pth_remap_spot_checks_real_keys() -> None:
    """Spot-check the remap against real ``Wan2.2_VAE.pth`` key names.

    Guards the regex against the actual upstream key strings (taken from
    a ``torch.load`` key dump), not just the round-trip above.
    """
    from flashdreams.recipes.wan.autoencoder.vae import (
        wan22_ti2v_5b_vae_pth_state_dict_transform,
    )

    cases = {
        # encoder: residual block, shortcut, spatial resample, temporal conv
        "encoder.downsamples.0.downsamples.0.residual.2.weight": "encoder.downsamples.0.resnets.0.residual.2.weight",
        "encoder.downsamples.1.downsamples.0.shortcut.weight": "encoder.downsamples.1.resnets.0.shortcut.weight",
        "encoder.downsamples.0.downsamples.2.resample.1.weight": "encoder.downsamples.0.downsampler.resample.1.weight",
        "encoder.downsamples.1.downsamples.2.time_conv.weight": "encoder.downsamples.1.downsampler.time_conv.weight",
        # decoder: residual block + the upsampler resample / time_conv at index 3
        "decoder.upsamples.0.upsamples.0.residual.2.weight": "decoder.upsamples.0.resnets.0.residual.2.weight",
        "decoder.upsamples.0.upsamples.3.resample.1.weight": "decoder.upsamples.0.upsampler.resample.1.weight",
        "decoder.upsamples.0.upsamples.3.time_conv.weight": "decoder.upsamples.0.upsampler.time_conv.weight",
        # pass-through: middle / head / top-level convs are identical
        "encoder.middle.1.to_qkv.weight": "encoder.middle.1.to_qkv.weight",
        "decoder.head.2.weight": "decoder.head.2.weight",
        "conv1.weight": "conv1.weight",
    }
    fake = {k: torch.empty(1) for k in cases}
    out = wan22_ti2v_5b_vae_pth_state_dict_transform(fake)
    for src, want in cases.items():
        assert want in out, (
            f"{src!r} should remap to {want!r}; got keys {sorted(out)[:3]}..."
        )


# python tests/test_vae.py
if __name__ == "__main__":
    tokenizer_choices: list[Literal["lightvae", "vae"]] = ["lightvae", "vae"]
    detokenizer_choices: list[Literal["lighttae", "lightvae", "vae"]] = [
        "lighttae",
        "lightvae",
        "vae",
    ]
    for tokenizer_choice in tokenizer_choices:
        for detokenizer_choice in detokenizer_choices:
            test_tokenizer(tokenizer_choice, detokenizer_choice)
