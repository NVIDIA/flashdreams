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


def _build_wan22_ti2v_5b_vae_modules() -> dict[str, tuple[int, ...]]:
    """Build the Wan 2.2 TI2V-5B ``WanVAE`` module tree on ``meta``.

    Mirrors the knobs in :class:`Wan22TI2V5BVAEEncoderConfig` /
    :class:`Wan22TI2V5BVAEDecoderConfig`. Returns ``{key: shape}`` for the
    full encoder + decoder + top-level convs, with no checkpoint download.
    """
    from flashdreams.recipes.wan.autoencoder.vae import (
        CausalConv3d,
        Decoder3d,
        Encoder3d,
    )

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

    model: dict[str, tuple[int, ...]] = {}
    for name, mod in (
        ("encoder", enc),
        ("conv1", conv1),
        ("decoder", dec),
        ("conv2", conv2),
    ):
        for k, v in mod.state_dict().items():
            model[f"{name}.{k}"] = tuple(v.shape)
    return model


@pytest.mark.ci_cpu
def test_wan22_vae_native_pth_loads_without_remap() -> None:
    """The native ``Wan2.2_VAE.pth`` layout matches the module tree 1:1.

    The production configs pin :data:`WAN22_TI2V_5B_VAE_PATH` with
    ``state_dict_transform=None``: the native keys must land on the
    ``WanVAE`` modules directly, with nothing left on ``meta``. This builds
    the module tree on ``meta`` (no download) and checks real
    ``Wan2.2_VAE.pth`` key strings (from a ``torch.load`` key dump) are
    present verbatim, and that the renamed flat ``downsamples`` /
    ``upsamples`` layout carries no residue of the old grouped names.
    """
    model = _build_wan22_ti2v_5b_vae_modules()

    # Real upstream ``Wan2.2_VAE.pth`` keys spanning every down/up leaf kind
    # (residual conv, residual shortcut, spatial resample, temporal conv)
    # plus pass-through mid/head/top-level params.
    real_native_keys = [
        "encoder.downsamples.0.downsamples.0.residual.2.weight",
        "encoder.downsamples.1.downsamples.0.shortcut.weight",
        "encoder.downsamples.0.downsamples.2.resample.1.weight",
        "encoder.downsamples.1.downsamples.2.time_conv.weight",
        "decoder.upsamples.0.upsamples.0.residual.2.weight",
        "decoder.upsamples.0.upsamples.3.resample.1.weight",
        "decoder.upsamples.0.upsamples.3.time_conv.weight",
        "encoder.middle.1.to_qkv.weight",
        "decoder.head.2.weight",
        "conv1.weight",
    ]
    missing = [k for k in real_native_keys if k not in model]
    assert not missing, f"native .pth keys absent from the module tree: {missing}"

    stale = [
        k
        for k in model
        if ".resnets." in k or ".downsampler." in k or ".upsampler." in k
    ]
    assert not stale, f"grouped names survived the native-layout rename: {stale[:5]}"


@pytest.mark.ci_cpu
def test_wan22_vae_diffusers_remap_still_targets_real_params() -> None:
    """The opt-in diffusers remap lands on real module keys after the rename.

    Guards the down/up-block rules (whose targets moved from the grouped
    ``resnets`` / ``downsampler`` / ``upsampler`` names to the flat
    ``downsamples.{j}`` / ``upsamples.{j}`` layout) against representative
    diffusers ``AutoencoderKLWan`` key strings.
    """
    from flashdreams.recipes.wan.autoencoder.vae import (
        wan22_ti2v_5b_vae_state_dict_transform,
    )

    model = _build_wan22_ti2v_5b_vae_modules()
    # RMS norms carry ``.gamma`` (not ``.weight``); convs carry ``.weight``.
    cases = {
        "encoder.down_blocks.0.resnets.0.norm1.gamma": "encoder.downsamples.0.downsamples.0.residual.0.gamma",
        "encoder.down_blocks.0.resnets.1.conv2.weight": "encoder.downsamples.0.downsamples.1.residual.6.weight",
        "encoder.down_blocks.1.resnets.0.conv_shortcut.weight": "encoder.downsamples.1.downsamples.0.shortcut.weight",
        "encoder.down_blocks.0.downsampler.resample.1.weight": "encoder.downsamples.0.downsamples.2.resample.1.weight",
        "decoder.up_blocks.0.resnets.2.norm2.gamma": "decoder.upsamples.0.upsamples.2.residual.3.gamma",
        "decoder.up_blocks.0.upsampler.time_conv.weight": "decoder.upsamples.0.upsamples.3.time_conv.weight",
    }
    out = wan22_ti2v_5b_vae_state_dict_transform({k: torch.empty(1) for k in cases})
    for src, want in cases.items():
        assert want in out, (
            f"{src!r} should remap to {want!r}; got {sorted(out)[:3]}..."
        )
        assert want in model, f"remap target {want!r} is not a real module param"


@torch.no_grad()
@pytest.mark.manual
def test_wan22_vae_native_pth_and_diffusers_weights_identical() -> None:
    """Native ``.pth`` (no remap) and the diffusers remap yield identical weights.

    The verification behind defaulting to ``Wan2.2_VAE.pth``: loading the
    native checkpoint directly and loading the diffusers shard through
    :func:`wan22_ti2v_5b_vae_state_dict_transform` must produce the same
    state dict, so the switch is a pure checkpoint-source change.

    Marked ``manual``: downloads both checkpoints (~few GB) from
    HuggingFace. Set ``WAN22_VAE_PTH`` to a local ``Wan2.2_VAE.pth`` to skip
    that download.
    """
    import os

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from flashdreams.recipes.wan.autoencoder.vae import (
        wan22_ti2v_5b_vae_state_dict_transform,
    )

    diff_path = hf_hub_download(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "vae/diffusion_pytorch_model.safetensors"
    )
    pth_path = os.environ.get("WAN22_VAE_PTH") or hf_hub_download(
        "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"
    )

    from_diffusers = wan22_ti2v_5b_vae_state_dict_transform(load_file(diff_path))
    native = torch.load(pth_path, map_location="cpu", weights_only=True)
    while isinstance(native, dict) and "state_dict" in native and len(native) <= 4:
        native = native["state_dict"]

    assert set(from_diffusers) == set(native), "remapped key sets differ"
    worst = max(
        (from_diffusers[k].float() - native[k].float()).abs().max().item()
        for k in from_diffusers
    )
    assert worst == 0.0, f"weights differ between checkpoints (max |delta| = {worst})"


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
