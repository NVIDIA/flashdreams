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
