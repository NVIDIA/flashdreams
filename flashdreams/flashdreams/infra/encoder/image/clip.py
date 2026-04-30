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

"""Wan CLIP image encoder, exposed as a infra :class:`Encoder`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import Tensor
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from flashdreams.core.io.hf import should_use_local_files_only
from flashdreams.infra.encoder import Encoder, EncoderConfig


@dataclass(kw_only=True)
class CLIPImageEncoderConfig(EncoderConfig):
    _target: type["CLIPImageEncoder"] = field(default_factory=lambda: CLIPImageEncoder)

    model_id_or_local_path: Literal[
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ] = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    dtype: torch.dtype = torch.bfloat16


class CLIPImageEncoder(Encoder):
    """CLIP image encoder used by Wan I2V.

    Stateless: no per-rollout cache, so :meth:`forward` takes only ``input``.
    Input images are expected in shape ``[..., C, H, W]`` with values in
    ``[-1, 1]``.
    """

    def __init__(self, config: CLIPImageEncoderConfig) -> None:
        super().__init__(config)
        self.config: CLIPImageEncoderConfig = config

        local_files_only = should_use_local_files_only(config.model_id_or_local_path)

        self.image_encoder = CLIPVisionModel.from_pretrained(
            config.model_id_or_local_path,
            subfolder="image_encoder",
            dtype=config.dtype,
            local_files_only=local_files_only,
        )
        self.image_encoder.eval().requires_grad_(False)

        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.model_id_or_local_path,
            subfolder="image_processor",
            local_files_only=local_files_only,
        )

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        """Encode images of shape ``[..., C, H, W]`` (range ``[-1, 1]``).

        Returns embeddings of shape ``[..., 257, 1280]``.
        """
        batch_shape = input.shape[:-3]
        batch_size = math.prod(batch_shape)
        images = input.reshape(batch_size, *input.shape[-3:])

        device = self.image_encoder.device
        images = (images + 1) / 2.0
        images = cast(
            Tensor,
            self.image_processor(
                images=images.to(dtype=torch.float32),
                return_tensors="pt",
                do_rescale=False,
            ),
        ).to(device, dtype=self.image_encoder.dtype)
        image_embeds: BaseModelOutputWithPooling = self.image_encoder(
            **images,  # ty: ignore[invalid-argument-type]
            output_hidden_states=True,
        )

        assert image_embeds.hidden_states is not None
        output: Tensor = image_embeds.hidden_states[-2]
        return output.reshape(*batch_shape, *output.shape[-2:])


# python -m recipes.wan21.image_encoder
if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16

    image_encoder = CLIPImageEncoderConfig().setup().to(device)

    image = torch.rand(1, 2, 3, 224, 224, device=device, dtype=dtype) * 2.0 - 1.0
    image_embeds = image_encoder(image)

    print(image_embeds.shape)  # torch.Size([1, 2, 257, 1280])
    print(image_embeds.dtype)
    print(image_embeds.device)
    print(image_embeds.sum())
