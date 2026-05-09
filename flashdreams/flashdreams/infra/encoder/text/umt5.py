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

"""Wan 2.1 UMT5 text encoder, exposed as an infra Encoder."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Literal

import ftfy
import torch
from torch import Tensor
from transformers import AutoTokenizer, UMT5EncoderModel

from flashdreams.core.io.hf import should_use_local_files_only
from flashdreams.infra.encoder import Encoder, EncoderConfig


def prompt_clean(text: str) -> str:
    """Fix mojibake / HTML escapes / runs of whitespace in a prompt string."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass(kw_only=True)
class UMT5TextEncoderConfig(EncoderConfig):
    """Config for the Wan 2.x UMT5 text encoder."""

    _target: type = field(default_factory=lambda: UMT5TextEncoder)

    model_id_or_local_path: Literal[
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ] = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    """HF repo id or local snapshot path."""

    dtype: torch.dtype = torch.bfloat16


class UMT5TextEncoder(Encoder):
    """UMT5 text encoder for Wan 2.x.

    Stateless; called once before a rollout to encode prompts.

    Examples:

      >>> encoder = UMT5TextEncoderConfig().setup().to("cuda")
      >>> embeddings = encoder(["a cat playing piano"])
    """

    def __init__(self, config: UMT5TextEncoderConfig) -> None:
        super().__init__(config)
        self.config: UMT5TextEncoderConfig = config

        local_files_only = should_use_local_files_only(config.model_id_or_local_path)

        self.text_encoder = UMT5EncoderModel.from_pretrained(
            config.model_id_or_local_path,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        )
        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder.to(dtype=config.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_local_path,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

    @torch.no_grad()
    def forward(self, input: list[str]) -> Tensor:
        assert isinstance(input, list) and len(input) > 0, (
            "input must be a non-empty list of strings"
        )
        text = [prompt_clean(u) for u in input]

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            mask.to(self.text_encoder.device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(self.text_encoder.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        return prompt_embeds


if __name__ == "__main__":
    text_encoder = UMT5TextEncoderConfig().setup().to(torch.device("cuda"))

    text_embeddings = text_encoder(["hello world"])

    print(text_embeddings.shape)  # torch.Size([1, 512, 4096])
    print(text_embeddings.dtype)
    print(text_embeddings.device)
    print(text_embeddings.sum())  # 1.9766
