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

"""Cosmos-Reason1 (Qwen2.5-VL) text encoder used by Cosmos Predict2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import torch
import tyro
from loguru import logger
from torch import Tensor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from flashdreams.infra.encoder import Encoder, EncoderConfig


@dataclass(kw_only=True)
class CosmosReason1TextEncoderConfig(EncoderConfig):
    """Config for the Cosmos-Reason1 text encoder."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: CosmosReason1TextEncoder
    )

    model_name: str = "nvidia/Cosmos-Reason1-7B"
    """HF repo id of the underlying Qwen2.5-VL model."""

    revision: str = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    """HF commit hash to pin.

    Defaults to the Cosmos-Reason1.1 SFT checkpoint
    (``sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/iter_16000``)
    that the Cosmos-Predict 2.5 2B model was trained on.
    """

    max_length: int = 512
    """Token length to pad/truncate to."""

    dtype: torch.dtype = torch.bfloat16

    embedding_concat_strategy: str = "full_concat"
    """``"full_concat"`` (default, 100352 dims, matches upstream),
    ``"mean_pooling"``, or ``"pool_every_n_layers_and_concat"``."""

    n_layers_per_group: int = 5
    """Group size for the pool-every-N strategy."""


class CosmosReason1TextEncoder(Encoder):
    """Cosmos-Reason1 (Qwen2.5-VL) text encoder.

    Stateless. The default ``full_concat`` strategy concatenates all 28
    hidden layers into a 100,352-dim embedding (28 x 3584); the DiT
    projects this to 1024 via its ``crossattn_proj``.

    Examples:

      >>> encoder = CosmosReason1TextEncoderConfig().setup().to("cuda")
      >>> embeddings = encoder(["a beautiful sunset"]) # [1, 512, 100352]
    """

    FULL_CONCAT = "full_concat"
    MEAN_POOLING = "mean_pooling"
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __init__(self, config: CosmosReason1TextEncoderConfig) -> None:
        super().__init__(config)
        self.config: CosmosReason1TextEncoderConfig = config

        self.max_length = config.max_length
        self.dtype = config.dtype
        self.embedding_concat_strategy = config.embedding_concat_strategy
        self.n_layers_per_group = config.n_layers_per_group

        maybe_download_hf_repo_on_rank0(
            config.model_name,
            revision=config.revision,
        )

        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            revision=config.revision,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer

        logger.info(
            f"Loading Cosmos-Reason1 model from {config.model_name}"
            + (f"@{config.revision}" if config.revision else "")
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            revision=config.revision,
            local_files_only=True,
            dtype=config.dtype,
        )
        self.model.eval().requires_grad_(False)

        # ``transformers>=5.8`` nests LM dims under ``text_config``.
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = text_cfg.hidden_size  # 3584 for Cosmos-Reason1-7B
        self.num_layers = text_cfg.num_hidden_layers  # 28 for Cosmos-Reason1-7B

    def _mean_normalize(self, tensor: Tensor) -> Tensor:
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

    @torch.no_grad()
    def forward(self, input: list[str]) -> Tensor:
        assert isinstance(input, list) and len(input) > 0, (
            "input must be a non-empty list of strings"
        )

        input_ids_batch = []
        for prompt in input:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a helpful assistant who will provide "
                                "prompts to an image generator."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]

            formatted = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                return_tensors="pt",
            )
            # ``transformers>=5.8`` may return a ``BatchEncoding`` mapping here
            # instead of a raw tensor; keep backward compatibility with both.
            if not torch.is_tensor(formatted):
                formatted = formatted["input_ids"]
            if formatted.ndim == 1:
                formatted = formatted.unsqueeze(0)

            if formatted.shape[1] < self.max_length:
                pad_len = self.max_length - formatted.shape[1]
                padding = torch.full(
                    (1, pad_len),
                    self.tokenizer.pad_token_id or 0,
                    dtype=formatted.dtype,
                )
                formatted = torch.cat([formatted, padding], dim=1)
            else:
                formatted = formatted[:, : self.max_length]
            input_ids_batch.append(formatted)

        input_ids = torch.cat(input_ids_batch, dim=0).to(self.model.device)

        outputs = self.model(
            input_ids=input_ids, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors

        # Skip the embedding layer (index 0); normalize and combine.
        normalized_hidden_states = [
            self._mean_normalize(hidden_states[i]) for i in range(1, len(hidden_states))
        ]

        if self.embedding_concat_strategy == self.FULL_CONCAT:
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == self.MEAN_POOLING:
            text_embeddings = torch.stack(normalized_hidden_states).mean(dim=0)
        elif self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            pooled = []
            for i in range(0, len(normalized_hidden_states), self.n_layers_per_group):
                group = normalized_hidden_states[i : i + self.n_layers_per_group]
                pooled.append(torch.stack(group).mean(dim=0))
            text_embeddings = torch.cat(pooled, dim=-1)
        else:
            raise ValueError(
                f"Invalid embedding_concat_strategy: {self.embedding_concat_strategy}"
            )

        return text_embeddings

    @property
    def embedding_dim(self) -> int:
        if self.embedding_concat_strategy == self.FULL_CONCAT:
            return self.num_layers * self.hidden_size
        if self.embedding_concat_strategy == self.MEAN_POOLING:
            return self.hidden_size
        if self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            n_groups = (
                self.num_layers + self.n_layers_per_group - 1
            ) // self.n_layers_per_group
            return n_groups * self.hidden_size
        return self.hidden_size


if __name__ == "__main__":
    text_encoder = CosmosReason1TextEncoderConfig().setup().to(torch.device("cuda"))
    text_embeddings = text_encoder(["A beautiful sunset over a calm ocean."])
    print(f"text_embeddings.shape: {text_embeddings.shape}")
    print(f"text_embeddings.dtype: {text_embeddings.dtype}")
    print(f"text_embeddings.device: {text_embeddings.device}")
    print(f"text_embeddings.sum: {text_embeddings.sum()}")
