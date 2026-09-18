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

import time
from collections import OrderedDict
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

    run_on_cpu: bool = False
    """Keep the bf16 model on the host and run it there.

    ``.to(device)`` then records the compute device instead of moving the
    weights, and ``forward`` returns embeddings on that device. A 512-token
    encode takes seconds on a desktop CPU, and prompt encodes are rare
    (rollout start, prompt edits), so the encoder's VRAM is freed for the
    rollout on 32 GB GPUs.
    """

    embedding_cache_size: int = 0
    """Number of most recently used prompt batches whose embeddings are kept.

    Game hosts re-encode the same scene prompt on every restart. With
    ``run_on_cpu`` the cache lives in host memory and hits are copied to the
    compute device; otherwise entries stay on the encoder's device and the
    returned tensor must not be modified in place. Each full-concat entry is
    about 100 MiB per prompt. ``0`` disables the cache.
    """


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
        self._compute_device: torch.device | None = None
        self._embedding_cache: OrderedDict[tuple[str, ...], Tensor] = OrderedDict()

        # ``transformers>=5.8`` nests LM dims under ``text_config``.
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = text_cfg.hidden_size  # 3584 for Cosmos-Reason1-7B
        self.num_layers = text_cfg.num_hidden_layers  # 28 for Cosmos-Reason1-7B

    def _mean_normalize(self, tensor: Tensor) -> Tensor:
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

    def _apply(self, fn, recurse=True):
        """Keep host-resident weights on the host across device moves.

        ``to``/``cuda``/``half`` all funnel through ``_apply``. With
        ``run_on_cpu`` a probe decides what the call asked for: a device move
        records the compute device (or clears it for ``cpu``) instead of
        moving the weights, and a dtype change is applied to the host weights.
        Every such change invalidates the embedding cache.
        """
        if not self.config.run_on_cpu:
            return super()._apply(fn, recurse)
        # Probe from the current compute device so a dtype-only call (``half``)
        # leaves the recorded device alone while ``cpu()`` clears it.
        probe_device = self._compute_device or torch.device("cpu")
        try:
            probe = fn(torch.zeros((), dtype=self.dtype, device=probe_device))
        except NotImplementedError:
            # Meta tensors cannot be copied out; probe from the host instead.
            probe = fn(torch.zeros((), dtype=self.dtype))
        self._compute_device = None if probe.device.type == "cpu" else probe.device
        self._embedding_cache.clear()
        if probe.dtype != self.dtype:
            self.dtype = probe.dtype
            return super()._apply(
                lambda t: t.to(dtype=probe.dtype) if t.is_floating_point() else t,
                recurse,
            )
        return self

    @torch.no_grad()
    def forward(self, input: list[str]) -> Tensor:
        key = tuple(input)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return self._to_compute_device(cached)
        started = time.perf_counter()
        text_embeddings = self._encode(input)
        if self.config.run_on_cpu:
            logger.info(
                "Cosmos-Reason1 host encode of {} prompt(s) took {:.1f}s",
                len(input),
                time.perf_counter() - started,
            )
        if self.config.embedding_cache_size > 0:
            self._embedding_cache[key] = text_embeddings
            while len(self._embedding_cache) > self.config.embedding_cache_size:
                self._embedding_cache.popitem(last=False)
        return self._to_compute_device(text_embeddings)

    def _to_compute_device(self, embeddings: Tensor) -> Tensor:
        device = self._compute_device
        if device is None or embeddings.device == device:
            return embeddings
        return embeddings.to(device)

    def _encode(self, input: list[str]) -> Tensor:
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
