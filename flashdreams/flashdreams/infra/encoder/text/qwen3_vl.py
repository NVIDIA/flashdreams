# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless Qwen3-VL hidden-state conditioning through Transformers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from flashdreams.infra.encoder import Encoder, EncoderConfig


@dataclass(kw_only=True)
class Qwen3VLEncoderConfig(EncoderConfig):
    """Checkpoint source and raw hidden-state selection."""

    _target: type["Qwen3VLEncoder"] = field(default_factory=lambda: Qwen3VLEncoder)
    model_name: str = "MiniMaxAI/MiniMax-H3"
    """Repository or local checkpoint directory."""
    revision: str | None = None
    """Pinned checkpoint revision when using the Hub."""
    cache_dir: str | None = None
    """Optional Hugging Face cache directory."""
    hidden_layer: int = 50
    """Raw intermediate hidden state; must precede the final normalized state."""
    dtype: torch.dtype = torch.bfloat16
    """Weight and output precision."""
    subfolder: str = "text_encoder"
    """Checkpoint partition containing the conditioner."""
    processor_subfolder: str = "processor"
    """Checkpoint partition containing Qwen's media processor."""
    tokenizer_subfolder: str = "tokenizer"
    """Checkpoint partition containing the presentation tokenizer."""
    local_files_only: bool = False
    """Require all checkpoint files to be present locally."""
    loading_device: str | None = None
    """Place checkpoint tensors directly on this device to avoid a CPU weight copy."""


class Qwen3VLEncoder(Encoder):
    """Selected Qwen3-VL state without a language-model projection."""

    def __init__(self, config: Qwen3VLEncoderConfig) -> None:
        super().__init__(config)
        from transformers import Qwen2TokenizerFast, Qwen3VLModel, Qwen3VLProcessor

        kwargs = dict(
            revision=config.revision,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        self.model = (
            Qwen3VLModel.from_pretrained(
                config.model_name,
                subfolder=config.subfolder,
                dtype=config.dtype,
                device_map=config.loading_device,
                **kwargs,
            )
            .eval()
            .requires_grad_(False)
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            config.model_name, subfolder=config.processor_subfolder, **kwargs
        )
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            config.model_name, subfolder=config.tokenizer_subfolder, **kwargs
        )
        if (
            not 0
            <= config.hidden_layer
            < self.model.config.text_config.num_hidden_layers
        ):
            raise ValueError(
                "Qwen conditioning must select a raw intermediate hidden layer"
            )

    @torch.no_grad()
    def forward(self, input: dict[str, Any]) -> torch.Tensor:
        """Encode token IDs and optional processor-produced vision tensors."""
        device = self.model.device
        token_ids = input["token_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        modalities = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]),
            dtype=torch.long,
            device=device,
        )
        vision = {
            name: value.to(device=device, dtype=self.model.dtype)
            if name.startswith("pixel_")
            else value.to(device=device)
            for name, value in input.get("vision_inputs", {}).items()
        }
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=modalities,
            use_cache=False,
            output_hidden_states=True,
            **vision,
        )
        return output.hidden_states[self.config.hidden_layer].to(
            dtype=self.config.dtype
        )
