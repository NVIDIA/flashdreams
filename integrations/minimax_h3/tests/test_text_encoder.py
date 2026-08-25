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

"""CPU contracts for native MiniMax H3 Qwen conditioning."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from minimax_h3.text_encoder import (
    build_fl2va_presentation,
    build_t2va_presentation,
    encode_presentation,
)

pytestmark = pytest.mark.ci_cpu


class _Tokenizer:
    """Deterministic tokenizer double for presentation ordering."""

    _texts = {
        "prompt": [10, 11],
        "<Picture 1>: ": [21],
        "<Picture 2>: ": [22, 23],
        "": [],
    }
    _special = {
        "<|vision_start|>": 900,
        "<|image_pad|>": 901,
        "<|vision_end|>": 903,
    }

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": list(self._texts[text])}

    def convert_tokens_to_ids(self, token: str) -> int:
        """Return one frozen special-token id."""
        return self._special[token]


class _ImageProcessor:
    merge_size = 2

    def __call__(
        self, *, images: list[object], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert images == ["first", "last"]
        return {
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(2, 6),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
        }


class _Processor:
    image_processor = _ImageProcessor()

    def create_mm_token_type_ids(self, batches: list[list[int]]) -> list[list[int]]:
        """Tag the fake presentation as Qwen-internal text rows."""
        return [[0] * len(token_ids) for token_ids in batches]


class _QwenBase:
    """Record one submodel call and return frozen layer-50 features."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        rows = kwargs["input_ids"].shape[1]
        hidden_states = [torch.zeros(1, rows, 5120) for _ in range(51)]
        hidden_states[50] = torch.full((1, rows, 5120), 50.0)
        return SimpleNamespace(hidden_states=hidden_states)


class _TextEncoder:
    """Minimal Qwen3-VL top-level model double."""

    dtype = torch.bfloat16

    def __init__(self, num_hidden_layers: int = 64) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
        )
        self.model = _QwenBase()


def test_t2va_presentation_is_prompt_verbatim() -> None:
    """Add neither a chat template nor special tokens to T2VA prompts."""
    presentation = build_t2va_presentation(_Tokenizer(), "prompt")
    assert presentation.token_ids == (10, 11)
    assert presentation.token_tags.tolist() == [1, 1]
    assert presentation.vision_inputs == {}


def test_fl2va_presentation_labels_and_tags_vision_blocks() -> None:
    """Number keyframes, preserve order, and tag complete vision blocks."""
    presentation = build_fl2va_presentation(
        _Tokenizer(), _Processor(), "prompt", ["first", "last"]
    )
    assert presentation.token_ids == (
        21,
        900,
        901,
        901,
        901,
        901,
        903,
        22,
        23,
        900,
        901,
        901,
        903,
        10,
        11,
    )
    assert presentation.token_tags.tolist() == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
    ]
    assert tuple(presentation.vision_inputs["pixel_values"].shape) == (2, 6)


def test_encode_presentation_reads_layer_50_without_lm_head() -> None:
    """Call Qwen's base model and forward image tensors at model dtype."""
    processor = _Processor()
    presentation = build_fl2va_presentation(
        _Tokenizer(), processor, "prompt", ["first", "last"]
    )
    text_encoder = _TextEncoder()
    condition = encode_presentation(
        text_encoder, processor, presentation, device="cpu"
    )

    assert condition.prompt_embeds.dtype == torch.bfloat16
    assert tuple(condition.prompt_embeds.shape) == (1, 15, 5120)
    assert bool((condition.prompt_embeds == 50).all())
    assert torch.equal(condition.text_token_tags, presentation.token_tags)
    assert text_encoder.model.kwargs is not None
    assert text_encoder.model.kwargs["use_cache"] is False
    assert text_encoder.model.kwargs["output_hidden_states"] is True
    assert text_encoder.model.kwargs["pixel_values"].dtype == torch.bfloat16
    assert text_encoder.model.kwargs["image_grid_thw"].dtype == torch.long
    assert tuple(text_encoder.model.kwargs["mm_token_type_ids"].shape) == (1, 15)


def test_text_conditioning_rejects_unsupported_presentations() -> None:
    """Reject empty prompts, excess keyframes, and truncated conditioners."""
    with pytest.raises(ValueError, match="at least one token"):
        build_t2va_presentation(_Tokenizer(), "")
    with pytest.raises(ValueError, match="at most"):
        build_fl2va_presentation(
            _Tokenizer(), _Processor(), "prompt", ["first", "last", "third"]
        )
    with pytest.raises(ValueError, match="only 50 layers"):
        encode_presentation(
            _TextEncoder(num_hidden_layers=50),
            _Processor(),
            build_t2va_presentation(_Tokenizer(), "prompt"),
            device="cpu",
        )
