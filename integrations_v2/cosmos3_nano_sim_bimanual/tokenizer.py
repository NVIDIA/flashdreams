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

"""Qwen3-VL prompt tokens for AgiBot scene conditioning."""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import Tensor

DEFAULT_TOKENIZER = "Qwen/Qwen3-VL-8B-Instruct"

# Cosmos3 start-of-generation marker follows the prompt's EOS token.
_START_OF_GENERATION_TOKEN = "<|vision_start|>"


@functools.lru_cache(maxsize=4)
def load_tokenizer(tokenizer_path: str = DEFAULT_TOKENIZER) -> Any:
    """Load (and cache) the Qwen3-VL tokenizer via ``transformers.AutoTokenizer``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _generation_special_token_ids(tokenizer: Any) -> tuple[int, int]:
    return (
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids(_START_OF_GENERATION_TOKEN)),
    )


def tokenize_prompt(
    prompt: str,
    *,
    tokenizer_path: str = DEFAULT_TOKENIZER,
) -> Tensor:
    """Tokenize an unchanged GA caption followed by the generation markers.

    Args:
        prompt: Caption without appended resolution or action metadata.
        tokenizer_path: Hugging Face tokenizer repository or local directory.

    Returns:
        One-dimensional token IDs including EOS and vision-start markers.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    conversations = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    ids = list(
        tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
    )
    ids.extend(_generation_special_token_ids(tokenizer))
    return torch.tensor(ids, dtype=torch.long)
