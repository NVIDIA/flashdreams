# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL prompt tokenization for Cosmos3 conditioning.

Append resolution metadata, apply the Qwen3-VL chat template, then append the
end-of-turn and vision-generation tokens. The checkpoint determines whether a
system prompt is required. The tokenizer is ``Qwen/Qwen3-VL-8B-Instruct``.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import Tensor

DEFAULT_TOKENIZER = "Qwen/Qwen3-VL-8B-Instruct"

# Resolution metadata is part of the checkpoint's prompt format.
_RESOLUTION_TEMPLATE_IMAGE = "This image is of {height}x{width} resolution."
_RESOLUTION_TEMPLATE_VIDEO = "This video is of {height}x{width} resolution."
# Optional system prompts; preserve the exact training text.
_SYSTEM_PROMPT_IMAGE = (
    "You are a helpful assistant who will generate images from a give prompt."
)
_SYSTEM_PROMPT_VIDEO = (
    "You are a helpful assistant who will generate videos from a give prompt."
)
# Generation suffix appended after the chat template.
_START_OF_GENERATION_TOKEN = "<|vision_start|>"


@functools.lru_cache(maxsize=4)
def load_tokenizer(tokenizer_path: str = DEFAULT_TOKENIZER) -> Any:
    """Load (and cache) the Qwen3-VL tokenizer via ``transformers.AutoTokenizer``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def generation_special_token_ids(
    tokenizer_path: str = DEFAULT_TOKENIZER,
) -> tuple[int, int]:
    """Return the packed ``(eos, vision_start)`` generation-suffix IDs."""
    tokenizer = load_tokenizer(tokenizer_path)
    return (
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids(_START_OF_GENERATION_TOKEN)),
    )


def pack_generation_text_ids(
    text_ids: Tensor, *, tokenizer_path: str = DEFAULT_TOKENIZER
) -> Tensor:
    """Append the generation suffix to pre-tokenized IDs, at most once.

    Recorded scene pickles ship their caption already tokenized but without the
    generation marker. Appending is idempotent, so a scene that already carries
    the suffix passes through unchanged.
    """
    ids = torch.as_tensor(text_ids, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("text token IDs must be nonempty")
    suffix = torch.tensor(
        generation_special_token_ids(tokenizer_path), dtype=ids.dtype, device=ids.device
    )
    if ids.numel() >= 2 and torch.equal(ids[-2:], suffix):
        return ids
    return torch.cat((ids, suffix))


def _with_resolution_suffix(
    caption: str, height: int, width: int, *, is_video: bool
) -> str:
    """Append the model's resolution metadata string (``ResolutionTextInfo`` convention)."""
    template = _RESOLUTION_TEMPLATE_VIDEO if is_video else _RESOLUTION_TEMPLATE_IMAGE
    metadata = template.format(height=height, width=width)
    separator = " " if caption.rstrip().endswith(".") else ". "
    return caption + separator + metadata


def tokenize_prompt(
    prompt: str,
    *,
    height: int,
    width: int,
    tokenizer_path: str = DEFAULT_TOKENIZER,
    is_video: bool = True,
    use_system_prompt: bool = False,
    system_prompt: str | None = None,
    append_resolution: bool = True,
    append_generation_tokens: bool = True,
) -> Tensor:
    """Tokenize a prompt to ``[N_text]`` IDs with the checkpoint's prompt format.

    Args:
        prompt: raw text prompt / caption.
        height, width: target pixel resolution for the resolution suffix
            (480p landscape is ``height=480, width=832``; equals ``gh*32``/``gw*32``).
        is_video: use the video (vs image) resolution template + system prompt.
        append_resolution: Append the resolution metadata suffix.
        append_generation_tokens: Append ``eos`` and ``<|vision_start|>``.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    caption = (
        _with_resolution_suffix(prompt, height, width, is_video=is_video)
        if append_resolution
        else prompt
    )

    conversations: list[dict[str, str]] = []
    if system_prompt is not None:
        conversations.append({"role": "system", "content": system_prompt})
    elif use_system_prompt:
        conversations.append(
            {
                "role": "system",
                "content": _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE,
            }
        )
    conversations.append({"role": "user", "content": caption})

    ids = list(
        tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
    )
    if append_generation_tokens:
        ids.extend(generation_special_token_ids(tokenizer_path))
    return torch.tensor(ids, dtype=torch.long)
