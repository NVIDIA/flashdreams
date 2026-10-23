# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL prompt packing contracts on CPU."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from cosmos3_nano_sim_bimanual import tokenizer

pytestmark = pytest.mark.ci_cpu


class _FakeTokenizer:
    eos_token_id = 7

    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []
        self.chat_kwargs: dict[str, Any] = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|vision_start|>"
        return 9

    def apply_chat_template(
        self, conversations: list[dict[str, str]], **kwargs: Any
    ) -> list[int]:
        self.conversations.append(conversations)
        self.chat_kwargs = kwargs
        return [1, 2, 3]


@pytest.mark.parametrize(
    "prompt", ["Fold the shirt", "Fold the shirt.", "  Keep original spacing.\n"]
)
def test_tokenize_prompt_matches_hand_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    fake = _FakeTokenizer()
    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda _: fake)

    ids = tokenizer.tokenize_prompt(prompt, tokenizer_path="fake")

    assert fake.conversations == [
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    ]
    assert fake.chat_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "add_vision_id": False,
        "return_dict": False,
    }
    torch.testing.assert_close(ids, torch.tensor([1, 2, 3, 7, 9]))
