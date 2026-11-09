# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small streaming stack for CPU tests of the stream and the application.

A tiny network, a control encoder that pools the 832x480 control to the tiny
latent grid, a decoder that paints a latent back to 832x480 frames, and a
tokenizer stand-in, so :class:`~cosmos3_nano_sim_transfer.impl.stream.TransferStream`
runs its real rollout without weights or downloads.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from cosmos3_nano_sim_transfer.impl import stream as stream_module
from cosmos3_nano_sim_transfer.config import (
    EXPORT_METADATA_KEY,
    HistoryProfile,
    TransferModelConfig,
)
from cosmos3_nano_sim_transfer.impl.prompt import HEIGHT, WIDTH
from cosmos3_nano_sim_transfer.impl.stream import TransferStream
from cosmos_interactive._network import CosmosMoTNetwork, CosmosMoTNetworkConfig
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec
from torch import Tensor

TINY_LATENT_HW = (4, 6)
TINY_CHANNELS = 4


class PoolingEncoder:
    """Pools ``[T, 3, 480, 832]`` control frames to a ``[4, 1, 4, 6]`` latent; 1 frame, then 4."""

    def __init__(self) -> None:
        self.calls: list[Tensor] = []
        self.resets = 0
        self._latents = 0

    def reset(self) -> None:
        self.resets += 1
        self._latents = 0

    def encode(self, frames: Tensor) -> Tensor:
        expected = 1 if self._latents == 0 else 4
        assert tuple(frames.shape) == (expected, 3, HEIGHT, WIDTH), frames.shape
        self.calls.append(frames.clone())
        self._latents += 1
        pooled = F.adaptive_avg_pool2d(frames.float().mean(0), TINY_LATENT_HW)
        return torch.cat([pooled, pooled.mean(0, keepdim=True)])[:, None]


class PaintingDecoder:
    """Paints a ``[4, 1, 4, 6]`` latent to ``[T, 3, 480, 832]`` in ``[-1, 1]``; 1 frame, then 4."""

    def __init__(self) -> None:
        self.resets = 0
        self._latents = 0

    def reset(self) -> None:
        self.resets += 1
        self._latents = 0

    def decode(self, latent: Tensor) -> Tensor:
        frames = 1 if self._latents == 0 else 4
        self._latents += 1
        image = torch.tanh(latent[:3, 0].float())[None]
        image = F.interpolate(image, size=(HEIGHT, WIDTH), mode="nearest")
        return image.expand(frames, -1, -1, -1).contiguous()


def fake_prompt_ids(
    caption: str,
    *,
    num_frames: int,
    fps: float,
    tokenizer_path: Any,
    max_tokens: Any = None,
    hint: str | None = None,
    emphasize_control_in_prompt: bool = False,
    aspect_ratio: str | None = None,
) -> Tensor:
    """Ids that change with the caption and the stated clip length (vocabulary of 50)."""
    del fps, tokenizer_path, max_tokens, hint, emphasize_control_in_prompt, aspect_ratio
    ids = [3 + ord(c) % 40 for c in caption[:6]] + [num_frames % 47]
    return torch.tensor(ids, dtype=torch.long)


@pytest.fixture(scope="session")
def tiny_model() -> TransferModelConfig:
    data = json.loads(
        (Path(__file__).parent / "data" / "checkpoint_metadata.json").read_text()
    )
    return TransferModelConfig.from_metadata(data[EXPORT_METADATA_KEY])


@pytest.fixture(scope="session")
def tiny_network() -> CosmosMoTNetwork:
    torch.manual_seed(0)
    spec = Cosmos3InteractiveSpec(
        llm=LLMSpec(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=50,
            mrope_section=(4, 2, 2),
        ),
        vfm=VFMSpec(vae_latent_channels=TINY_CHANNELS),
    )
    net = CosmosMoTNetworkConfig(
        spec=spec, dtype=torch.float32, include_vision_tower=False
    ).setup()
    assert isinstance(net, CosmosMoTNetwork)
    return net.eval()


@pytest.fixture
def make_stream(
    tiny_network: CosmosMoTNetwork,
    tiny_model: TransferModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., TransferStream]:
    """Build a :class:`TransferStream` on the tiny stack (keyword overrides allowed,
    ``model`` among them)."""
    monkeypatch.setattr(stream_module, "prompt_token_ids", fake_prompt_ids)

    def make(**overrides: Any) -> TransferStream:
        model = overrides.pop("model", tiny_model)
        settings: dict[str, Any] = {
            "encoder": PoolingEncoder(),
            "decoder": PaintingDecoder(),
            "tokenizer_path": "unused",
            "hint": "seg",
            "history": HistoryProfile(3, 1),
            "fps": 30.0,
            "latent_hw": TINY_LATENT_HW,
            "noise_dtype": torch.float32,
            **overrides,
        }
        return TransferStream(tiny_network, model, **settings)

    return make
