# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-only output for the ordinary headless game and input loop."""

from typing import Any

import torch
from omnidreams_game_engine.config import ChunkConfig


class SemanticDebugPipeline:
    """Present engine conditioning without constructing or loading a world model.

    The rollout keeps its ordinary chunk, input, rules, and recording paths.
    Initial RGB and prompts are intentionally unused: every output pixel comes
    directly from the semantic renderer.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._chunks = ChunkConfig()
        self._closed = False

    def initialize_cache(self, **_conditioning: Any) -> None:
        self._check_open()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        return (
            self._chunks.initial_chunk_frames
            if autoregressive_index == 0
            else self._chunks.chunk_frames
        )

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: None,
        input: torch.Tensor,
    ) -> torch.Tensor:
        self._check_open()
        expected = self.get_num_output_frames(autoregressive_index)
        if input.ndim != 6 or tuple(input.shape[:4]) != (1, 1, expected, 3):
            raise ValueError("Semantic output requires single-view BVTCHW RGB chunks")
        return input

    def finalize(self, *, autoregressive_index: int, cache: None) -> None:
        self._check_open()

    def close(self) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Semantic debug pipeline is closed")
