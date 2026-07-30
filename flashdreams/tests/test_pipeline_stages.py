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

"""CPU tests for independently deployable pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from flashdreams.infra.pipeline import (
    DecoderStage,
    DiffusionStage,
    StreamingEncoderStage,
)

pytestmark = pytest.mark.ci_cpu


@dataclass
class _Config:
    target: nn.Module

    def setup(self) -> nn.Module:
        return self.target


class _Encoder(nn.Module):
    def initialize_autoregressive_cache(self, *, value: int) -> dict[str, int]:
        return {"value": value}

    def forward(
        self,
        *,
        input: Tensor,
        autoregressive_index: int,
        cache: dict[str, int],
    ) -> Tensor:
        cache["value"] += autoregressive_index
        return input + cache["value"]


class _Decoder(nn.Module):
    def initialize_autoregressive_cache(self) -> list[int]:
        return []

    def forward(
        self,
        *,
        input: Tensor,
        autoregressive_index: int,
        cache: list[int],
    ) -> Tensor:
        cache.append(autoregressive_index)
        return input * 2


class _Transformer:
    device = torch.device("cpu")

    def initialize_autoregressive_cache(self, *, prompt: Tensor) -> dict[str, Tensor]:
        return {"prompt": prompt}


class _DiffusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer()
        self.finalized: list[object] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: dict[str, Tensor],
        input: Tensor,
    ) -> tuple[Tensor, object]:
        return input + cache["prompt"], SimpleNamespace(index=autoregressive_index)

    def finalize(self, final_state: object) -> None:
        self.finalized.append(final_state)


def test_encoder_and_decoder_stages_own_only_their_component() -> None:
    encoder_stage = StreamingEncoderStage(cast(Any, _Config(_Encoder())))
    encoder_cache = encoder_stage.initialize_cache(value=3)
    encoded = encoder_stage.encode(torch.tensor(2), 1, encoder_cache)
    assert encoded.item() == 6

    decoder_stage = DecoderStage(cast(Any, _Config(_Decoder())))
    decoder_cache = decoder_stage.initialize_cache()
    decoded = decoder_stage.decode(torch.tensor(4), 0, decoder_cache)
    assert decoded.item() == 8
    assert decoder_cache == [0]


def test_diffusion_stage_keeps_finalization_state_on_dit_worker() -> None:
    model = _DiffusionModel()
    stage = DiffusionStage(cast(Any, _Config(model)))
    cache = stage.initialize_cache(prompt=torch.tensor(5))

    output = stage.generate(0, cache, input=torch.tensor(7))
    assert output.item() == 12
    assert cache.final_state is not None

    stage.finalize(0, cache)
    assert cache.final_state is None
    assert len(model.finalized) == 1


def test_diffusion_stage_rejects_out_of_order_ar_steps() -> None:
    stage = DiffusionStage(cast(Any, _Config(_DiffusionModel())))
    cache = stage.initialize_cache(prompt=torch.tensor(0))
    with pytest.raises(AssertionError, match="expected 0, got 1"):
        stage.generate(1, cache, input=torch.tensor(0))
