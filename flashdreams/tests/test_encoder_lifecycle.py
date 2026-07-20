# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.infra.acceleration.encoder_lifecycle import (
    ensure_one_shot_encoder,
    move_tensors_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
    setup_one_shot_encoder,
)

pytestmark = pytest.mark.ci_cpu


def test_setup_one_shot_encoder_moves_torch_module_to_device() -> None:
    config = _EncoderConfig(_FakeModule())

    encoder = setup_one_shot_encoder(config, device=torch.device("cpu"))

    assert isinstance(encoder, _FakeModule)
    assert encoder.to_calls == [{"device": torch.device("cpu")}]


def test_ensure_one_shot_encoder_reuses_loaded_encoder() -> None:
    existing = object()
    config = _EncoderConfig(object())

    encoder = ensure_one_shot_encoder(
        existing,
        config,
        device=torch.device("cpu"),
    )

    assert encoder is existing
    assert config.setup_calls == 0


def test_ensure_one_shot_encoder_handles_missing_optional_config() -> None:
    assert (
        ensure_one_shot_encoder(
            None,
            None,
            name="image_encoder",
            required=False,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="text_encoder"):
        ensure_one_shot_encoder(None, None, name="text_encoder")


def test_release_one_shot_encoder_references_clears_attrs_and_cuda_cache() -> None:
    owner = SimpleNamespace(text_encoder=object(), image_encoder=None)
    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    released = release_one_shot_encoder_references(
        owner,
        "text_encoder",
        "image_encoder",
        device="cuda:0",
        synchronize_cuda=True,
        torch_module=fake_torch,
    )

    assert released == ("text_encoder",)
    assert owner.text_encoder is None
    assert owner.image_encoder is None
    assert fake_torch.cuda.synchronize_calls == ["cuda:0"]
    assert fake_torch.cuda.empty_cache_calls == 1


def test_move_tensors_to_cpu_recurses_through_containers() -> None:
    tensor = torch.ones(2)
    result = move_tensors_to_cpu(
        {
            "tensor": tensor,
            "nested": [tensor],
            "tuple": (tensor,),
            "plain": "value",
        }
    )

    assert result["tensor"].device.type == "cpu"
    assert result["nested"][0].device.type == "cpu"
    assert result["tuple"][0].device.type == "cpu"
    assert result["plain"] == "value"


def test_run_one_shot_encoder_stage_moves_result_and_releases() -> None:
    released: list[str] = []

    result = run_one_shot_encoder_stage(
        lambda: {"tensor": torch.ones(1)},
        release=lambda: released.append("released"),
    )

    assert result["tensor"].device.type == "cpu"
    assert released == ["released"]


class _FakeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[dict[str, Any]] = []

    def to(self, *args: Any, **kwargs: Any) -> "_FakeModule":
        del args
        self.to_calls.append(kwargs)
        return self


class _EncoderConfig:
    def __init__(self, encoder: object) -> None:
        self.encoder = encoder
        self.setup_calls = 0

    def setup(self) -> object:
        self.setup_calls += 1
        return self.encoder


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronize_calls: list[object] = []
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self, device: object | None = None) -> None:
        self.synchronize_calls.append(device)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1
