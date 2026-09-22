# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host execution and embedding cache of the Cosmos-Reason1 text encoder.

The real Qwen2.5-VL weights are replaced by a tiny module so the device and
cache semantics can be checked on CPU-only CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import flashdreams.infra.encoder.text.cosmos_reason1 as cosmos_reason1
from flashdreams.infra.encoder.text.cosmos_reason1 import (
    CosmosReason1TextEncoderConfig,
)

pytestmark = pytest.mark.ci_cpu

_HIDDEN = 4
_LAYERS = 2
_MAX_LENGTH = 6


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=_HIDDEN, num_hidden_layers=_LAYERS)
        self.proj = nn.Linear(1, _HIDDEN)
        self.calls = 0

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device

    def forward(self, input_ids, output_hidden_states, return_dict):
        self.calls += 1
        base = self.proj(input_ids.to(self.proj.weight.dtype).unsqueeze(-1))
        hidden = tuple(base + layer for layer in range(_LAYERS + 1))
        return SimpleNamespace(hidden_states=hidden)


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, conversations, **kwargs):
        text = conversations[-1]["content"][0]["text"]
        return torch.tensor([[len(text) % 7 + 1, 2, 3]])


@pytest.fixture
def make_encoder(monkeypatch):
    monkeypatch.setattr(
        cosmos_reason1, "maybe_download_hf_repo_on_rank0", lambda *a, **k: None
    )
    monkeypatch.setattr(
        cosmos_reason1.AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: SimpleNamespace(tokenizer=_Tokenizer())),
    )
    monkeypatch.setattr(
        cosmos_reason1.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _TinyModel().to(k["dtype"])),
    )

    def build(**overrides):
        config = CosmosReason1TextEncoderConfig(
            max_length=_MAX_LENGTH, dtype=torch.float32, **overrides
        )
        return config.setup()

    return build


def test_default_keeps_module_move_semantics(make_encoder) -> None:
    encoder = make_encoder()
    encoder.to("meta")
    assert encoder.model.proj.weight.device.type == "meta"
    assert encoder._compute_device is None


def test_run_on_cpu_records_device_and_returns_there(make_encoder) -> None:
    encoder = make_encoder(run_on_cpu=True)
    encoder.to("meta")
    assert encoder.model.proj.weight.device.type == "cpu"
    assert encoder._compute_device == torch.device("meta")

    out = encoder(["a prompt"])
    assert out.device.type == "meta"
    assert out.shape == (1, _MAX_LENGTH, _LAYERS * _HIDDEN)

    encoder.to("cpu")
    assert encoder._compute_device is None
    assert encoder(["a prompt"]).device.type == "cpu"


def test_run_on_cpu_applies_dtype_changes_on_host(make_encoder) -> None:
    encoder = make_encoder(run_on_cpu=True)
    encoder.to("meta")
    encoder.half()
    assert encoder.dtype is torch.float16
    assert encoder.model.proj.weight.dtype is torch.float16
    assert encoder.model.proj.weight.device.type == "cpu"
    assert encoder._compute_device == torch.device("meta")


def test_embedding_cache_hits_and_evicts_in_insertion_order(make_encoder) -> None:
    encoder = make_encoder(run_on_cpu=True, embedding_cache_size=2)
    first = encoder(["one"])
    again = encoder(["one"])
    assert encoder.model.calls == 1
    assert torch.equal(first, again)

    encoder(["two"])
    encoder(["one"])  # a hit, which must not reorder the cache
    encoder(["three"])  # evicts "one", the oldest entry
    assert encoder.model.calls == 3
    encoder(["two"])
    assert encoder.model.calls == 3
    encoder(["one"])
    assert encoder.model.calls == 4

    encoder.to("meta")  # a device change invalidates the cache
    assert not encoder._embedding_cache


def test_cache_disabled_by_default(make_encoder) -> None:
    encoder = make_encoder()
    encoder(["one"])
    encoder(["one"])
    assert encoder.model.calls == 2
