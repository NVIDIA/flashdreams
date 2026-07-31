# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from flashdreams.infra.encoder.image import clip as clip_module
from flashdreams.infra.encoder.text import umt5 as umt5_module

pytestmark = pytest.mark.ci_cpu


def test_umt5_text_encoder_uses_low_cpu_mem_dtype_load(monkeypatch) -> None:
    kwargs_seen: dict[str, Any] = {}

    monkeypatch.setattr(
        umt5_module,
        "maybe_download_hf_repo_on_rank0",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        umt5_module.UMT5EncoderModel,
        "from_pretrained",
        lambda *args, **kwargs: kwargs_seen.update(kwargs) or _FakeModule(),
    )
    monkeypatch.setattr(
        umt5_module.T5Tokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    umt5_module.UMT5TextEncoder(
        umt5_module.UMT5TextEncoderConfig(dtype=torch.bfloat16)
    )

    assert kwargs_seen["dtype"] is torch.bfloat16
    assert kwargs_seen["low_cpu_mem_usage"] is True
    assert kwargs_seen["local_files_only"] is True


def test_clip_image_encoder_uses_low_cpu_mem_dtype_load(monkeypatch) -> None:
    kwargs_seen: dict[str, Any] = {}

    monkeypatch.setattr(
        clip_module,
        "maybe_download_hf_repo_on_rank0",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        clip_module.CLIPVisionModel,
        "from_pretrained",
        lambda *args, **kwargs: kwargs_seen.update(kwargs) or _FakeModule(),
    )
    monkeypatch.setattr(
        clip_module.CLIPImageProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    clip_module.CLIPImageEncoder(
        clip_module.CLIPImageEncoderConfig(dtype=torch.bfloat16)
    )

    assert kwargs_seen["dtype"] is torch.bfloat16
    assert kwargs_seen["low_cpu_mem_usage"] is True
    assert kwargs_seen["local_files_only"] is True


class _FakeModule(torch.nn.Module):
    pass
