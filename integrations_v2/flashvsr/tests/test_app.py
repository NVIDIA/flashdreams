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

"""CPU contract tests for the FlashVSR v2v binding."""

import os

import pytest
from flashvsr.apps.v2v.adapter import (
    create_app,
    create_app_full_attn,
    create_app_sparse_ratio_1_5,
)
from flashvsr.impl import corrector

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("factory", "model_name", "sparse_ratio", "attention_mode"),
    [
        (
            create_app,
            "flashvsr-v1.1-sparse-ratio-2.0",
            2.0,
            "sparse",
        ),
        (
            create_app_sparse_ratio_1_5,
            "flashvsr-v1.1-sparse-ratio-1.5",
            1.5,
            "sparse",
        ),
        (
            create_app_full_attn,
            "flashvsr-v1.1-full-attn",
            2.0,
            "full",
        ),
    ],
)
def test_entry_point_factories_bind_flashvsr_defaults(
    factory,
    model_name: str,
    sparse_ratio: float,
    attention_mode: str,
) -> None:
    application = factory()

    assert application.defaults.model_name == model_name
    assert application.defaults.first_chunk_size == 13
    assert application.defaults.steady_chunk_size == 16
    processor = application.defaults.processor
    assert processor.sparse_ratio == sparse_ratio
    assert processor.attention_mode == attention_mode


def test_adain_cuda_extension_uses_windows_preprocessor_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    extension = object()

    def fake_load(**kwargs: object) -> object:
        captured.update(kwargs)
        return extension

    monkeypatch.setattr(corrector, "_ADAIN_CUDA_EXTENSION", None)
    monkeypatch.setattr(corrector, "_ADAIN_CUDA_EXTENSION_LOAD_ERROR", None)
    monkeypatch.setattr(corrector, "_load_cuda_extension", fake_load)

    assert corrector._load_adain_cuda_extension() is extension
    cuda_flags = captured["extra_cuda_cflags"]
    assert isinstance(cuda_flags, list)
    assert ("-Xcompiler=/Zc:preprocessor" in cuda_flags) == (os.name == "nt")
