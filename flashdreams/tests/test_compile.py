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

"""Compiler cache and Triton bundle integrity tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch.nn as nn
from torch._inductor.runtime.cache_dir_utils import default_cache_dir

from flashdreams.infra import compile as compile_module_impl
from flashdreams.infra.compile import (
    _add_static_autotuner_hashes_to_winners,
    _configure_inductor_cache,
)

pytestmark = pytest.mark.ci_cpu


def test_configure_inductor_cache_uses_flashdreams_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

    _configure_inductor_cache()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor")


def test_configure_inductor_cache_preserves_explicit_override(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "explicit"
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path / "flashdreams"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(override))

    _configure_inductor_cache()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(override)


def test_configure_inductor_cache_replaces_pytorch_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", default_cache_dir())

    _configure_inductor_cache()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor")


def test_static_autotuner_cubins_are_retained_with_filtered_winners() -> None:
    compile_results = [
        SimpleNamespace(kernel=SimpleNamespace(hash="missing-a")),
        SimpleNamespace(kernel=SimpleNamespace(hash="missing-b")),
    ]
    bundler = SimpleNamespace(
        _winners={"existing-winner"},
        _static_autotuners=[
            SimpleNamespace(kernel=SimpleNamespace(compile_results=compile_results))
        ],
    )

    _add_static_autotuner_hashes_to_winners(
        bundler,
        to_path_key=lambda value: f"path-{value}",
    )

    assert bundler._winners == {
        "existing-winner",
        "path-missing-a",
        "path-missing-b",
    }


def test_empty_winner_set_keeps_bundle_all_behavior() -> None:
    bundler = SimpleNamespace(
        _winners=set(),
        _static_autotuners=[
            SimpleNamespace(
                kernel=SimpleNamespace(
                    compile_results=[
                        SimpleNamespace(kernel=SimpleNamespace(hash="kernel"))
                    ]
                )
            )
        ],
    )

    _add_static_autotuner_hashes_to_winners(
        bundler,
        to_path_key=lambda value: f"path-{value}",
    )

    assert bundler._winners == set()


def test_compile_module_installs_cache_repairs_before_compile(monkeypatch) -> None:
    calls: list[str] = []
    module = nn.Identity()

    monkeypatch.setattr(
        compile_module_impl,
        "_configure_inductor_cache",
        lambda: calls.append("cache"),
    )
    monkeypatch.setattr(
        compile_module_impl,
        "_patch_triton_bundle_collection",
        lambda: calls.append("bundle"),
    )
    monkeypatch.setattr(
        compile_module_impl.torch,
        "compile",
        lambda value, *, mode: calls.append(f"compile:{mode}") or value,
    )

    result = compile_module_impl.compile_module(module)

    assert result is module
    assert calls == ["cache", "bundle", "compile:max-autotune-no-cudagraphs"]
