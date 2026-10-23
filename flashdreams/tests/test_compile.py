# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler cache and Triton bundle integrity tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch._dynamo import config as dynamo_config
from torch._inductor.runtime.cache_dir_utils import default_cache_dir

from flashdreams.infra import compile as compile_module_impl
from flashdreams.infra.compile import (
    _add_static_autotuner_hashes_to_winners,
    _configure_dynamo_cache,
    _configure_inductor_cache,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ((None, None), (256, 8192)),
        (("128", "4096"), (128, 4096)),
        (("512", "16384"), (512, 16384)),
        (("128", None), (128, 8192)),
        ((None, "4096"), (256, 4096)),
        (("0", "0"), (0, 0)),
    ],
)
def test_configure_dynamo_cache_environment_overrides(
    monkeypatch, overrides, expected
) -> None:
    for name, value in zip(
        ("FLASHDREAMS_RECOMPILE_LIMIT", "FLASHDREAMS_ACCUMULATED_RECOMPILE_LIMIT"),
        overrides,
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with dynamo_config.patch(recompile_limit=256, accumulated_recompile_limit=8192):
        _configure_dynamo_cache()
        assert dynamo_config.recompile_limit == expected[0]
        assert dynamo_config.accumulated_recompile_limit == expected[1]
        assert dynamo_config.cache_size_limit == expected[0]
        assert dynamo_config.accumulated_cache_size_limit == expected[1]


@pytest.mark.parametrize(
    "name", ["FLASHDREAMS_RECOMPILE_LIMIT", "FLASHDREAMS_ACCUMULATED_RECOMPILE_LIMIT"]
)
@pytest.mark.parametrize("value", ["", "invalid", "-1"])
def test_configure_dynamo_cache_rejects_invalid_limits(monkeypatch, name, value):
    monkeypatch.delenv("FLASHDREAMS_RECOMPILE_LIMIT", raising=False)
    monkeypatch.delenv("FLASHDREAMS_ACCUMULATED_RECOMPILE_LIMIT", raising=False)
    monkeypatch.setenv(name, value)
    with dynamo_config.patch(recompile_limit=8, accumulated_recompile_limit=256):
        with pytest.raises(ValueError, match=name):
            _configure_dynamo_cache()


def test_compile_module_reuses_more_than_eight_variants(monkeypatch) -> None:
    monkeypatch.setenv("FLASHDREAMS_RECOMPILE_LIMIT", "128")
    monkeypatch.setenv("FLASHDREAMS_ACCUMULATED_RECOMPILE_LIMIT", "4096")

    class VariantModule(nn.Module):
        def forward(self, x, scale):
            return x * scale + 1

    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    real_compile = torch.compile
    monkeypatch.setattr(
        compile_module_impl.torch,
        "compile",
        lambda module, *, mode, dynamic: real_compile(
            module, backend=backend, dynamic=dynamic, fullgraph=True
        ),
    )
    monkeypatch.setattr(compile_module_impl, "_configure_inductor_cache", lambda: None)
    monkeypatch.setattr(
        compile_module_impl, "_patch_triton_bundle_collection", lambda: None
    )
    torch.compiler.reset()
    try:
        with dynamo_config.patch(
            recompile_limit=8,
            accumulated_recompile_limit=256,
            fail_on_recompile_limit_hit=True,
        ):
            model = compile_module_impl.compile_module(VariantModule(), dynamic=False)
            x = torch.ones(3)
            for scale in range(12):
                torch.testing.assert_close(model(x, scale), x * scale + 1)
            assert len(compiled_graphs) == 12
            for scale in reversed(range(12)):
                torch.testing.assert_close(model(x, scale), x * scale + 1)
            assert len(compiled_graphs) == 12
    finally:
        torch.compiler.reset()


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
        "_configure_dynamo_cache",
        lambda: calls.append("dynamo"),
    )
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
        lambda value, *, mode, dynamic: calls.append(f"compile:{mode}:{dynamic}")
        or value,
    )

    result = compile_module_impl.compile_module(module)

    assert result is module
    assert calls == [
        "dynamo",
        "cache",
        "bundle",
        "compile:max-autotune-no-cudagraphs:None",
    ]
