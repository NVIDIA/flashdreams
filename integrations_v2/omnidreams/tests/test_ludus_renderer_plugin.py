# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


@pytest.mark.skipif(os.name != "nt", reason="Windows-only DLL search path behavior")
def test_windows_dll_directory_handles_are_kept_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ludus_renderer._ops import _plugin

    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    torch_init = tmp_path / "torch" / "__init__.py"
    torch_init.write_text("", encoding="utf-8")
    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    (cuda_bin / "cudart64_12.dll").write_text("", encoding="utf-8")
    handles: list[object] = []
    added: list[str] = []

    def fake_add_dll_directory(path: str) -> object:
        handle = object()
        handles.append(handle)
        added.append(str(Path(path)))
        return handle

    monkeypatch.setattr(_plugin.os, "name", "nt")
    monkeypatch.setattr(_plugin.os, "add_dll_directory", fake_add_dll_directory)
    monkeypatch.setattr(_plugin.torch, "__file__", str(torch_init))
    monkeypatch.setattr(_plugin, "_dll_directory_handles", [])
    monkeypatch.setattr(_plugin, "_dll_directory_paths", set())
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "cuda"))
    monkeypatch.setenv("PATH", str(cuda_bin))

    _plugin._ensure_windows_dll_directories()
    _plugin._ensure_windows_dll_directories()

    assert added == [str(torch_lib), str(cuda_bin)]
    assert _plugin._dll_directory_handles == handles
