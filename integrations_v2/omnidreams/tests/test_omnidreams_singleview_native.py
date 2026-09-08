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

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnidreams.impl.native import (
    NativePrepError,
    NativeTensorLayout,
    NativeTensorSpec,
    prepare_tensor_for_native,
)
from omnidreams.impl.native import omnidreams_singleview as native
from omnidreams.impl.native.acceleration import (
    NativeAccelerationConfig,
    NativeAccelerationUnavailable,
    NativeSourcesUnavailable,
    require_extension_symbols,
    select_native_extension,
)


def _fake_extension_module(**attrs: object) -> ModuleType:
    module = ModuleType("test_native_extension")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def _fake_thirdparty_info(tmp_path: Path) -> dict[str, dict[str, object]]:
    cutlass = tmp_path / "cutlass"
    cudnn_frontend = tmp_path / "cudnn-frontend"
    (cutlass / "include").mkdir(parents=True)
    (cudnn_frontend / "include").mkdir(parents=True)
    return {
        "cutlass": {
            "name": "cutlass",
            "path": str(cutlass),
            "repo": "https://github.com/NVIDIA/cutlass.git",
            "commit": "cutlass-test-sha",
            "source_sha256": "cutlass-source-hash",
            "tree_sha256": "cutlass-tree-hash",
            "stamp_path": str(cutlass / ".flashdreams_source.json"),
        },
        "SageAttention": {
            "name": "SageAttention",
            "path": str(tmp_path / "SageAttention"),
            "repo": "https://github.com/thu-ml/SageAttention.git",
            "commit": "sage-test-sha",
            "source_sha256": "sage-source-hash",
            "tree_sha256": "sage-tree-hash",
            "stamp_path": str(tmp_path / "SageAttention" / ".flashdreams_source.json"),
        },
        "SpargeAttn": {
            "name": "SpargeAttn",
            "path": str(tmp_path / "SpargeAttn"),
            "repo": "https://github.com/thu-ml/SpargeAttn.git",
            "commit": "sparge-test-sha",
            "source_sha256": "sparge-source-hash",
            "tree_sha256": "sparge-tree-hash",
            "stamp_path": str(tmp_path / "SpargeAttn" / ".flashdreams_source.json"),
        },
        "cudnn-frontend": {
            "name": "cudnn-frontend",
            "path": str(cudnn_frontend),
            "repo": "https://github.com/NVIDIA/cudnn-frontend.git",
            "commit": "cudnn-frontend-test-sha",
            "source_sha256": "cudnn-frontend-source-hash",
            "tree_sha256": "cudnn-frontend-tree-hash",
            "stamp_path": str(cudnn_frontend / ".flashdreams_source.json"),
        },
    }


def _fake_source_infos(helper: Any, tmp_path: Path) -> dict[str, Any]:
    return {
        name: helper.SourceInfo(
            name=str(info["name"]),
            path=Path(str(info["path"])),
            repo=str(info["repo"]),
            commit=str(info["commit"]),
            source_sha256=str(info["source_sha256"]),
            tree_sha256=str(info["tree_sha256"]),
            stamp_path=Path(str(info["stamp_path"])),
        )
        for name, info in _fake_thirdparty_info(tmp_path).items()
    }


@pytest.mark.ci_cpu
def test_build_info_uses_script_managed_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = native._native_build()
    monkeypatch.setattr(
        helper,
        "validate_thirdparty",
        lambda: _fake_source_infos(helper, tmp_path),
    )

    build_root = tmp_path / "native-build"
    info = native.build_info(build_root=build_root)

    assert info["build_root"] == str(build_root.resolve())
    assert info["thirdparty"]["cutlass"]["commit"] == "cutlass-test-sha"
    assert info["thirdparty"]["SageAttention"]["commit"] == "sage-test-sha"
    assert info["thirdparty"]["SpargeAttn"]["commit"] == "sparge-test-sha"
    assert Path(info["cutlass_include"]).parts[-3:] == (
        "3rdparty",
        "cutlass",
        "include",
    )


@pytest.mark.ci_cpu
def test_load_extension_uses_build_root_for_torch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    build_root = tmp_path / "native-build"
    thirdparty_info = _fake_thirdparty_info(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_torch_extension(**kwargs: object) -> ModuleType:
        captured.update(kwargs)
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        captured["cuda_arch_list_env"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        return _fake_extension_module()

    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setattr(native.os, "cpu_count", lambda: 48)
    monkeypatch.setattr(native, "_python_package_dir", lambda package: None)
    monkeypatch.setattr(native, "_detected_cuda_arch_list", lambda: None)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", raising=False)

    extension = native.load_extension(build_root=build_root)

    assert extension is not None
    extension_name = captured["name"]
    assert captured["build_directory"] == str(
        build_root / "torch_extensions" / str(extension_name)
    )
    cutlass_dir = Path(str(thirdparty_info["cutlass"]["path"]))
    sage_attention_dir = Path(str(thirdparty_info["SageAttention"]["path"]))
    sparge_attn_csrc = Path(str(thirdparty_info["SpargeAttn"]["path"])) / "csrc"
    cudnn_frontend_include = (
        Path(str(thirdparty_info["cudnn-frontend"]["path"])) / "include"
    )
    assert captured["extra_include_paths"] == [
        str(native._SOURCE_DIR),
        str(native._DIT_STREAMING_DIR),
        str(native._DIT_STREAMING_KERNEL_DIR),
        str(native._DIT_STREAMING_PYEXT_DIR),
        str(native._DIT_STREAMING_COMMON_DIR),
        str(native._VAE_STREAMING_DIR),
        str(cutlass_dir / "include"),
        str(cutlass_dir / "tools" / "util" / "include"),
        str(cutlass_dir / "examples" / "common"),
        str(cutlass_dir / "examples" / "41_fused_multi_head_attention"),
        str(sage_attention_dir),
        str(sage_attention_dir / "sageattention3_blackwell" / "sageattn3"),
        str(
            sage_attention_dir / "sageattention3_blackwell" / "sageattn3" / "blackwell"
        ),
        str(
            sage_attention_dir
            / "sageattention3_blackwell"
            / "sageattn3"
            / "quantization"
        ),
        str(sparge_attn_csrc),
        str(sparge_attn_csrc / "qattn"),
        str(sparge_attn_csrc / "fused"),
        str(cudnn_frontend_include),
    ]
    sources = [Path(str(source)).name for source in captured["sources"]]
    assert sources == [
        "omnidreams_singleview_ext.cpp",
        "native_primitives.cpp",
        "native_primitives_cuda.cu",
        "streaming_dit_bindings.cpp",
        "vae_streaming_bindings.cpp",
        "lightvae_ops.cu",
        "lightvae_fp8_ops.cu",
        "lightvae_fp8_direct_stages.cu",
        "lightvae_fp8_warp_mma_stages.cu",
        "lightvae_fp8_attention.cu",
        "streaming_dit_bridge.cu",
        "attention.cu",
        "block_quant.cu",
        "cosmos_adaln_lora.cu",
        "cosmos_block.cu",
        "cosmos_fp8_flash.cu",
        "cosmos_fp8_flash_tc.cu",
        "cosmos_fp8_tc_probe.cu",
        "cosmos_fp8_two_gemm.cu",
        "cosmos_gemm_bf16.cu",
        "cosmos_modulate.cu",
        "ops.cu",
        "sage3_attention_stub.cu",
        "sparge_attention_sm89_inst.cu",
        "transformer_block.cu",
    ]
    fingerprint_sources = {
        source.relative_to(native._ROOT).as_posix()
        for source in native._extension_fingerprint_sources()
    }
    assert {
        "src/native_common/macros.h",
        "src/native_common/scalar_types.h",
        "src/native_common/tensor_ref.h",
        "src/native_common/tensor_ref_torch.h",
        "src/native_common/workspace_allocator.h",
        "src/vae_streaming/vae_streaming_bindings.h",
    }.issubset(fingerprint_sources)
    assert "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA" in captured["extra_cflags"]
    if os.name == "nt":
        assert "/Zc:preprocessor" in captured["extra_cflags"]
    assert (
        '-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA=\\"cutlass-test-sha\\"'
        in captured["extra_cflags"]
    )
    assert (
        '-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SOURCE_SHA=\\"cutlass-source-hash\\"'
        in captured["extra_cflags"]
    )
    assert any(
        flag.startswith("-DOMNIDREAMS_SINGLEVIEW_CUDA_SOURCE_SHA=")
        for flag in captured["extra_cflags"]
    )
    assert any(
        flag.startswith("-DOMNIDREAMS_SINGLEVIEW_SOURCE_FINGERPRINT_SHA=")
        for flag in captured["extra_cflags"]
    )
    assert any(
        flag.startswith("-DOMNIDREAMS_SINGLEVIEW_NATIVE_PRIMITIVES_SOURCE_SHA=")
        for flag in captured["extra_cflags"]
    )
    assert (
        '-DOMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA=\\"sage-test-sha\\"'
        in captured["extra_cflags"]
    )
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=0" in captured["extra_cflags"]
    assert (
        '-DOMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA=\\"sparge-test-sha\\"'
        in captured["extra_cflags"]
    )
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SPARGE=1" in captured["extra_cflags"]
    assert (
        '-DOMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=\\"pytorch-default\\"'
        in captured["extra_cflags"]
    )
    assert "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA" in captured["extra_cuda_cflags"]
    if os.name == "nt":
        assert "-Xcompiler=/Zc:preprocessor" in captured["extra_cuda_cflags"]
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=0" in captured["extra_cuda_cflags"]
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SPARGE=1" in captured["extra_cuda_cflags"]
    assert captured["with_cuda"] is True
    assert captured["max_jobs_env"] == "8"
    assert captured["cuda_arch_list_env"] is None
    assert "MAX_JOBS" not in os.environ
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    ("capability", "device_name", "expected"),
    [
        ((12, 0), "NVIDIA GeForce RTX 5090", "12.0a"),
        ((12, 0), "NVIDIA RTX PRO 6000 Blackwell", "12.0a"),
        ((12, 0), "Unvalidated Compute Capability 12.0 GPU", None),
        ((10, 3), "NVIDIA GB300", None),
        ((8, 9), "NVIDIA RTX 6000 Ada Generation", None),
    ],
)
def test_detected_cuda_arch_list_only_selects_validated_sm120a_devices(
    capability: tuple[int, int],
    device_name: str,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: device_name)

    assert native._detected_cuda_arch_list() == expected


@pytest.mark.ci_cpu
def test_effective_cuda_arch_list_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "_detected_cuda_arch_list", lambda: "12.0a")
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", raising=False)

    assert native._effective_cuda_arch_list() == "12.0a"

    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "10.3a")
    assert native._effective_cuda_arch_list() == "10.3a"

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
    assert native._effective_cuda_arch_list() == "8.9"


@pytest.mark.ci_cpu
def test_effective_cuda_arch_list_uses_pytorch_default_without_sm120a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "_detected_cuda_arch_list", lambda: None)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", raising=False)

    assert native._effective_cuda_arch_list() is None
    assert native._cuda_arch_identity(None) == "pytorch-default"


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("yes", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
    ],
)
def test_sage3_build_opt_out_parses_affirmative_values(
    value: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "12.0a")
    if value is None:
        monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", raising=False)
    else:
        monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", value)

    assert native._sage3_disabled() is expected
    sources = {source.name for source in native._extension_sources()}
    if expected:
        assert "sage3_attention_stub.cu" in sources
        assert "sage3_blackwell_api_shim.cu" not in sources
        assert "sage3_fp4_quant_shim.cu" not in sources
        assert "sage3_attention.cu" not in sources
    else:
        assert "sage3_attention_stub.cu" not in sources
        assert "sage3_blackwell_api_shim.cu" in sources
        assert "sage3_fp4_quant_shim.cu" in sources
        assert "sage3_attention.cu" in sources


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    ("cuda_arch_list", "expected"),
    [
        ("12.0a", False),
        ("pytorch-default", True),
        ("10.3a", True),
        ("12.0", True),
    ],
)
def test_sage3_build_requires_exact_sm120a_target(
    cuda_arch_list: str,
    expected: bool,
) -> None:
    assert native._sage3_disabled(cuda_arch_list) is expected


@pytest.mark.ci_cpu
def test_load_extension_uses_sage3_stub_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    captured: dict[str, Any] = {}

    def fake_load_torch_extension(**kwargs: object) -> ModuleType:
        captured.update(kwargs)
        return _fake_extension_module()

    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(
        native, "validate_thirdparty", lambda: _fake_thirdparty_info(tmp_path)
    )
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setattr(native, "_python_package_dir", lambda package: None)
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", "1")

    extension = native.load_extension(build_root=tmp_path / "native-build")

    assert extension is not None
    sources = {Path(str(source)).name for source in captured["sources"]}
    assert "sage3_blackwell_api_shim.cu" not in sources
    assert "sage3_fp4_quant_shim.cu" not in sources
    assert "sage3_attention.cu" not in sources
    assert "sage3_attention_stub.cu" in sources
    assert "lightvae_fp8_attention.cu" in sources
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=0" in captured["extra_cflags"]
    assert "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=0" in captured["extra_cuda_cflags"]


@pytest.mark.ci_cpu
def test_load_extension_caches_separate_sage3_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    extensions: list[ModuleType] = []

    def fake_load_torch_extension(**_: object) -> ModuleType:
        extension = _fake_extension_module()
        extensions.append(extension)
        return extension

    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setattr(native, "_python_package_dir", lambda package: None)
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "12.0a")
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", raising=False)

    sage3_extension = native.load_extension(build_root=tmp_path / "native-build")
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", "1")
    stub_extension = native.load_extension(build_root=tmp_path / "native-build")

    assert sage3_extension is extensions[0]
    assert stub_extension is extensions[1]
    assert native.load_extension(build_root=tmp_path / "native-build") is stub_extension
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", raising=False)
    assert (
        native.load_extension(build_root=tmp_path / "native-build") is sage3_extension
    )
    assert len(extensions) == 2


@pytest.mark.ci_cpu
def test_load_extension_caches_separate_cuda_architectures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    extensions: list[ModuleType] = []

    def fake_load_torch_extension(**_: object) -> ModuleType:
        extension = _fake_extension_module()
        extensions.append(extension)
        return extension

    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setattr(native, "_python_package_dir", lambda package: None)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "10.3a")
    sm103_extension = native.load_extension(build_root=tmp_path / "native-build")
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "12.0a")
    sm120_extension = native.load_extension(build_root=tmp_path / "native-build")

    assert sm103_extension is extensions[0]
    assert sm120_extension is extensions[1]
    assert (
        native.load_extension(build_root=tmp_path / "native-build") is sm120_extension
    )
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "10.3a")
    assert (
        native.load_extension(build_root=tmp_path / "native-build") is sm103_extension
    )
    assert len(extensions) == 2


@pytest.mark.ci_cpu
def test_extension_name_isolated_by_sage3_build_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_source_fingerprint", lambda: "fixed-fingerprint")
    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST", "12.0a")
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", raising=False)
    full_name = native._extension_name(thirdparty_info)

    monkeypatch.setenv("OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3", "true")
    stubbed_name = native._extension_name(thirdparty_info)

    assert stubbed_name != full_name
    assert "_sage3_1_" in full_name
    assert "_sage3_0_" in stubbed_name


@pytest.mark.ci_cpu
def test_extension_name_isolated_by_cuda_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_source_fingerprint", lambda: "fixed-fingerprint")

    sm103_name = native._extension_name(
        thirdparty_info,
        cuda_arch_list="10.3a",
    )
    sm120_name = native._extension_name(
        thirdparty_info,
        cuda_arch_list="12.0a",
    )

    assert sm103_name != sm120_name


@pytest.mark.ci_cpu
def test_load_extension_respects_existing_max_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    captured: dict[str, Any] = {}

    def fake_load_torch_extension(**kwargs: object) -> ModuleType:
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        captured["cuda_arch_list_env"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        return _fake_extension_module()

    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(
        native, "validate_thirdparty", lambda: _fake_thirdparty_info(tmp_path)
    )
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setenv("MAX_JOBS", "3")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")

    extension = native.load_extension(build_root=tmp_path / "native-build")

    assert extension is not None
    assert captured["max_jobs_env"] == "3"
    assert captured["cuda_arch_list_env"] == "8.9"
    assert os.environ["MAX_JOBS"] == "3"
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


@pytest.mark.ci_cpu
def test_load_extension_retries_after_failed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_load_torch_extension(**_: object) -> ModuleType:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first build failed")
        return _fake_extension_module()

    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_extension", {})
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr("torch.utils.cpp_extension.load", fake_load_torch_extension)

    assert native.load_extension(build_root=tmp_path / "native-build") is None
    assert attempts == 1
    assert isinstance(native.extension_load_error(), RuntimeError)

    extension = native.load_extension(build_root=tmp_path / "native-build")
    assert attempts == 2
    assert extension is not None, native.extension_load_error()
    assert native.extension_load_error() is None


@pytest.mark.ci_cpu
def test_native_build_wraps_sync_setup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = native._native_build()

    class BrokenSyncTool:
        class ThirdPartySyncError(RuntimeError):
            pass

        @staticmethod
        def load_manifest() -> tuple[object, ...]:
            raise FileNotFoundError("missing manifest")

    monkeypatch.setattr(helper, "_sync_thirdparty_module", BrokenSyncTool)

    with pytest.raises(helper.NativeBuildError, match="missing manifest"):
        helper.validate_thirdparty()


@pytest.mark.ci_cpu
def test_native_acceleration_disabled_does_not_load_extension() -> None:
    called = False

    def loader(**_: object) -> None:
        nonlocal called
        called = True
        return None

    selection = select_native_extension(
        NativeAccelerationConfig(mode="disabled"),
        component="dit",
        extension_loader=loader,
        extension_error=lambda: None,
    )

    assert selection.component == "dit"
    assert selection.mode == "disabled"
    assert selection.enabled is False
    assert "disabled" in selection.reason
    assert called is False


@pytest.mark.ci_cpu
def test_native_acceleration_auto_reports_missing_extension() -> None:
    error = RuntimeError("compile failed")

    selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="vae_decoder",
        extension_loader=lambda **_: None,
        extension_error=lambda: error,
    )

    assert selection.enabled is False
    assert selection.error is error
    assert "vae_decoder" in selection.reason
    assert "compile failed" in selection.reason
    assert "sync_thirdparty.py sync" not in selection.reason


@pytest.mark.ci_cpu
def test_native_acceleration_reports_how_to_sync_missing_sources() -> None:
    error = NativeSourcesUnavailable("third-party sources are missing")

    selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="vae_decoder",
        extension_loader=lambda **_: None,
        extension_error=lambda: error,
    )

    assert selection.error is error
    assert "sync_thirdparty.py sync" in selection.reason


@pytest.mark.ci_cpu
def test_native_acceleration_required_raises_when_extension_is_missing() -> None:
    with pytest.raises(NativeAccelerationUnavailable, match="native extension"):
        select_native_extension(
            NativeAccelerationConfig(mode="required"),
            component="vae_encoder",
            extension_loader=lambda **_: None,
            extension_error=lambda: RuntimeError("not built"),
        )


@pytest.mark.ci_cpu
def test_native_acceleration_symbol_check_gates_component_support() -> None:
    extension = _fake_extension_module(
        is_available=lambda: True, run_dit_block=object()
    )

    selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="dit",
        extension_loader=lambda **_: extension,
        extension_error=lambda: None,
        availability_check=require_extension_symbols("run_dit_block"),
    )

    assert selection.enabled is True
    assert selection.require_extension() is extension

    missing_selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="vae_decoder",
        extension_loader=lambda **_: extension,
        extension_error=lambda: None,
        availability_check=require_extension_symbols("run_vae_decoder"),
    )

    assert missing_selection.enabled is False
    assert "run_vae_decoder" in missing_selection.reason


@pytest.mark.ci_cpu
def test_native_acceleration_auto_reports_false_availability() -> None:
    extension = _fake_extension_module(is_available=lambda: False)

    selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="dit",
        extension_loader=lambda **_: extension,
        extension_error=lambda: None,
    )

    assert selection.enabled is False
    assert selection.reason == "native extension is_available returned false"


@pytest.mark.ci_cpu
def test_native_acceleration_auto_reports_check_errors() -> None:
    extension = _fake_extension_module(is_available=lambda: True)

    def failing_check(_: object) -> bool:
        raise RuntimeError("unsupported tensor layout")

    selection = select_native_extension(
        NativeAccelerationConfig(mode="auto"),
        component="dit",
        extension_loader=lambda **_: extension,
        extension_error=lambda: None,
        availability_check=failing_check,
    )

    assert selection.enabled is False
    assert isinstance(selection.error, RuntimeError)
    assert "unsupported tensor layout" in selection.reason


@pytest.mark.ci_cpu
def test_omnidreams_select_backend_forwards_build_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _fake_extension_module(
        is_available=lambda: True,
        run_vae_encoder=object(),
    )
    captured: dict[str, Any] = {}

    def fake_load_extension(**kwargs: object) -> ModuleType:
        captured.update(kwargs)
        return extension

    monkeypatch.setattr(native, "load_extension", fake_load_extension)
    monkeypatch.setattr(native, "extension_load_error", lambda: None)

    selection = native.select_backend(
        "vae_encoder",
        NativeAccelerationConfig(
            build_root=str(tmp_path),
            max_jobs=2,
            verbose_build=True,
        ),
        availability_check=require_extension_symbols("run_vae_encoder"),
    )

    assert selection.enabled is True
    assert selection.require_extension() is extension
    assert captured == {
        "build_root": str(tmp_path),
        "max_jobs": 2,
        "verbose": True,
    }


@pytest.mark.ci_cpu
def test_native_tensor_layout_parses_and_rejects_duplicates() -> None:
    layout = NativeTensorLayout.parse("B V T C H W")

    assert layout.axes == ("B", "V", "T", "C", "H", "W")
    assert layout.axis_index("C") == 3

    with pytest.raises(ValueError, match="unique"):
        NativeTensorLayout.parse("B T T C")


@pytest.mark.ci_cpu
def test_native_tensor_spec_validates_shape_dtype_and_divisibility() -> None:
    tensor = torch.empty((1, 2, 8, 16, 9, 10), dtype=torch.float16)
    spec = NativeTensorSpec(
        name="video_latent",
        layout="B V T C H W",
        shape=(1, 2, None, 16, None, None),
        dtypes=(torch.float16, torch.bfloat16),
        device_type="cpu",
        axis_divisibility=(("T", 4),),
    )

    descriptor = prepare_tensor_for_native(tensor, spec).descriptor

    assert descriptor.shape == (1, 2, 8, 16, 9, 10)
    assert descriptor.layout.axes == ("B", "V", "T", "C", "H", "W")

    with pytest.raises(NativePrepError, match="axis T size 7"):
        prepare_tensor_for_native(
            torch.empty((1, 2, 7, 16, 9, 10), dtype=torch.float16), spec
        )

    with pytest.raises(NativePrepError, match="expected dtype"):
        prepare_tensor_for_native(
            torch.empty((1, 2, 8, 16, 9, 10), dtype=torch.float32), spec
        )


@pytest.mark.ci_cpu
def test_prepare_tensor_for_native_makes_contiguous_copy() -> None:
    tensor = torch.empty((2, 3, 4), dtype=torch.float32).transpose(1, 2)
    spec = NativeTensorSpec(
        name="activation",
        layout="B T C",
        shape=(2, 4, 3),
        dtypes=(torch.float32,),
    )

    prepared = prepare_tensor_for_native(tensor, spec)

    assert prepared.copied is True
    assert prepared.tensor.is_contiguous()
    assert prepared.descriptor.stride == tuple(prepared.tensor.stride())

    already_contiguous = prepare_tensor_for_native(prepared.tensor, spec)
    assert already_contiguous.copied is False


@pytest.mark.ci_cpu
def test_optimized_dit_shape_ops_preserves_configured_dtype() -> None:
    optimized_dit = native.load_python_module("optimized_dit")
    shape_ops = optimized_dit._CosmosNetworkShapeOps(
        SimpleNamespace(patch_temporal=1, patch_spatial=2),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert next(shape_ops.parameters()).dtype == torch.bfloat16


@pytest.mark.ci_cpu
def test_cosmos_transformer_config_defaults_to_auto_native_attention() -> None:
    from omnidreams.impl.transformer import CosmosTransformerConfig

    config = CosmosTransformerConfig()

    assert config.native_dit_backend == "fp8_kvcache_cudnn"
    assert config.native_dit_attention_backend == "auto"


@pytest.mark.ci_cpu
def test_optimized_dit_default_attention_backend_uses_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimized_dit = native.load_python_module("optimized_dit")
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sage3_status",
        lambda self, device=None: (True, ""),
    )
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sparge_status",
        lambda self, device=None: (True, ""),
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            network=SimpleNamespace(
                num_blocks=28,
                num_heads=16,
                model_channels=2048,
                adaln_lora_dim=256,
                timestep_scale=0.001,
            ),
            num_views=1,
            use_cuda_graph=False,
            cuda_graph_warmup_iters=0,
        ),
        network=SimpleNamespace(),
    )
    extension = SimpleNamespace(
        optimized_dit_forward=lambda *args, **kwargs: None,
        optimized_dit_supports_block_mod_cache=lambda: True,
        optimized_dit_supports_hdmap_cache=lambda: True,
    )

    executor = optimized_dit.OptimizedDiTExecutor(transformer, extension)
    executor._resolve_runtime_attention_backend(torch.device("cuda:0"))

    assert executor._requested_attention_backend == "auto"
    assert executor._attention_backend == "cudnn"
    assert executor._sparge_hybrid_period == 0


@pytest.mark.ci_cpu
def test_optimized_dit_sparge_period_one_is_pure_sparge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimized_dit = native.load_python_module("optimized_dit")
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sage3_status",
        lambda self, device=None: (False, "Sage3 unavailable in test"),
    )
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sparge_status",
        lambda self, device=None: (True, ""),
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            network=SimpleNamespace(
                num_blocks=28,
                num_heads=16,
                model_channels=2048,
                adaln_lora_dim=256,
                timestep_scale=0.001,
            ),
            num_views=1,
            use_cuda_graph=False,
            cuda_graph_warmup_iters=0,
        ),
        network=SimpleNamespace(),
    )
    extension = SimpleNamespace(
        optimized_dit_forward=lambda *args, **kwargs: None,
        optimized_dit_supports_block_mod_cache=lambda: True,
        optimized_dit_supports_hdmap_cache=lambda: True,
    )

    executor = optimized_dit.OptimizedDiTExecutor(
        transformer,
        extension,
        dit_backend="fp8_kvcache_cudnn",
        attention_backend="sparge",
        sparge_hybrid_period=1,
    )
    executor._resolve_runtime_attention_backend(torch.device("cuda:0"))

    assert executor._attention_backend == "sparge"
    assert executor._sparge_hybrid_period == 0
    assert executor._sparge_hybrid_phase == 0
    assert executor._sparge_topk == optimized_dit._DEFAULT_SPARGE_TOPK

    auto_executor = optimized_dit.OptimizedDiTExecutor(
        transformer,
        extension,
        dit_backend="fp8_kvcache_cudnn",
        attention_backend="auto",
    )
    auto_executor._resolve_runtime_attention_backend(torch.device("cuda:0"))

    assert auto_executor._attention_backend == "cudnn"
    assert auto_executor._sparge_hybrid_period == 0
    assert auto_executor._sparge_topk == optimized_dit._DEFAULT_SPARGE_TOPK


@pytest.mark.ci_cpu
def test_optimized_dit_explicit_sparge_hybrid_sends_fp8_only_cache_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimized_dit = native.load_python_module("optimized_dit")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for fp8 runtime setup")

    monkeypatch.setattr(
        optimized_dit,
        "_make_cosmos_streaming_workspace",
        lambda **_: {"workspace": torch.empty(1, dtype=torch.uint8)},
    )
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sage3_status",
        lambda self, device=None: (True, ""),
    )
    monkeypatch.setattr(
        optimized_dit.OptimizedDiTExecutor,
        "_sparge_status",
        lambda self, device=None: (True, ""),
    )

    transformer = SimpleNamespace(
        config=SimpleNamespace(
            network=SimpleNamespace(
                num_blocks=2,
                num_heads=16,
                model_channels=2048,
                adaln_lora_dim=256,
                timestep_scale=0.001,
            ),
            num_views=1,
            use_cuda_graph=False,
            cuda_graph_warmup_iters=0,
        ),
        network=SimpleNamespace(),
    )
    extension = SimpleNamespace(
        optimized_dit_forward=lambda *args, **kwargs: None,
        optimized_dit_supports_block_mod_cache=lambda: True,
        optimized_dit_supports_hdmap_cache=lambda: True,
        sage3_quantize_cross_kv_bf16=lambda k, v: (
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
        ),
    )
    executor = optimized_dit.OptimizedDiTExecutor(
        transformer,
        extension,
        dit_backend="fp8_kvcache_cudnn",
        attention_backend="sparge",
        sparge_hybrid_period=optimized_dit._DEFAULT_SPARGE_HYBRID_PERIOD,
    )
    monkeypatch.setattr(executor, "_release_network_after_fp8_snapshot", lambda: None)

    k_cache = torch.zeros((1, 4, 16, 128), dtype=torch.bfloat16)
    runtime = executor._ensure_fp8_runtime(
        k_cross=[k_cache],
        v_cross=[k_cache],
        k_self=[k_cache],
        v_self=[k_cache],
        tokens=4,
        cache=SimpleNamespace(),
    )

    assert executor._attention_backend == "sparge"
    assert executor._sparge_hybrid_period == optimized_dit._DEFAULT_SPARGE_HYBRID_PERIOD
    assert executor._sparge_topk == optimized_dit._DEFAULT_SPARGE_HYBRID_TOPK
    assert runtime["cosmos_write_bf16_kv_cache"] is False
    assert (
        runtime["cosmos_sparge_topk_ratio"] == optimized_dit._DEFAULT_SPARGE_HYBRID_TOPK
    )


@pytest.mark.ci_cpu
@pytest.mark.skipif(
    os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_THIRDPARTY_VERIFY") != "1",
    reason="Set OMNIDREAMS_SINGLEVIEW_RUN_THIRDPARTY_VERIFY=1 to verify downloaded sources.",
)
def test_real_thirdparty_sources_verify() -> None:
    info = native.validate_thirdparty()

    assert set(info) == {"cutlass", "SageAttention", "SpargeAttn", "cudnn-frontend"}


@pytest.mark.ci_gpu
@pytest.mark.skipif(
    os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST") != "1",
    reason="Set OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 to build the native extension.",
)
def test_cuda_native_extension_builds(tmp_path: Path) -> None:
    extension = native.load_extension(build_root=tmp_path)

    assert extension is not None, native.extension_load_error()
    assert extension.is_available()
    build_info = extension.build_info()
    assert build_info["with_cuda"] is True
    expected_arch = native._cuda_arch_identity(native._effective_cuda_arch_list())
    assert build_info["cuda_arch_list"] == expected_arch
    assert hasattr(extension, "native_tensor_descriptor")
    assert hasattr(extension, "native_tensor_ref_descriptor")
    assert hasattr(extension, "workspace_allocation_plan")
    assert hasattr(extension, "prepare_contiguous")
    assert hasattr(extension, "zero_workspace_")
    assert hasattr(extension, "omnidreams_vae_backend_status")
    assert hasattr(extension, "omnidreams_vae_create_wan_encoder_fp8")
    assert hasattr(extension, "omnidreams_vae_reset_wan_encoder_fp8")
    assert hasattr(extension, "omnidreams_vae_encode_wan_fp8")
    assert hasattr(extension, "lightvae_fp8_prepare_conv2d_weight_krsc")
    vae_fp8_status = extension.omnidreams_vae_backend_status("vae_encoder", "fp8")
    assert vae_fp8_status["available"] is True

    if not torch.cuda.is_available():
        return

    workspace = torch.empty((16,), device="cuda", dtype=torch.float16)
    extension.zero_workspace_(workspace)
    torch.cuda.synchronize()
    assert torch.count_nonzero(workspace).item() == 0

    source = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    transposed = source.transpose(1, 2)
    prepared = extension.prepare_contiguous(transposed)
    torch.cuda.synchronize()

    assert prepared.is_contiguous()
    assert torch.equal(prepared.cpu(), transposed.cpu())

    descriptor = extension.native_tensor_ref_descriptor(transposed)
    assert descriptor["rank"] == 3
    assert tuple(descriptor["shape"]) == tuple(transposed.shape)
    assert tuple(descriptor["stride"]) == tuple(transposed.stride())
    assert descriptor["nbytes"] == transposed.numel() * transposed.element_size()

    byte_workspace = torch.empty((128,), device="cuda", dtype=torch.uint8)
    plan = extension.workspace_allocation_plan(byte_workspace, [13, 16, 32], 16)
    assert list(plan["offsets"]) == [0, 16, 32]
    assert list(plan["sizes"]) == [13, 16, 32]
    assert plan["used_bytes"] == 64
    assert plan["remaining_bytes"] == 64
    assert plan["total_bytes"] == 128
