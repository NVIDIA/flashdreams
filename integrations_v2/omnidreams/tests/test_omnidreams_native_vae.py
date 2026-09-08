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

from pathlib import Path
from types import ModuleType

import pytest
import torch
from omnidreams.config import OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
from omnidreams.impl.native.acceleration import (
    NativeAccelerationMode,
    NativeAccelerationUnavailable,
    NativeBackendSelection,
)
from omnidreams.impl.vae_native import (
    DEFAULT_LIGHTVAE_FP8_STATE_PATH,
    NATIVE_LIGHTVAE_FP8_STATE_ENV,
    OmnidreamsWanVAEEncoder,
    OmnidreamsWanVAEEncoderConfig,
    _native_vae_availability_check,
    _native_vae_fp8_state_path,
    _native_vae_preflight_reason,
    _NativeWanVAEEncoderExecutor,
)

from flashdreams.recipes.wan.autoencoder.vae import WanVAECache


def _fake_extension_module(**attrs: object) -> ModuleType:
    module = ModuleType("test_native_vae_extension")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def _enabled_selection(
    extension: ModuleType,
    *,
    mode: NativeAccelerationMode = "auto",
    component: str = "vae_encoder",
) -> NativeBackendSelection:
    return NativeBackendSelection(
        component=component,
        mode=mode,
        enabled=True,
        reason="native VAE available in test",
        extension=extension,
    )


@pytest.mark.ci_cpu
def test_omnidreams_vae_configs_default_native_disabled() -> None:
    encoder = OmnidreamsWanVAEEncoderConfig()

    assert encoder._target is OmnidreamsWanVAEEncoder
    assert encoder.native_vae_acceleration == "disabled"
    assert encoder.native_vae_backend == "fp8"
    assert encoder.native_vae_fp8_state_path is None
    assert encoder.native_vae_fp8_auto_export is False


@pytest.mark.ci_cpu
def test_native_vae_availability_uses_status_mapping() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_create_wan_encoder_fp8=object(),
        omnidreams_vae_reset_wan_encoder_fp8=object(),
        omnidreams_vae_encode_wan_fp8=object(),
        omnidreams_vae_backend_status=lambda component, backend: {
            "component": component,
            "backend": backend,
            "available": False,
            "reason": "pending kernel port",
        },
    )

    check = _native_vae_availability_check(
        component="vae_encoder",
        backend="fp8",
        run_symbol="omnidreams_vae_encode_wan_fp8",
        setup_symbols=(
            "omnidreams_vae_create_wan_encoder_fp8",
            "omnidreams_vae_reset_wan_encoder_fp8",
        ),
    )

    assert check(extension) == (False, "pending kernel port")


@pytest.mark.ci_cpu
def test_native_vae_availability_reports_missing_symbols() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_backend_status=lambda component, backend: {
            "component": component,
            "backend": backend,
            "available": True,
        },
    )
    check = _native_vae_availability_check(
        component="vae_encoder",
        backend="fp8",
        run_symbol="omnidreams_vae_encode_wan_fp8",
    )

    ok, reason = check(extension)

    assert ok is False
    assert "omnidreams_vae_encode_wan_fp8" in reason


@pytest.mark.ci_cpu
def test_native_vae_fp8_availability_uses_status_mapping() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_create_wan_encoder_fp8=object(),
        omnidreams_vae_reset_wan_encoder_fp8=object(),
        omnidreams_vae_encode_wan_fp8=object(),
        omnidreams_vae_backend_status=lambda component, backend: {
            "component": component,
            "backend": backend,
            "available": True,
            "reason": "fp8 ready",
        },
    )

    check = _native_vae_availability_check(
        component="vae_encoder",
        backend="fp8",
        run_symbol="omnidreams_vae_encode_wan_fp8",
        setup_symbols=(
            "omnidreams_vae_create_wan_encoder_fp8",
            "omnidreams_vae_reset_wan_encoder_fp8",
        ),
    )

    assert check(extension) == (True, "fp8 ready")


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_fp8_requires_state_path() -> None:
    assert (
        _NativeWanVAEEncoderExecutor.from_config(
            OmnidreamsWanVAEEncoderConfig(
                native_vae_acceleration="auto",
                native_vae_backend="fp8",
            )
        )
        is None
    )

    with pytest.raises(
        NativeAccelerationUnavailable,
        match="requires native_vae_fp8_state_path",
    ):
        _NativeWanVAEEncoderExecutor.from_config(
            OmnidreamsWanVAEEncoderConfig(
                native_vae_acceleration="required",
                native_vae_backend="fp8",
            )
        )


@pytest.mark.ci_cpu
def test_fast_perf_config_auto_exports_default_fp8_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from omnidreams.impl.scripts import export_lightvae_fp8_state as exporter

    monkeypatch.delenv(NATIVE_LIGHTVAE_FP8_STATE_ENV, raising=False)
    expected = tmp_path / DEFAULT_LIGHTVAE_FP8_STATE_PATH
    calls: list[Path] = []

    def ensure(path: Path) -> Path:
        calls.append(path)
        return expected

    monkeypatch.setattr(exporter, "ensure_lightvae_fp8_state", ensure)
    config = OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG
    image_encoder = config.image_encoder
    encoder = config.encoder
    assert isinstance(image_encoder, OmnidreamsWanVAEEncoderConfig)
    assert isinstance(encoder, OmnidreamsWanVAEEncoderConfig)

    assert _native_vae_fp8_state_path(image_encoder) == str(expected)
    assert _native_vae_fp8_state_path(encoder) == str(expected)
    assert calls == [
        Path(DEFAULT_LIGHTVAE_FP8_STATE_PATH),
        Path(DEFAULT_LIGHTVAE_FP8_STATE_PATH),
    ]


@pytest.mark.ci_cpu
def test_lightvae_fp8_state_export_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from omnidreams.impl.scripts import export_lightvae_fp8_state as exporter

    cached = tmp_path / "lightvae_fp8_state.pt"
    cached.touch()
    monkeypatch.setattr(
        exporter,
        "_ensure_hf_calibration_video",
        lambda: pytest.fail("cached state must not download calibration data"),
    )

    assert exporter.ensure_lightvae_fp8_state(cached) == cached.resolve()


@pytest.mark.ci_cpu
def test_lightvae_fp8_state_export_populates_missing_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from omnidreams.impl.scripts import export_lightvae_fp8_state as exporter

    target = tmp_path / "artifacts" / "native_vae" / "lightvae_fp8_state.pt"
    video = tmp_path / "calibration.mp4"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(exporter, "_ensure_hf_calibration_video", lambda: video)

    def export(out: Path, *, calibration_video: Path) -> Path:
        calls.append((out, calibration_video))
        out.parent.mkdir(parents=True)
        out.touch()
        return out

    monkeypatch.setattr(exporter, "export_lightvae_fp8_state", export)

    assert exporter.ensure_lightvae_fp8_state(target) == target.resolve()
    assert calls == [(target.resolve(), video)]


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_fp8_state_path_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmnidreamsWanVAEEncoderConfig(
        native_vae_acceleration="auto",
        native_vae_backend="fp8",
    )

    assert (
        _native_vae_preflight_reason(component="vae_encoder", config=config) is not None
    )

    monkeypatch.setenv(NATIVE_LIGHTVAE_FP8_STATE_ENV, "state.pt")

    assert _native_vae_fp8_state_path(config) == "state.pt"
    assert _native_vae_preflight_reason(component="vae_encoder", config=config) is None


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_executor_prepares_contiguous_fp8_boundary() -> None:
    captured: dict[str, object] = {}

    def encode_wan(
        native_encoder: object,
        tensor: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        captured["native_encoder"] = native_encoder
        captured["shape"] = tuple(tensor.shape)
        captured["is_contiguous"] = tensor.is_contiguous()
        captured["dtype"] = tensor.dtype
        captured["use_cache"] = use_cache
        return tensor

    extension = _fake_extension_module(
        omnidreams_vae_create_wan_encoder_fp8=lambda _state: "native_encoder",
        omnidreams_vae_reset_wan_encoder_fp8=lambda _encoder: None,
        omnidreams_vae_encode_wan_fp8=encode_wan,
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension),
        backend="fp8",
    )
    executor._fp8_state_path = "state.pt"
    executor._helper = _fake_extension_module(
        load_lightvae_fp8_state=lambda _path: {"input.activation_scale": torch.ones(1)},
        build_lightvae_encoder_fp8_staged_state=lambda _model, _state, _ext: {
            "scale_input": torch.empty(())
        },
    )
    source = torch.empty((1, 3, 1, 8, 8), dtype=torch.float16).transpose(3, 4)

    result = executor.try_encode(object(), source, WanVAECache())

    assert result is not None
    assert captured == {
        "native_encoder": "native_encoder",
        "shape": (1, 3, 1, 8, 8),
        "is_contiguous": True,
        "dtype": torch.float16,
        "use_cache": True,
    }


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_executor_builds_fp8_state() -> None:
    captured: dict[str, object] = {}

    def build_state(
        model: object,
        fp8_state: object,
        extension: ModuleType,
    ) -> dict[str, object]:
        captured["model"] = model
        captured["fp8_state"] = fp8_state
        captured["extension"] = extension
        return {"scale_input": torch.empty(())}

    def encode_wan(
        native_encoder: object,
        tensor: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        captured["native_encoder"] = native_encoder
        captured["shape"] = tuple(tensor.shape)
        captured["use_cache"] = use_cache
        return tensor

    extension = _fake_extension_module(
        omnidreams_vae_create_wan_encoder_fp8=lambda state: ("fp8_encoder", state),
        omnidreams_vae_reset_wan_encoder_fp8=lambda _encoder: None,
        omnidreams_vae_encode_wan_fp8=encode_wan,
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension),
        backend="fp8",
    )
    executor._fp8_state_path = "state.pt"
    fp8_state = {"input.activation_scale": torch.ones(1)}
    model = object()
    executor._helper = _fake_extension_module(
        load_lightvae_fp8_state=lambda _path: fp8_state,
        build_lightvae_encoder_fp8_staged_state=build_state,
    )
    source = torch.empty((1, 3, 1, 8, 8), dtype=torch.float16)

    result = executor.try_encode(model, source, WanVAECache())

    assert result is not None
    assert captured["model"] is model
    assert captured["fp8_state"] is fp8_state
    assert captured["extension"] is extension
    native_encoder = captured["native_encoder"]
    assert isinstance(native_encoder, tuple)
    assert native_encoder[0] == "fp8_encoder"
    assert captured["shape"] == (1, 3, 1, 8, 8)
    assert captured["use_cache"] is True


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_fp8_rejects_batch_gt_one() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_encode_wan_fp8=lambda *_args: pytest.fail(
            "native dispatch should not run for unsupported fp8 batch size"
        )
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension, mode="required"),
        backend="fp8",
    )
    source = torch.empty((2, 3, 1, 8, 8), dtype=torch.float16)

    with pytest.raises(NativeAccelerationUnavailable, match="axis B"):
        executor.try_encode(object(), source, WanVAECache())


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_executor_matches_streaming_chunking() -> None:
    calls: list[tuple[int, ...]] = []

    def encode_wan(
        _native_encoder: object,
        tensor: torch.Tensor,
        _use_cache: bool,
    ) -> torch.Tensor:
        calls.append(tuple(tensor.shape))
        return torch.empty(
            (tensor.shape[0], 16, 1, tensor.shape[3] // 8, tensor.shape[4] // 8),
            dtype=tensor.dtype,
        )

    extension = _fake_extension_module(
        omnidreams_vae_create_wan_encoder_fp8=lambda _state: "native_encoder",
        omnidreams_vae_reset_wan_encoder_fp8=lambda _encoder: None,
        omnidreams_vae_encode_wan_fp8=encode_wan,
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension),
        backend="fp8",
    )
    executor._fp8_state_path = "state.pt"
    executor._helper = _fake_extension_module(
        load_lightvae_fp8_state=lambda _path: {"input.activation_scale": torch.ones(1)},
        build_lightvae_encoder_fp8_staged_state=lambda _model, _state, _ext: {
            "scale_input": torch.empty(())
        },
    )
    source = torch.empty((1, 3, 5, 16, 16), dtype=torch.float16)

    result = executor.try_encode(object(), source, WanVAECache())

    assert result is not None
    assert tuple(result.shape) == (1, 16, 2, 2, 2)
    assert calls == [(1, 3, 1, 16, 16), (1, 3, 4, 16, 16)]


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_executor_auto_fallback_on_bad_dtype() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_encode_wan_fp8=lambda *_args: pytest.fail(
            "native dispatch should not run for bf16 input"
        )
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension),
        backend="fp8",
    )
    source = torch.empty((1, 3, 1, 8, 8), dtype=torch.bfloat16)

    assert executor.try_encode(object(), source, WanVAECache()) is None


@pytest.mark.ci_cpu
def test_native_wan_vae_encoder_executor_required_raises_on_bad_dtype() -> None:
    extension = _fake_extension_module(
        omnidreams_vae_encode_wan_fp8=lambda *_args: pytest.fail(
            "native dispatch should not run for bf16 input"
        )
    )
    executor = _NativeWanVAEEncoderExecutor(
        selection=_enabled_selection(extension, mode="required"),
        backend="fp8",
    )
    source = torch.empty((1, 3, 1, 8, 8), dtype=torch.bfloat16)

    with pytest.raises(NativeAccelerationUnavailable, match="expected dtype"):
        executor.try_encode(object(), source, WanVAECache())


@pytest.mark.ci_cpu
def test_native_vae_path_does_not_keep_stale_extension_names() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    paths = [
        repo_root / "integrations_v2" / "omnidreams" / "impl" / "vae_native.py",
        repo_root
        / "integrations_v2"
        / "omnidreams"
        / "impl"
        / "omnidreams_singleview"
        / "src"
        / "vae_streaming"
        / "vae_streaming_bindings.cpp",
        repo_root
        / "integrations_v2"
        / "omnidreams"
        / "impl"
        / "omnidreams_singleview"
        / "src"
        / "vae_streaming"
        / "vae_streaming_bindings.h",
        repo_root
        / "integrations_v2"
        / "omnidreams"
        / "impl"
        / "omnidreams_singleview"
        / "python"
        / "vae_weights.py",
        *(
            repo_root
            / "integrations_v2"
            / "omnidreams"
            / "impl"
            / "omnidreams_singleview"
            / "src"
            / "vae_streaming"
        ).glob("*.cu"),
        *(
            repo_root
            / "integrations_v2"
            / "omnidreams"
            / "impl"
            / "omnidreams_singleview"
            / "src"
            / "vae_streaming"
        ).glob("*.h"),
    ]
    stale_project_name = "true" + "sight"
    stale_raw_abi = stale_project_name + "_fp8_latent_v1"
    stale_legacy_extension = "tin" + "_ext"
    stale_sister_project = "alpa" + "dreams"
    stale_cute_stage_name = "lightvae_fp8_" + "cute" + "_stages"
    banned = (
        stale_legacy_extension,
        stale_sister_project,
        stale_project_name,
        stale_project_name.upper(),
        stale_raw_abi,
        stale_cute_stage_name,
        "cutlass_" + "cute",
        "_cute_",
        "cute_",
        "Cu" + "Te",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token!r} should not appear in {path}"
