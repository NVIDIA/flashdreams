# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the SANA-WM Cam2V binding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sana_wm.impl.conditioning as conditioning_module
import sana_wm.impl.transformer as transformer_module
import tomli as tomllib
import torch
from cam2v import Cam2VApplication, CameraControlInput, CameraPoseIntegrator
from PIL import Image
from sana_wm.apps.cam2v import adapter
from sana_wm.apps.cam2v.adapter import SanaWMCam2VApplication, create_app
from sana_wm.config import PIPELINE_SANA_WM_STREAMING
from sana_wm.impl.conditioning import resolve_sana_wm_conditioning
from sana_wm.impl.controls import SanaWMCameraPoseIntegrator
from sana_wm.impl.decoder import SanaWMDecodedVideo
from sana_wm.impl.transformer import SanaWMStreamingTransformerConfig

from flashdreams.infra.pipeline import StreamInferencePipeline

pytestmark = pytest.mark.ci_cpu

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_package_registers_the_shared_cam2v_application() -> None:
    """Keep the V2 application entry point with the Sana model package."""
    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text())

    assert "flashdreams-cam2v" in manifest["project"]["dependencies"]
    assert "imageio[ffmpeg]>=2.31" not in manifest["project"]["dependencies"]
    entry_points = manifest["project"]["entry-points"]
    assert entry_points == {
        "flashdreams.applications_v2": {
            "cam2v-sana-wm-streaming": "sana_wm.apps.cam2v.adapter:create_app"
        }
    }
    assert (_PACKAGE_ROOT / "config.py").is_file()
    assert (_PACKAGE_ROOT / "impl").is_dir()
    assert (_PACKAGE_ROOT / "apps" / "cam2v" / "adapter.py").is_file()


def test_application_passes_sana_owned_adapters_to_cam2v() -> None:
    """Keep model-specific conditioning and controls outside shared Cam2V."""
    application = SanaWMCam2VApplication()

    assert isinstance(application, Cam2VApplication)
    assert application.pipeline_config.name == PIPELINE_SANA_WM_STREAMING.name
    assert application.pipeline_config._target is adapter.SanaWMCam2VPipeline
    assert application.defaults.input_resolver is resolve_sana_wm_conditioning
    assert application.defaults.generate_step is adapter.generate_sana_wm_step
    assert application.defaults.pose_integrator_factory is SanaWMCameraPoseIntegrator
    assert isinstance(
        application.defaults.pose_integrator_factory(), CameraPoseIntegrator
    )
    assert application.defaults.total_blocks == 10
    assert application.defaults.input_defaults["example_data"] is False
    assert application.defaults.input_defaults["example_idx"] == 0
    assert application.session_desc().video_width == 1280
    assert application.session_desc().video_height == 704
    assert application.session_desc().frames_per_second_for_step == 16
    assert isinstance(create_app(), SanaWMCam2VApplication)


@pytest.mark.parametrize(
    ("flag", "enabled"), [("--compile", True), ("--no-compile", False)]
)
def test_application_applies_compile_flag(flag: str, enabled: bool) -> None:
    """Keep the shared compile override valid for the SANA-WM config."""
    application = SanaWMCam2VApplication()

    application.init([flag])

    transformer = application.pipeline_config.diffusion_model.transformer
    assert isinstance(transformer, SanaWMStreamingTransformerConfig)
    assert transformer.compile_network is enabled


def test_streaming_transformer_compiles_lazy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile the loaded Stage-1 module when the shared flag is enabled."""
    raw_model = torch.nn.Linear(1, 1)
    compiled_model = torch.nn.Identity()
    monkeypatch.setattr(
        transformer_module, "SanaWMStage1Model", lambda _spec: raw_model
    )
    monkeypatch.setattr(
        transformer_module, "load_checkpoint", lambda _path: raw_model.state_dict()
    )
    monkeypatch.setattr(
        transformer_module,
        "compile_module",
        lambda model: compiled_model if model is raw_model else model,
    )
    transformer = SanaWMStreamingTransformerConfig(compile_network=True).setup()
    monkeypatch.setattr(
        transformer,
        "_ensure_runtime_config",
        lambda: SimpleNamespace(model=SimpleNamespace(model="dummy")),
    )
    transformer.weight_dtype = torch.float32

    transformer._ensure_model()

    assert transformer.model is compiled_model


def test_resolver_center_crops_intrinsics_with_the_first_frame(tmp_path: Path) -> None:
    """Keep first-frame geometry and camera calibration on the same crop."""
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1000, 1000)).save(image_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("  a bright   courtyard  \n", encoding="utf-8")
    intrinsic_path = tmp_path / "intrinsics.npy"
    np.save(intrinsic_path, np.array([500.0, 500.0, 500.0, 500.0]))

    result = resolve_sana_wm_conditioning(
        {
            "prompt": "",
            "prompt_path": prompt_path,
            "image_path": image_path,
            "intrinsic_path": intrinsic_path,
            "world_scale": None,
            "pixel_height": 704,
            "pixel_width": 1280,
        }
    )

    assert result.prompt == "a bright   courtyard"
    assert result.first_frame_path == image_path
    assert result.base_intrinsics.shape == (1, 4)
    assert result.base_intrinsics[0, 0].item() == pytest.approx(640.0)
    assert result.base_intrinsics[0, 2].item() == pytest.approx(640.0)
    assert result.base_intrinsics[0, 3].item() == pytest.approx(352.0)
    assert result.world_scale == 1.0


def test_example_data_supplies_the_official_image_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "demo_0.png"
    Image.new("RGB", (1280, 704)).save(image_path)
    prompt_path = tmp_path / "demo_0.txt"
    prompt_path.write_text("official SANA-WM prompt\n", encoding="utf-8")
    monkeypatch.setattr(
        conditioning_module, "_ensure_example_data", lambda _index: tmp_path
    )

    result = resolve_sana_wm_conditioning({"example_data": True, "example_idx": 0})

    assert result.first_frame_path == image_path
    assert result.prompt == "official SANA-WM prompt"


def test_pose_integrator_uses_sana_control_speed_and_aliases() -> None:
    """Map shared keyboard aliases through Sana's trained control semantics."""
    integrator = SanaWMCameraPoseIntegrator()
    forward = integrator.integrate_chunk(
        segments=[(0.0, 1.0 / 16.0, frozenset({"w"}))],
        frame_times=[1.0 / 16.0],
    )
    integrator.reset()
    strafe = integrator.integrate_chunk(
        segments=[(0.0, 1.0 / 16.0, frozenset({"q"}))],
        frame_times=[1.0 / 16.0],
    )
    integrator.reset()
    yaw = integrator.integrate_chunk(
        segments=[(0.0, 1.0 / 16.0, frozenset({"j"}))],
        frame_times=[1.0 / 16.0],
    )

    assert forward[0, 2, 3] == pytest.approx(0.025, abs=1e-6)
    assert strafe[0, 0, 3] == pytest.approx(-0.025, abs=1e-6)
    assert yaw[0, 0, 2] == pytest.approx(-np.sin(np.deg2rad(0.6)), abs=1e-6)
    assert integrator.current_pose().tolist() == yaw[0].tolist()


class _FakePipeline(adapter.SanaWMCam2VPipeline):
    requests: list[Any]
    initialize_kwargs: dict[str, Any]

    def __init__(self) -> None:
        object.__setattr__(self, "config", PIPELINE_SANA_WM_STREAMING)
        object.__setattr__(self, "requests", [])
        object.__setattr__(self, "initialize_kwargs", {})

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(self, autoregressive_index: int, cache: Any, input: Any) -> Any:
        del autoregressive_index, cache
        self.requests.append(input)
        return SanaWMDecodedVideo(video_hwc=np.full((24, 2, 3, 3), 255, dtype=np.uint8))


def _camera_input(start: float) -> CameraControlInput:
    poses = torch.eye(4).repeat(24, 1, 1)
    poses[:, 0, 3] = torch.arange(24) * 0.025 + start
    return CameraControlInput(
        intrinsics=torch.tensor([640.0, 640.0, 640.0, 352.0]).repeat(24, 1),
        poses=poses,
        world_scale=1.0,
    )


def test_generate_step_passes_accumulated_history_to_sana_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grow camera history while preserving Sana's one-time static inputs."""
    pipeline = _FakePipeline()
    monkeypatch.setattr(adapter, "SANA_WM_CAM2V_DEFAULTS", SimpleNamespace(fps=30))
    monkeypatch.setattr(
        StreamInferencePipeline,
        "initialize_cache",
        lambda _self, **kwargs: (
            pipeline.initialize_kwargs.update(kwargs) or SimpleNamespace()
        ),
    )
    image = torch.empty(1, 3, 704, 1280)
    cache = pipeline.initialize_cache(text=["a road"], image=image)

    first = adapter.generate_sana_wm_step(pipeline, 0, cache, _camera_input(0.025))
    second = adapter.generate_sana_wm_step(pipeline, 1, cache, _camera_input(0.625))

    assert pipeline.initialize_kwargs == {
        "decoder_context": {"prompt": "a road", "fps": 30}
    }
    assert [request.num_frames for request in pipeline.requests] == [25, 49]
    assert [request.fps for request in pipeline.requests] == [30, 30]
    assert pipeline.requests[0].image is image
    assert pipeline.requests[0].poses_c2w[0].tolist() == np.eye(4).tolist()
    assert pipeline.requests[1].poses_c2w[-1, 0, 3] == pytest.approx(1.2)
    assert first.shape == second.shape == (24, 3, 2, 3)
    assert torch.all(first == 1.0)


def test_application_rejects_nonpositive_world_scale(tmp_path: Path) -> None:
    """Reject camera scaling that would make the controls remap undefined."""
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 704)).save(image_path)

    with pytest.raises(ValueError, match="world_scale must be > 0"):
        resolve_sana_wm_conditioning(
            {
                "prompt": "scene",
                "image_path": image_path,
                "world_scale": 0.0,
            }
        )
