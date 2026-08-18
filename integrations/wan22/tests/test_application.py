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

"""CPU application checks for the Wan 2.2 TI2V demo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import tomli as tomllib
import torch
from wan22.config import WAN22_TI2V_5B_DIT_DIFFUSERS_PATH
from wan22.ti2v import app as app_module
from wan22.ti2v.app import Wan22TI2VApplication, create_app

from flashdreams.demo import CanonicalInputWindow, IFlashDreamsApplication
from flashdreams.infra.time import TimeWindow

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.initialize_calls: list[dict[str, Any]] = []
        self.generated: list[int] = []
        self.finalized: list[int] = []
        self.closed = False

    def to(self, device: str) -> "_FakePipeline":
        self.device = torch.device(device)
        return self

    def eval(self) -> "_FakePipeline":
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.initialize_calls.append(kwargs)
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        assert autoregressive_index == 0
        return 81

    def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        return torch.zeros((81, 3, 4, 4))

    def finalize(self, *, autoregressive_index: int, cache: object) -> None:
        del cache
        self.finalized.append(autoregressive_index)

    def close(self) -> None:
        self.closed = True


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _FakePipeline:
        return self.pipeline


def _first_frame(tmp_path: Path) -> Path:
    path = tmp_path / "first-frame.png"
    path.write_bytes(b"test image placeholder")
    return path


def test_factory_exposes_static_ti2v_application_defaults() -> None:
    application = create_app()

    assert isinstance(application, IFlashDreamsApplication)
    assert isinstance(application, Wan22TI2VApplication)
    assert application.defaults.total_blocks == 1
    assert application.defaults.pixel_height == 640
    assert application.defaults.pixel_width == 1280
    assert application.defaults.fps == 16
    assert not application.input_schema.modalities


def test_transformer_uses_the_published_sharded_checkpoint_index() -> None:
    application = Wan22TI2VApplication()
    transformer = application.defaults.pipeline_config.diffusion_model.transformer

    assert transformer.checkpoint_path == WAN22_TI2V_5B_DIT_DIFFUSERS_PATH
    assert WAN22_TI2V_5B_DIT_DIFFUSERS_PATH.endswith(
        "transformer/diffusion_pytorch_model.safetensors.index.json"
    )


def test_application_requires_prompt_and_existing_first_frame(tmp_path: Path) -> None:
    application = Wan22TI2VApplication()
    first_frame = _first_frame(tmp_path)

    with pytest.raises(ValueError, match="--prompt is required"):
        application.init(["--image-path", str(first_frame)])
    with pytest.raises(SystemExit):
        application.init(["--prompt", "A waterfall"])
    with pytest.raises(FileNotFoundError, match="first-frame image does not exist"):
        application.init(
            [
                "--prompt",
                "A waterfall",
                "--image-path",
                str(tmp_path / "missing.png"),
            ]
        )


def test_application_rejects_multiple_blocks(tmp_path: Path) -> None:
    application = Wan22TI2VApplication()

    with pytest.raises(ValueError, match="exactly one autoregressive block"):
        application.init(
            [
                "--prompt",
                "A waterfall",
                "--image-path",
                str(_first_frame(tmp_path)),
                "--total-blocks",
                "2",
            ]
        )


def test_session_initializes_from_first_frame_and_completes_one_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_frame_path = _first_frame(tmp_path)
    first_frame = torch.zeros((1, 3, 4, 4), dtype=torch.bfloat16)
    load_calls: list[dict[str, Any]] = []

    def _load_first_frame(path: Path, **kwargs: Any) -> torch.Tensor:
        load_calls.append({"path": path, **kwargs})
        return first_frame

    monkeypatch.setattr(app_module, "load_first_frame_tensor", _load_first_frame)
    pipeline = _FakePipeline()
    application = Wan22TI2VApplication()
    application.init(
        [
            "--prompt",
            "A waterfall",
            "--image-path",
            str(first_frame_path),
            "--device",
            "cpu",
        ]
    )
    assert application._session_config is not None
    application._session_config = replace(
        application._session_config,
        pipeline_config=_FakePipelineConfig(pipeline),
    )

    session = application.create_session()
    session.init()

    assert load_calls == [
        {
            "path": first_frame_path,
            "pixel_height": 640,
            "pixel_width": 1280,
            "device": torch.device("cpu"),
            "dtype": torch.bfloat16,
        }
    ]
    assert pipeline.initialize_calls == [
        {"text": ["A waterfall"], "image": first_frame}
    ]
    assert session.session_info().steady_output_frame_count == 81
    assert session.next_step_requirements() is not None

    result = session.step(
        CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))
    )

    assert result.step_index == 0
    assert pipeline.generated == [0]
    assert pipeline.finalized == [0]
    assert session.next_step_requirements() is None

    session.close()
    application.close()
    assert pipeline.closed


def test_application_entry_point_is_owned_by_wan22_package() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())

    assert "flashdreams-t2v" in manifest["project"]["dependencies"]
    assert "mediapy>=1.1" in manifest["project"]["dependencies"]
    assert "opencv-python-headless>=4.5" in manifest["project"]["dependencies"]
    assert manifest["project"]["entry-points"]["flashdreams.applications"] == {
        "ti2v-wan22": "wan22.ti2v.app:create_app"
    }
