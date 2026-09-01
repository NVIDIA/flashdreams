# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the Wan 2.2 TI2V application adapter."""

from pathlib import Path

import pytest
import torch
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig
from wan22.apps.t2v import adapter as adapter_module
from wan22.apps.t2v.adapter import Wan22TI2VApplication, create_app

from flashdreams.api_v2.application import IApplication

pytestmark = pytest.mark.ci_cpu


def _first_frame(tmp_path: Path) -> Path:
    path = tmp_path / "first-frame.png"
    path.write_bytes(b"test image placeholder")
    return path


def test_factory_exposes_single_block_ti2v_defaults() -> None:
    """Keep the application geometry and rollout mode with its model config."""
    application = create_app()

    assert isinstance(application, IApplication)
    assert isinstance(application, Wan22TI2VApplication)
    assert application.defaults.total_blocks == 1
    assert application.defaults.pixel_height == 640
    assert application.defaults.pixel_width == 1280
    assert application.defaults.fps == 16


def test_application_requires_prompt_and_existing_first_frame(tmp_path: Path) -> None:
    """Reject missing static conditioning before loading the checkpoint."""
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


def test_first_frame_seeds_initial_and_reset_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain image conditioning when the shared model loop resets."""
    first_frame_path = _first_frame(tmp_path)
    first_frame = torch.zeros((1, 3, 4, 4), dtype=torch.bfloat16)
    monkeypatch.setattr(
        adapter_module,
        "load_first_frame_tensor",
        lambda *_args, **_kwargs: first_frame,
    )
    pipeline = FakeT2VPipeline()
    application = Wan22TI2VApplication(pipeline_config=FakeT2VPipelineConfig(pipeline))
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
    session = application.create_session(application.session_desc())

    session.init()
    session.model_loop.reset()

    assert pipeline.caches == [
        {"text": ["A waterfall"], "image": first_frame},
        {"text": ["A waterfall"], "image": first_frame},
    ]

    session.close()
    application.close()


@pytest.mark.parametrize("total_blocks", [0, 2])
def test_application_rejects_non_single_block_runs(
    tmp_path: Path,
    total_blocks: int,
) -> None:
    """Reject rollout lengths the bidirectional model cannot continue."""
    application = Wan22TI2VApplication()

    with pytest.raises(ValueError, match="must be"):
        application.init(
            [
                "--prompt",
                "A waterfall",
                "--image-path",
                str(_first_frame(tmp_path)),
                "--total-blocks",
                str(total_blocks),
            ]
        )
