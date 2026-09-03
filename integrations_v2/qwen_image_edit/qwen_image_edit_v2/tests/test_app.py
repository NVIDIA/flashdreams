# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the one-shot Qwen Image Edit V2 application."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from PIL import Image
from qwen_image_edit_v2.app import QwenImageEditApplication

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _Editor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def generate(self, *args, **kwargs) -> Image.Image:
        self.calls.append((*args, kwargs))
        return Image.new("RGB", kwargs["output_size"], (9, 8, 7))


def test_application_emits_one_uint8_frame_then_finishes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "black").save(input_path)
    editor = _Editor()
    app = QwenImageEditApplication(editor_factory=lambda device: editor)
    app.init(
        [
            "--input",
            str(input_path),
            "--prompt",
            "make it a city",
            "--seed",
            "12",
        ]
    )
    desc = SessionDesc(
        output_layout=VideoTensorLayout.tchw, video_width=32, video_height=16
    )
    session = app.create_session(desc)
    session.init()
    _, loop = session._take_loops()

    [result] = cast(list, loop.step(0, UserInputEvents([])))

    assert result.read_output().shape == (1, 3, 16, 32)
    assert result.read_output().dtype == torch.uint8
    assert tuple(result.read_output()[0, :, 0, 0].tolist()) == (9, 8, 7)
    assert loop.is_finished()
    assert editor.calls[0][-1]["seed"] == 12
    assert editor.calls[0][-1]["negative_prompt"] == " "
    assert editor.calls[0][-1]["true_cfg_scale"] == 4.0


def test_application_rejects_non_grid_resolution(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(input_path)
    app = QwenImageEditApplication(editor_factory=lambda device: _Editor())
    app.init(["--input", str(input_path), "--prompt", "edit"])

    with pytest.raises(ValueError, match="divisible by 16"):
        app.create_session(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw,
                video_width=31,
                video_height=16,
            )
        )


def test_application_rejects_invalid_true_cfg_scale(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(input_path)
    app = QwenImageEditApplication(editor_factory=lambda device: _Editor())

    with pytest.raises(ValueError, match="greater than 1"):
        app.init(
            [
                "--input",
                str(input_path),
                "--prompt",
                "edit",
                "--true-cfg-scale",
                "1",
            ]
        )
