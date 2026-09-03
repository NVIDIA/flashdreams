# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the V2 PNG output mode."""

from argparse import Namespace

import pytest
import torch
from PIL import Image

from flashdreams.runtime_v2.client_window_factory import client_window_mode
from flashdreams.runtime_v2.png_output_sink import PngOutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_png_sink_writes_first_and_numbered_frames(tmp_path) -> None:
    path = tmp_path / "frame.png"
    desc = SessionDesc(
        output_layout=VideoTensorLayout.tchw, video_width=3, video_height=2
    )
    sink = PngOutputSink(path)
    sink.open(desc)
    sink.write(
        StepResult(
            step_index=0,
            output=torch.stack(
                [
                    torch.full((3, 2, 3), -1.0),
                    torch.full((3, 2, 3), 1.0),
                ]
            ),
            frame_count=2,
            output_layout=VideoTensorLayout.tchw,
        )
    )
    sink.close()

    assert Image.open(path).getpixel((0, 0)) == (0, 0, 0)
    assert Image.open(tmp_path / "frame-00001.png").getpixel((0, 0)) == (
        255,
        255,
        255,
    )


def test_png_mode_requires_png_output_path(tmp_path) -> None:
    mode = client_window_mode("png")
    with pytest.raises(ValueError, match="must end in .png"):
        mode.check_arguments(Namespace(output_path=tmp_path / "frame.jpg"))
