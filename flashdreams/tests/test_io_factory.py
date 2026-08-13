# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from flashdreams.runtime.demo import (
    FlashDreamsApplication,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
from flashdreams.serving.io_factory import IOOptions, io_factories

pytestmark = pytest.mark.ci_cpu


def _application() -> FlashDreamsApplication:
    return cast(
        FlashDreamsApplication,
        SimpleNamespace(
            application_name="example",
            fps=30,
            video_width=16,
            video_height=8,
            output_layout="tchw",
            title="Example",
        ),
    )


def test_io_factories_cover_public_application_modes() -> None:
    assert tuple(io_factories()) == ("mp4", "null", "webrtc", "local-window")


def test_io_factories_create_mode_specific_outputs(tmp_path: Path) -> None:
    application = _application()
    factories = io_factories()
    mp4 = factories["mp4"].create_output(
        application,
        IOOptions(output_path=tmp_path / "result.mp4"),
    )
    null = factories["null"].create_output(application, IOOptions())
    webrtc = factories["webrtc"].create_output(
        application,
        IOOptions(host="127.0.0.1", port=9000),
    )
    native = factories["local-window"].create_output(application, IOOptions())

    assert isinstance(mp4, Mp4OutputSpec)
    assert isinstance(null, NullOutputSpec)
    assert isinstance(webrtc, WebRTCOutputSpec)
    assert isinstance(native, NativeWindowOutputSpec)
    assert webrtc.port == 9000


def test_io_factories_reject_irrelevant_options(tmp_path: Path) -> None:
    application = _application()
    with pytest.raises(ValueError, match="does not accept --output"):
        io_factories()["null"].create_output(
            application,
            IOOptions(output_path=tmp_path / "unused"),
        )
    with pytest.raises(ValueError, match="does not accept IO overrides"):
        io_factories()["local-window"].create_output(
            application,
            IOOptions(host="127.0.0.1"),
        )
