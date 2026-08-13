# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from flashdreams.runtime import ApplicationRunner, FlashDreamsApplication
from flashdreams.runtime.demo import (
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.serving.io_handlers import (
    Mp4IOHandler,
    NativeWindowIOHandler,
    NullIOHandler,
    WebRTCIOHandler,
)

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
            supported_control_keys=frozenset(),
            native_presenter_factory=None,
            native_key_bindings=None,
            webrtc_app_resources=WebRTCAppResources(preload_name="Example"),
        ),
    )


def test_io_handlers_create_transport_specific_outputs(tmp_path: Path) -> None:
    application = _application()

    mp4 = Mp4IOHandler(tmp_path / "result.mp4").create_output(application)
    null = NullIOHandler().create_output(application)
    webrtc = WebRTCIOHandler(host="127.0.0.1", port=9000).create_output(application)
    native = NativeWindowIOHandler().create_output(application)

    assert isinstance(mp4, Mp4OutputSpec)
    assert isinstance(null, NullOutputSpec)
    assert isinstance(webrtc, WebRTCOutputSpec)
    assert isinstance(native, NativeWindowOutputSpec)
    assert webrtc.port == 9000


def test_io_parsers_consume_only_their_arguments(tmp_path: Path) -> None:
    handler, remaining = Mp4IOHandler.from_argv(
        [f"--output={tmp_path / 'result.mp4'}", "--model-flag", "value"]
    )

    assert isinstance(handler, Mp4IOHandler)
    assert handler.output_path == tmp_path / "result.mp4"
    assert remaining == ["--model-flag", "value"]


def test_io_handlers_reject_irrelevant_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not accept --output"):
        NullIOHandler.from_argv([f"--output={tmp_path / 'unused'}"])
    with pytest.raises(ValueError, match="does not accept --host"):
        NativeWindowIOHandler.from_argv(["--host", "127.0.0.1"])
    with pytest.raises(ValueError, match="host must be non-empty"):
        WebRTCIOHandler(host="").create_output(_application())
    with pytest.raises(ValueError, match="port must be between"):
        WebRTCIOHandler(port=0).create_output(_application())


def test_webrtc_io_rejects_multi_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    application = _application()

    with pytest.raises(RuntimeError, match="one process"):
        WebRTCIOHandler().run(
            ApplicationRunner(
                application=application,
                io_handler=WebRTCIOHandler(),
            )
        )
