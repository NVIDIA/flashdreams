# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for MP4 output combined with synthetic input."""

import json
from pathlib import Path
from typing import cast

import pytest
from flashdreams.api_v2.input_source import SessionInputSource
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.synthetic_input_mp4_client_window import (
    SyntheticInputMp4ClientWindow,
)
from flashdreams.runtime_v2.synthetic_input_source import SyntheticInputSource

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _Output:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[object] = []

    def open(self, session_desc: SessionDesc) -> None:
        self.calls.append(session_desc)

    def write(self, result: StepResult) -> None:
        self.calls.append(result)

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        self.calls.append(new_window_size)

    def close(self) -> None:
        self.calls.append("close")


class _SessionInputSource(SessionInputSource):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")

    def get_user_input_events(self):
        raise AssertionError("Input is not polled by this test.")


def test_the_composite_delegates_mp4_output_and_opens_its_input_clock(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "input.json"
    descriptor.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {"on": "ui_loop", "at": 0, "event": {"type": "reset"}}
                ],
            }
        ),
        encoding="utf-8",
    )
    clock = _Clock(1_000_000)
    output = _Output(tmp_path / "clip.mp4")
    window = SyntheticInputMp4ClientWindow(
        cast(Mp4ClientWindow, output),
        SyntheticInputSource(descriptor, clock_ns=clock),
    )
    session_desc = cast(SessionDesc, object())
    result = cast(StepResult, object())

    window.open(session_desc)
    clock.now_ns = 1_005_000
    events = window.get_user_input_events().get_events()
    window.write(result)
    window.request_new_window_size((320, 180))
    window.close()

    assert window.path == tmp_path / "clip.mp4"
    assert events[0].get_timestamp() == 5
    assert output.calls == [session_desc, result, (320, 180), "close"]


def test_the_composite_accepts_any_session_input_source(tmp_path: Path) -> None:
    source = _SessionInputSource()
    window = SyntheticInputMp4ClientWindow(
        cast(Mp4ClientWindow, _Output(tmp_path / "clip.mp4")), source
    )

    window.open(cast(SessionDesc, object()))
    window.close()

    assert source.calls == ["open", "close"]