# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 client window abstraction."""

import json

import pytest
import torch
from null_model import NULL_MODEL_CONFIG
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.input_source import InputSource
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    NumeralKeypadUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class FakeClientWindow(IClientWindow):
    """Provide fake client input and output for one session."""

    def __init__(self) -> None:
        self.session_desc: SessionDesc | None = None
        self._input = UserInputEvents([])
        self.results: list[StepResult] = []

        # Only applies to writing to output; reading from input will just not produce "new"
        # results if closed, it does not imply the InputSource contains invalid data.
        self._is_open = False

    def get_user_input_events(self) -> UserInputEvents:
        return self._input

    def put_user_input_event(self, event: UserInputEvent) -> None:
        del event

    def open(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc
        self._is_open = True

    def write(self, result: StepResult) -> None:
        assert self._is_open
        self.results.append(result)

    def close(self) -> None:
        self._is_open = False

    def update_input_events(self, input: UserInputEvents) -> None:
        self._input = input


def test_client_window_for_null_model() -> None:
    # InputSource + OutputSink setup; the session layout desc arrives on open.
    client_window = FakeClientWindow()
    assert isinstance(client_window, IClientWindow)
    assert isinstance(client_window, InputSource)
    assert isinstance(client_window, OutputSink)
    ## Assume the sink takes time to open due to startup time for backend
    client_window.open(
        SessionDesc(
            output_layout=NULL_MODEL_CONFIG.output_layout,
            frames_per_second_for_ui=1,
            frames_per_second_for_step=1,
            video_width=1,
            video_height=1,
        )
    )

    # Pipeline setup
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()

    current_timestamp = uint64(0)
    current_step_index = 0
    test_event_data = 2
    while current_timestamp < 1000:
        numeral_keypad_input = NumeralKeypadUserInputEvent(
            timestamp=current_timestamp,
            value=test_event_data,
        )

        # This is the client-windowing system updating user-inputs handled by the
        # InputSource.
        client_window.update_input_events(UserInputEvents([numeral_keypad_input]))

        # Model and UI threads share this input.
        get_user_input_events = client_window.get_user_input_events()
        assert get_user_input_events.get_events() == [numeral_keypad_input]
        event = get_user_input_events.get_events()[0]
        assert isinstance(event, NumeralKeypadUserInputEvent)

        # This is inside our `step` loop.
        output = pipeline.generate(
            current_step_index, cache, input=torch.tensor([[event.value]])
        )
        ## Note: model output is in bcthw layout, but in theory the model could output bctwh and we would require a swizzle operation to get to bcthw
        client_window.write(
            StepResult(
                step_index=current_step_index,
                output=output,
                frame_count=1,
                output_layout=NULL_MODEL_CONFIG.output_layout,
                metrics={},
            )
        )

        assert event.get_type_name() == "numeral_keypad"
        assert event.value == test_event_data
        assert output.shape == (1, 3, 1, 1, 1)
        assert output[0, 0, 0, 0, 0].item() == current_step_index + test_event_data

        # Increment the step index and timestamp
        current_step_index += 1
        current_timestamp += 100
    client_window.close()


def test_webrtc_input_clock_survives_session_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("aiohttp")
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

    session_desc = SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )
    window = WebRTCClientWindow()
    timestamp_us = 10_000
    monkeypatch.setattr(window.server, "_timestamp_us", lambda: uint64(timestamp_us))
    try:
        window.open(session_desc)
        server_origin_ns = window.server.input_timestamp_origin_ns
        first_session_origin_ns = window.input_timestamp_origin_ns
        if server_origin_ns is None or first_session_origin_ns is None:
            pytest.fail("opening a session did not bind the input clock")
        timestamp_us = 20_000
        window.server._buffer_browser_message(
            json.dumps({"type": "keyboard", "key": "w", "pressed": True})
        )
        first_session_event = window.get_user_input_events().get_events()[0]

        window.server._buffer_browser_message(json.dumps({"type": "close"}))
        timestamp_us = 100_000
        window.open(session_desc)
        replacement_origin_ns = window.input_timestamp_origin_ns
        if replacement_origin_ns is None:
            pytest.fail("the replacement session did not bind the input clock")
        timestamp_us = 101_000
        window.server._buffer_browser_message(
            json.dumps({"type": "focus", "focused": True})
        )

        replacement_events = window.get_user_input_events().get_events()
        if [type(event) for event in replacement_events] != [FocusUserInputEvent]:
            pytest.fail("the replacement session retained a previous session's events")
        relative_times_us = (
            int(first_session_event.get_timestamp()),
            int(replacement_events[0].get_timestamp()),
        )
        if relative_times_us != (10_000, 1_000):
            pytest.fail(
                f"session-relative input timestamps changed: {relative_times_us}"
            )
        absolute_times_ns = (
            first_session_origin_ns + relative_times_us[0] * 1_000,
            replacement_origin_ns + relative_times_us[1] * 1_000,
        )
        if absolute_times_ns != (
            server_origin_ns + 20_000_000,
            server_origin_ns + 101_000_000,
        ):
            pytest.fail(
                f"session replacement changed the clock mapping: {absolute_times_ns}"
            )
    finally:
        window.close()
