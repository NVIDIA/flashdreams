# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 client window abstraction."""

import pytest
import torch
from null_model import NULL_MODEL_CONFIG
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.input_source import InputSource
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    NumeralKeypadUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

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
        numeral_keypad_input = UserInputEvent(
            timestamp=current_timestamp,
            event_data=NumeralKeypadUserInputEventData(value=test_event_data),
        )

        # This is the client-windowing system updating user-inputs handled by the
        # InputSource.
        client_window.update_input_events(UserInputEvents([numeral_keypad_input]))

        # This is the InputSource getting the user-inputs to send to our step/step_ui loops
        get_user_input_events = client_window.get_user_input_events()
        assert get_user_input_events.get_events() == [numeral_keypad_input]
        event_data = get_user_input_events.get_events()[0].get_event_data()
        assert isinstance(event_data, NumeralKeypadUserInputEventData)

        # This is inside our `step` loop.
        output = pipeline.generate(
            current_step_index, cache, input=torch.tensor([[event_data.value]])
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

        assert (
            numeral_keypad_input.get_event_data().get_type_name()
            == NumeralKeypadUserInputEventData.get_type_name()
        )
        assert event_data.get_type_name() == "numeral_keypad"
        assert event_data.value == test_event_data
        assert output.shape == (1, 3, 1, 1, 1)
        assert output[0, 0, 0, 0, 0].item() == current_step_index + test_event_data

        # Increment the step index and timestamp
        current_step_index += 1
        current_timestamp += 100
    client_window.close()
