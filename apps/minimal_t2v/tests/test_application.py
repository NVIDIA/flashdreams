# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the tutorial's startup, prompt messages, and cache lifecycle."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from minimal_t2v import MinimalT2VApplication
from minimal_t2v.ui_loop import MinimalT2VUILoop

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _application() -> tuple[MinimalT2VApplication, Mock, Mock]:
    pipeline = Mock()
    pipeline.to.return_value = pipeline
    pipeline.eval.return_value = pipeline
    pipeline.decoder.spatial_compression_ratio = 8
    pipeline.generate.return_value = torch.zeros(2, 3, 16, 32)
    pipeline.finalize.return_value = {"total_ms": 1.0}
    config = Mock()
    config.setup.return_value = pipeline
    app = MinimalT2VApplication(
        config, SessionDesc(video_width=32, video_height=16), total_blocks=2
    )
    return app, pipeline, config


def test_help_and_description_do_not_load_the_model() -> None:
    app, _, config = _application()
    assert app.session_desc().video_width == 32
    with pytest.raises(SystemExit) as exit_info:
        app.init(["--help"])
    assert exit_info.value.code == 0
    config.setup.assert_not_called()
    with pytest.raises(RuntimeError, match="init"):
        app.create_session(app.session_desc())


def test_prompt_generation_reset_and_close() -> None:
    app, pipeline, _ = _application()
    app.init(["--default-prompt", " First prompt ", "--device", "cpu"])
    pipeline.to.assert_called_once_with("cpu")
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.model_loop
    events = UserInputEvents([])
    pipeline.initialize_cache.assert_not_called()

    first = loop.step(7, events)[0]
    cache = pipeline.initialize_cache.return_value
    pipeline.initialize_cache.assert_called_once_with(
        text=["First prompt"], height=2, width=4
    )
    pipeline.generate.assert_called_once_with(autoregressive_index=0, cache=cache)
    pipeline.finalize.assert_called_once_with(autoregressive_index=0, cache=cache)
    assert first.step_index == 7
    assert first.frame_count == 2
    assert first.output_layout is VideoTensorLayout.tchw
    assert first.read_output().shape == (2, 3, 16, 32)
    assert first.metrics == {"total_ms": 1.0}
    loop.step(8, events)
    assert loop.step(9, events) == []
    assert pipeline.generate.call_count == 2
    assert not loop.is_finished()  # The UI can still send another prompt.

    loop.reset()
    loop.step(10, events)
    assert pipeline.initialize_cache.call_count == 2
    pipeline.generate.assert_called_with(autoregressive_index=0, cache=cache)
    loop.close()
    assert loop.state.cache is None
    app.close()
    assert app.pipeline is None


def test_ui_captures_prompt_and_queues_model_thread_update() -> None:
    app, pipeline, _ = _application()
    app.init(["--device", "cpu"])
    session = app.create_session(app.session_desc())
    session.init()
    loop, ui = session.model_loop, session.ui_loop
    assert isinstance(ui, MinimalT2VUILoop)
    events = UserInputEvents([])
    assert loop.step(0, events) == []
    imgui = Mock()
    imgui.input_text.return_value = (True, "Second prompt")
    imgui.button.return_value = True

    ui.step_ui(imgui, 0, events)
    imgui.end.assert_called_once()
    ui.state.text = "Edited after clicking Send"
    assert loop.state.prompt == ""
    loop._run_message_batch()
    assert loop.state.prompt == "Second prompt"
    loop.step(4, events)
    pipeline.initialize_cache.assert_called_once_with(
        text=["Second prompt"], height=2, width=4
    )

    imgui.input_text.return_value = (True, "   ")
    ui.step_ui(imgui, 1, events)
    loop._run_message_batch()
    assert loop.step(5, events) == []
    assert loop.state.cache is None


@pytest.mark.parametrize(
    "changes",
    [
        {"video_width": 64},
        {"video_height": 32},
        {"output_layout": VideoTensorLayout.btchw},
        {"frames_per_second_for_step": 16},
    ],
)
def test_session_must_match_model_output(changes: dict) -> None:
    app, _, _ = _application()
    app.init(["--device", "cpu"])
    with pytest.raises(ValueError, match="output description"):
        app.create_session(replace(app.session_desc(), **changes))


def test_rejects_dimensions_that_truncate_latents() -> None:
    app, _, _ = _application()
    app.desc = replace(app.desc, video_width=31)
    app.init(["--device", "cpu"])
    with pytest.raises(ValueError, match="divisible"):
        app.create_session(app.session_desc())
