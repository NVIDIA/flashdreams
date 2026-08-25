# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from typing import Any

import pytest
import torch
from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VModelLoop,
    T2VModelState,
    TextInputSpec,
)

from flashdreams.api_v2.loop import invoke_async
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _Decoder:
    spatial_compression_ratio = 8


class _Pipeline:
    decoder = _Decoder()

    def __init__(self) -> None:
        self.device: str | None = None
        self.cache_calls: list[dict[str, Any]] = []
        self.generate_calls: list[int] = []
        self.finalize_calls: list[int] = []

    def to(self, device: str) -> "_Pipeline":
        self.device = device
        return self

    def eval(self) -> "_Pipeline":
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.cache_calls.append(kwargs)
        return object()

    def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
        del cache
        self.generate_calls.append(autoregressive_index)
        return torch.zeros((2, 3, 8, 12))

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, int]:
        del cache
        self.finalize_calls.append(autoregressive_index)
        return {"block": autoregressive_index}


class _PipelineConfig:
    def __init__(self, pipeline: _Pipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _Pipeline:
        return self.pipeline


def _defaults(pipeline: _Pipeline, **kwargs: Any) -> T2VApplicationDefaults:
    return T2VApplicationDefaults(
        pipeline_config=_PipelineConfig(pipeline),
        total_blocks=kwargs.pop("total_blocks", 2),
        pixel_height=64,
        pixel_width=96,
        **kwargs,
    )


def _desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=16,
        video_width=96,
        video_height=64,
    )


def _registered_loop(state: T2VModelState) -> T2VModelLoop:
    loop = T2VModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    return loop


def test_prompt_and_block_count_are_validated() -> None:
    app = T2VApplication(defaults=_defaults(_Pipeline(), prompt=""))
    with pytest.raises(ValueError, match="prompt"):
        app.init([])
    with pytest.raises(ValueError, match="total-blocks"):
        app.init(["--prompt", "A waterfall", "--total-blocks", "0"])


def test_session_registers_separate_model_and_slangpy_ui_loops() -> None:
    pipeline = _Pipeline()
    app = T2VApplication(
        defaults=_defaults(pipeline, prompt="A waterfall"),
        ui_renderer_factory=lambda width, height: object(),
    )
    app.init(["--device", "cpu"])
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()
    assert type(ui_loop).__name__ == "T2VSlangPyUILoop"
    assert type(model_loop).__name__ == "T2VModelLoop"
    assert ui_loop.state is not model_loop.state
    assert pipeline.device is None
    model_loop.step(0, UserInputEvents([]))
    assert pipeline.device == "cpu"


def test_model_loop_owns_cache_and_advances_autoregressive_blocks() -> None:
    pipeline = _Pipeline()
    state = T2VModelState(
        pipeline_factory=lambda: pipeline,
        session_desc=_desc(),
        prompt="A waterfall",
        total_blocks=2,
        text_values={},
        cache_factory=lambda model_state: pipeline.initialize_cache(
            text=[model_state.prompt]
        ),
    )
    loop = _registered_loop(state)
    first = loop.step(0, UserInputEvents([]))[0]
    second = loop.step(1, UserInputEvents([]))[0]
    assert len(pipeline.cache_calls) == 1
    assert pipeline.generate_calls == [0, 1]
    assert pipeline.finalize_calls == [0, 1]
    assert first.output.shape == second.output.shape == (2, 3, 8, 12)
    assert loop.is_finished()


def test_invoke_async_request_is_applied_before_next_model_step() -> None:
    pipeline = _Pipeline()
    state = T2VModelState(
        pipeline_factory=lambda: pipeline,
        session_desc=_desc(),
        prompt="old",
        total_blocks=1,
        text_values={},
        cache_factory=lambda model_state: pipeline.initialize_cache(
            text=[model_state.prompt], **model_state.text_values
        ),
    )
    loop = _registered_loop(state)
    invoke_async(
        loop,
        lambda model_state: model_state.request_generation(
            "new prompt", 3, {"image_path": "frame.png"}
        ),
    )
    assert loop._begin_run(UserInputEvents([]), 0) == 0
    loop.step(0, UserInputEvents([]))
    assert state.prompt == "new prompt"
    assert state.total_blocks == 3
    assert pipeline.cache_calls == [{"text": ["new prompt"], "image_path": "frame.png"}]


def test_required_integration_text_input_is_validated() -> None:
    app = T2VApplication(
        defaults=_defaults(
            _Pipeline(),
            prompt="A waterfall",
            text_inputs=(
                TextInputSpec(name="image_path", label="First frame", required=True),
            ),
        )
    )
    with pytest.raises(ValueError, match="First frame"):
        app.init([])
