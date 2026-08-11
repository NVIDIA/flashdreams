# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from flashdreams.runtime import InferenceConfig, InferenceInput, StepRequirements
from flashdreams.runtime.demo import (
    DemoSpec,
    NullOutputSpec,
    PreparedScenario,
    UserInputWindow,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashvsr.demo import (
    FLASHVSR_MODEL_ID,
    FlashVSRDemoAdapter,
    FlashVSRVideoInputProvider,
    FlashVSRVideoScenario,
    PreparedFlashVSRVideo,
)
from flashvsr.demo.app import _replay_spec, _webrtc_spec, parse_args
from flashvsr.demo.providers import PREPARED_VIDEO_METADATA_KEY
from flashvsr.demo.spec import prepare_video_source
from flashvsr.demo.webrtc import serve_flashvsr_webrtc_demo
from flashvsr.runtime import (
    FIELD_CHUNK_SIZE,
    FIELD_FPS,
    FIELD_INPUT_HEIGHT,
    FIELD_INPUT_WIDTH,
    FIELD_TAIL_POLICY,
    FIELD_TOTAL_FRAMES,
    FIELD_VALID_FRAME_COUNT,
    FIELD_VIDEO_CHUNK,
    FlashVSRModelAdapter,
)

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    def __init__(self) -> None:
        self.diffusion_model = SimpleNamespace(
            dtype=torch.float32,
            rng=torch.Generator().manual_seed(0),
        )
        self.generated: list[tuple[int, tuple[int, ...]]] = []
        self.finalized: list[int] = []
        self.cache = SimpleNamespace(reset_count=0)

    def initialize_cache(self) -> Any:
        return self.cache

    def reset_cache_in_place(self, cache: Any) -> None:
        assert cache is self.cache
        cache.reset_count += 1

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: Any,
        input: torch.Tensor,
    ) -> torch.Tensor:
        assert cache is self.cache
        self.generated.append((autoregressive_index, tuple(input.shape)))
        return input + 0.25

    def finalize(self, *, autoregressive_index: int, cache: Any) -> dict[str, float]:
        assert cache is self.cache
        self.finalized.append(autoregressive_index)
        return {"total_ms": 2.5}


class _FakeRuntime:
    def __init__(self, *, config: InferenceConfig, options: Any) -> None:
        self.config = config
        self.options = options

    def preload(self) -> None:
        return

    def peek_input_fps(self) -> float:
        return float(self.config.runtime_options["fps"])

    def peek_steady_output_num_frames(self) -> int:
        return int(self.config.runtime_options["chunk_size"])

    def start_session(self, inputs: InferenceInput) -> Any:
        raise AssertionError(f"test server should not start a session: {inputs}")

    def close(self) -> None:
        return


def _pipeline_config() -> Any:
    return SimpleNamespace(encoder=SimpleNamespace(scale=2))


def _initial_inputs(
    *,
    total_frames: int | None = 13,
    tail_policy: str = "drop",
) -> InferenceInput:
    values: dict[str, Any] = {
        FIELD_INPUT_HEIGHT: 64,
        FIELD_INPUT_WIDTH: 64,
        FIELD_FPS: 20.0,
        FIELD_CHUNK_SIZE: 8,
        FIELD_TAIL_POLICY: tail_policy,
    }
    if total_frames is not None:
        values[FIELD_TOTAL_FRAMES] = total_frames
    return InferenceInput(global_conditioning=values)


def _prepared_video(
    *,
    frames: int = 6,
    loop_input: bool = False,
    tail_policy: str = "pad",
) -> PreparedFlashVSRVideo:
    video = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1)
    video = video.expand(1, 3, frames, 64, 64).contiguous()
    scenario = FlashVSRVideoScenario(
        input_path="memory.mp4",
        chunk_size=8,
        fps=20.0,
        tail_policy=tail_policy,
        loop_input=loop_input,
    )
    return PreparedFlashVSRVideo(
        scenario=scenario,
        resolved_path=Path("memory.mp4"),
        video=video,
        input_height=64,
        input_width=64,
        target_height=128,
        target_width=128,
        fps=20.0,
    )


def test_adapter_declares_native_video_inputs_and_demo_modes() -> None:
    adapter = FlashVSRDemoAdapter()

    assert adapter.model_id == FLASHVSR_MODEL_ID
    assert adapter.supported_input_modes() == ("replay",)
    assert adapter.supported_output_modes() == ("mp4", "null", "webrtc")
    global_fields = {
        field.name
        for field in adapter.inference_input_schema.global_conditioning_fields
    }
    step_fields = {field.name for field in adapter.inference_input_schema.step_fields}
    assert {
        FIELD_INPUT_HEIGHT,
        FIELD_INPUT_WIDTH,
        FIELD_FPS,
        FIELD_CHUNK_SIZE,
        FIELD_TAIL_POLICY,
    }.issubset(global_fields)
    assert step_fields == {FIELD_VIDEO_CHUNK}


def test_native_runtime_session_requests_and_processes_cold_then_steady() -> None:
    pipeline = _FakePipeline()
    config = InferenceConfig(
        model_id=FLASHVSR_MODEL_ID,
        device="cpu",
        seed=7,
        runtime_options={
            "pipeline_config": _pipeline_config(),
            "pipeline": pipeline,
            "chunk_size": 8,
            "fps": 20.0,
        },
    )
    runtime = FlashVSRModelAdapter().create_runtime(config)
    session = runtime.start_session(_initial_inputs())

    first = session.next_step_requirements()
    assert first is not None
    assert first.input_frame_count == 5
    first_result = session.step(
        InferenceInput(
            step={FIELD_VIDEO_CHUNK: torch.zeros(1, 3, 5, 64, 64)},
            metadata={FIELD_VALID_FRAME_COUNT: 5},
        )
    )
    second = session.next_step_requirements()
    assert second is not None
    assert second.input_frame_count == 8
    second_result = session.step(
        InferenceInput(
            step={FIELD_VIDEO_CHUNK: torch.zeros(1, 3, 8, 64, 64)},
            metadata={FIELD_VALID_FRAME_COUNT: 8},
        )
    )

    assert first_result.layout == "bcthw"
    assert first_result.frame_count == 5
    assert first_result.output_window is not None
    assert first_result.output_window.end_s == pytest.approx(0.25)
    assert first_result.metadata["resolution"] == {"width": 64, "height": 64}
    assert second_result.frame_count == 8
    assert second_result.output_window is not None
    assert second_result.output_window.start_s == pytest.approx(0.25)
    assert second_result.metrics["total_ms"] == 2.5
    assert session.next_step_requirements() is None
    assert pipeline.generated == [
        (0, (1, 3, 5, 64, 64)),
        (1, (1, 3, 8, 64, 64)),
    ]
    assert pipeline.finalized == [0, 1]

    session.close()
    second_session = runtime.start_session(_initial_inputs(total_frames=5))
    assert pipeline.cache.reset_count == 1
    second_session.close()
    runtime.close()


def test_session_rejects_invalid_tail_metadata_before_model_execution() -> None:
    pipeline = _FakePipeline()
    runtime = FlashVSRModelAdapter().create_runtime(
        InferenceConfig(
            model_id=FLASHVSR_MODEL_ID,
            device="cpu",
            runtime_options={
                "pipeline_config": _pipeline_config(),
                "pipeline": pipeline,
                "chunk_size": 8,
                "fps": 20.0,
            },
        )
    )
    session = runtime.start_session(_initial_inputs(total_frames=3, tail_policy="pad"))

    with pytest.raises(ValueError, match="valid frame count mismatch"):
        session.step(
            InferenceInput(
                step={FIELD_VIDEO_CHUNK: torch.zeros(1, 3, 5, 64, 64)},
                metadata={FIELD_VALID_FRAME_COUNT: 5},
            )
        )

    request = session.next_step_requirements()
    assert request is not None
    assert request.step_index == 0
    assert pipeline.generated == []
    assert pipeline.finalized == []
    session.close()
    runtime.close()


def test_provider_loops_short_source_to_exact_requested_shape() -> None:
    prepared = _prepared_video(frames=3, loop_input=True)
    scenario = PreparedScenario(
        initial_inputs=_initial_inputs(total_frames=None),
        metadata={PREPARED_VIDEO_METADATA_KEY: prepared},
    )
    provider = FlashVSRVideoInputProvider(
        scenario=scenario,
        inference_input_schema=FlashVSRModelAdapter().inference_input_schema,
    )

    step = provider.prepare_step(
        request=StepRequirements(
            step_index=0,
            input_frame_count=5,
            metadata={FIELD_VALID_FRAME_COUNT: 5},
        ),
        user_window=UserInputWindow(start_s=0.0, end_s=0.25),
    )

    assert step.inference_input is not None
    chunk = step.inference_input.step[FIELD_VIDEO_CHUNK]
    assert tuple(chunk.shape) == (1, 3, 5, 64, 64)
    assert chunk[0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_shared_replay_uses_provider_for_padded_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashvsr.demo.adapter as adapter_module

    prepared = _prepared_video(frames=6, tail_policy="pad")
    monkeypatch.setattr(
        adapter_module, "prepare_video_source", lambda *a, **k: prepared
    )
    pipeline = _FakePipeline()
    spec = DemoSpec(
        model_id=FLASHVSR_MODEL_ID,
        input_mode="replay",
        scenario=prepared.scenario,
        output=NullOutputSpec(),
        config=InferenceConfig(
            model_id=FLASHVSR_MODEL_ID,
            device="cpu",
            seed=0,
            runtime_options={
                "pipeline_config": _pipeline_config(),
                "pipeline": pipeline,
                "chunk_size": 8,
                "fps": 20.0,
            },
        ),
    )

    result = run_replay_demo(spec=spec, adapter=FlashVSRDemoAdapter())

    assert result.status == "completed"
    assert pipeline.generated == [
        (0, (1, 3, 5, 64, 64)),
        (1, (1, 3, 8, 64, 64)),
    ]


def test_prepare_video_source_normalizes_and_derives_target_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flashvsr.demo.spec as spec_module

    path = tmp_path / "input.mp4"
    path.write_bytes(b"fixture")
    pixels = np.full((5, 64, 96, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(spec_module, "resolve_input_path", lambda *a, **k: path)
    monkeypatch.setattr(spec_module, "read_video_rgb", lambda _: pixels)

    prepared = prepare_video_source(
        FlashVSRVideoScenario(input_path=path, chunk_size=8, fps=24.0),
        scale=2,
    )

    assert tuple(prepared.video.shape) == (1, 3, 5, 64, 96)
    assert torch.all(prepared.video == 1)
    assert (prepared.target_height, prepared.target_width) == (128, 128)


def test_cli_builds_null_and_webrtc_specs(tmp_path: Path) -> None:
    replay_args = parse_args(
        [
            "replay",
            "--input",
            str(tmp_path / "input.mp4"),
            "--output-mode",
            "null",
            "--chunk-size",
            "8",
            "--fps",
            "24",
            "--no-compile",
            "--no-cuda-graph",
            "--color-corrector",
            "torch",
        ]
    )
    replay_spec = _replay_spec(replay_args)
    assert isinstance(replay_spec.output, NullOutputSpec)
    assert replay_spec.config is not None
    assert replay_spec.config.compile is False
    assert replay_spec.config.runtime_options["use_cuda_graph"] is False
    assert (
        replay_spec.config.runtime_options["color_corrector_implementation"] == "torch"
    )

    webrtc_args = parse_args(
        [
            "webrtc",
            "--input",
            str(tmp_path / "input.mp4"),
            "--port",
            "9090",
            "--fps",
            "29.97",
            "--warmup-chunks",
            "1",
            "--prefer-sw-encoder",
        ]
    )
    webrtc_spec = _webrtc_spec(webrtc_args, device="cuda:3")
    assert isinstance(webrtc_spec.output, WebRTCOutputSpec)
    assert webrtc_spec.output.port == 9090
    assert isinstance(webrtc_spec.scenario, FlashVSRVideoScenario)
    assert webrtc_spec.scenario.loop_input is True
    assert webrtc_spec.scenario.fps == 30.0
    assert webrtc_spec.output.fps == 30
    assert webrtc_spec.config is not None
    assert webrtc_spec.config.device == "cuda:3"
    assert webrtc_spec.config.runtime_options["fps"] == 30.0


def test_cli_uses_source_fps_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import flashvsr.demo.app as app_module

    input_path = tmp_path / "input.mp4"
    monkeypatch.setattr(
        app_module,
        "resolve_input_path",
        lambda *args, **kwargs: input_path,
    )
    monkeypatch.setattr(app_module, "read_video_fps", lambda path: 24.0)

    args = parse_args(["replay", "--input", str(input_path), "--output-mode", "null"])
    spec = _replay_spec(args)

    assert isinstance(spec.scenario, FlashVSRVideoScenario)
    assert spec.scenario.fps == 24.0
    assert spec.config is not None
    assert spec.config.runtime_options["fps"] == 24.0


def test_webrtc_uses_native_shared_host_and_resolved_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashvsr.demo.adapter as adapter_module

    prepared = _prepared_video(frames=5, loop_input=True)
    monkeypatch.setattr(
        adapter_module, "prepare_video_source", lambda *a, **k: prepared
    )
    calls: list[dict[str, Any]] = []
    spec = DemoSpec(
        model_id=FLASHVSR_MODEL_ID,
        input_mode="replay",
        scenario=prepared.scenario,
        output=WebRTCOutputSpec(
            host="127.0.0.1",
            port=8088,
            fps=20,
            video_height=1,
            video_width=1,
            warmup_chunks=0,
        ),
        config=InferenceConfig(
            model_id=FLASHVSR_MODEL_ID,
            device="cpu",
            runtime_options={
                "pipeline_config": _pipeline_config(),
                "chunk_size": 8,
                "fps": 20.0,
                "prefer_sw_encoder": True,
            },
        ),
    )

    result = serve_flashvsr_webrtc_demo(
        spec=spec,
        world_rank=1,
        runtime_factory=_FakeRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    assert result is None
    assert len(calls) == 1
    manager = calls[0]["session_manager"]
    assert type(manager) is BaseWebRTCSessionManager
    assert manager._shared_host is not None
    assert isinstance(manager._shared_adapter, FlashVSRDemoAdapter)
    assert manager._shared_scenario is not None
    assert manager.runtime_config.video_height == 128
    assert manager.runtime_config.video_width == 128
    assert manager.runtime_config.encoder_backend == "default"
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8088
    manager._shared_host.close()


def test_replay_cli_requires_output_only_for_mp4(tmp_path: Path) -> None:
    parse_args(["replay", "--output", str(tmp_path / "output.mp4")])
    with pytest.raises(SystemExit):
        parse_args(["replay"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "replay",
                "--output-mode",
                "null",
                "--output",
                str(tmp_path / "output.mp4"),
            ]
        )
