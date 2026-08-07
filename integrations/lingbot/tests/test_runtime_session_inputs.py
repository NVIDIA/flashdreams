# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The replay session must consume its per-step inputs, not ignore them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lingbot.runtime as runtime_module
import pytest
import torch
from lingbot.input_mapping import FIELD_CAMERA_INTRINSICS, FIELD_CAMERA_TRAJECTORY
from lingbot.runtime import (
    LINGBOT_MODEL_ID,
    LingbotReplayRuntime,
    LingbotReplayRuntimeOptions,
    LingbotSessionInputs,
)

from flashdreams.runtime import InferenceConfig, InferenceInput

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    """Records what the session hands the model."""

    def __init__(self, *, supports_text_swap: bool = True) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.text_encoder_calls: list[list[str]] = []
        self.encoders_loaded = 0
        self.diffusion_model = _FakeDiffusionModel(
            supports_text_swap=supports_text_swap
        )

    def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> Any:
        del text, image
        return _FakeCache()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        del autoregressive_index
        return 2

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        input: Any,
    ) -> torch.Tensor:
        del cache
        self.generate_calls.append(
            {
                "autoregressive_index": autoregressive_index,
                "poses": input.poses.clone(),
                "intrinsics": input.intrinsics.clone(),
                "world_scale": input.world_scale,
            }
        )
        return torch.zeros(2, 3, 2, 2)

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"denoise_s": 0.1}

    def _ensure_oneshot_encoders_loaded(self) -> None:
        self.encoders_loaded += 1

    def text_encoder(self, texts: list[str]) -> torch.Tensor:
        self.text_encoder_calls.append(list(texts))
        return torch.ones(1, 4)


class _FakeCache:
    def __init__(self) -> None:
        self.transformer_cache = object()


class _FakeTransformer:
    def __init__(self) -> None:
        self.replaced: list[torch.Tensor] = []

    def replace_text_embeddings(self, cache: object, embeddings: torch.Tensor) -> None:
        del cache
        self.replaced.append(embeddings)


class _FakeDiffusionModel:
    def __init__(self, *, supports_text_swap: bool) -> None:
        self.transformer = _FakeTransformer() if supports_text_swap else object()


def _session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FakePipeline,
    *,
    total_blocks: int = 2,
    total_camera_frames: int | None = None,
):
    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    runtime = LingbotReplayRuntime(
        config=InferenceConfig(model_id=LINGBOT_MODEL_ID, device="cpu"),
        options=LingbotReplayRuntimeOptions(
            pipeline_config=object(),
            pipeline_factory=lambda pipeline_config, device: pipeline,
        ),
    )
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    session = runtime_module.LingbotReplaySession(
        pipeline=pipeline,
        session_inputs=LingbotSessionInputs(
            prompt="a calm street",
            first_frame_path=image,
            total_blocks=total_blocks,
            pixel_height=2,
            pixel_width=2,
            fps=16,
            world_scale=2.5,
            total_camera_frames=total_camera_frames,
        ),
        device=torch.device("cpu"),
        is_rank_zero=True,
        output_layout="tchw",
    )
    return runtime, session


def _step_payload(value: float) -> InferenceInput:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 2, 3] = value
    return InferenceInput(
        step={
            FIELD_CAMERA_TRAJECTORY: poses,
            FIELD_CAMERA_INTRINSICS: torch.full((2, 4), value),
        }
    )


def test_step_forwards_its_camera_inputs_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    session.step(_step_payload(1.0))
    session.step(_step_payload(2.0))

    assert len(pipeline.generate_calls) == 2
    # Distinct per-step payloads must reach the model distinctly; a session that
    # ignored its inputs would send the same slice twice.
    assert pipeline.generate_calls[0]["poses"][0, 2, 3] == 1.0
    assert pipeline.generate_calls[1]["poses"][0, 2, 3] == 2.0
    assert pipeline.generate_calls[0]["intrinsics"][0, 0] == 1.0
    assert pipeline.generate_calls[1]["intrinsics"][0, 0] == 2.0
    assert pipeline.generate_calls[0]["world_scale"] == 2.5
    runtime.close()


def test_step_rejects_missing_camera_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    with pytest.raises(ValueError, match="missing 'camera_trajectory'"):
        session.step(InferenceInput())

    assert pipeline.generate_calls == []
    runtime.close()


def test_step_rejects_wrongly_shaped_camera_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    with pytest.raises(ValueError, match=r"must have shape \(2, 4, 4\)"):
        session.step(
            InferenceInput(
                step={
                    FIELD_CAMERA_TRAJECTORY: torch.eye(4).repeat(5, 1, 1),
                    FIELD_CAMERA_INTRINSICS: torch.zeros(5, 4),
                }
            )
        )

    runtime.close()


def test_step_request_publishes_the_input_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    first = session.next_step_request()
    assert first is not None
    assert first.user_input_window is not None
    assert first.user_input_window.start_s == 0.0
    assert first.user_input_window.end_s == 2 / 16
    assert first.metadata == {"num_frames": 2, "frame_start": 0}

    session.step(_step_payload(1.0))
    second = session.next_step_request()
    assert second is not None
    assert second.user_input_window is not None
    # Windows must advance with the rollout so each step maps its own events.
    assert second.user_input_window.start_s == 2 / 16
    assert second.user_input_window.end_s == 4 / 16
    runtime.close()


def test_rollout_ends_when_the_camera_source_runs_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(
        tmp_path, monkeypatch, pipeline, total_blocks=10, total_camera_frames=3
    )

    assert session.next_step_request() is not None
    session.step(_step_payload(1.0))
    # Only 3 frames are available and each step needs 2, so the second step
    # would overrun the source.
    assert session.next_step_request() is None
    runtime.close()


def test_unbounded_source_runs_until_total_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(
        tmp_path, monkeypatch, pipeline, total_blocks=2, total_camera_frames=None
    )

    steps = 0
    while session.next_step_request() is not None:
        session.step(_step_payload(float(steps)))
        steps += 1

    assert steps == 2
    runtime.close()


def test_text_event_prompt_update_swaps_the_rollout_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    payload = _step_payload(1.0)
    session.step(
        InferenceInput(
            global_conditioning={"prompt": "a violent storm"},
            step=payload.step,
        )
    )

    assert pipeline.text_encoder_calls == [["a violent storm"]]
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, _FakeTransformer)
    assert len(transformer.replaced) == 1

    # Re-sending the same prompt must not re-encode or re-swap.
    session.step(
        InferenceInput(
            global_conditioning={"prompt": "a violent storm"},
            step=payload.step,
        )
    )
    assert pipeline.text_encoder_calls == [["a violent storm"]]
    runtime.close()


def test_text_event_on_an_unsupported_pipeline_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _FakePipeline(supports_text_swap=False)
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    with pytest.raises(RuntimeError, match="replace_text_embeddings"):
        session.step(
            InferenceInput(
                global_conditioning={"prompt": "a violent storm"},
                step=_step_payload(1.0).step,
            )
        )

    # A rollout with no text event must not need the capability at all.
    pipeline.generate_calls.clear()
    session.step(_step_payload(1.0))
    assert len(pipeline.generate_calls) == 1
    runtime.close()


def test_full_standard_loop_drives_the_session_through_the_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real runner must accept the mapping and feed the session per step.

    A fake runner cannot catch a mapping the compatibility check rejects, so
    this exercises flashdreams.runtime.run_inference_session itself.
    """
    import numpy as np
    from lingbot.runtime import (
        LingbotModelAdapter,
        LingbotReplayInputs,
        inference_input_from_replay_inputs,
    )

    from flashdreams.runtime import InputCanonicalizer, UserInputs, UserInputSchema
    from flashdreams.runtime.metrics import NullMetricsRecorder
    from flashdreams.runtime.output import OutputArtifact
    from flashdreams.runtime.runner import run_inference_session

    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    poses_path = tmp_path / "poses.npy"
    intrinsics_path = tmp_path / "intrinsics.npy"
    trajectory = np.tile(np.eye(4, dtype=np.float32), (32, 1, 1))
    trajectory[:, 2, 3] = np.arange(32, dtype=np.float32)
    np.save(poses_path, trajectory)
    np.save(
        intrinsics_path,
        np.tile(np.array([416.0, 416.0, 416.0, 240.0], dtype=np.float32), (32, 1)),
    )

    pipeline = _FakePipeline()
    monkeypatch.setattr(
        runtime_module,
        "load_first_frame_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 2, 2),
    )
    adapter = LingbotModelAdapter(
        pipeline_factory=lambda pipeline_config, device: pipeline,
    )
    replay_inputs = LingbotReplayInputs(
        prompt="a calm street",
        first_frame_path=image,
        camera_poses_path=poses_path,
        camera_intrinsics_path=intrinsics_path,
        total_blocks=3,
        pixel_height=2,
        pixel_width=2,
        fps=16,
    )

    class _Collecting:
        def __init__(self) -> None:
            self.results: list[Any] = []

        def open(self) -> None:
            return None

        def write(self, result: Any) -> None:
            self.results.append(result)

        def close(self) -> tuple[OutputArtifact, ...]:
            return ()

    output = _Collecting()
    mapping = adapter.create_input_mapping(replay_inputs)
    run_inference_session(
        adapter=adapter,
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            device="cpu",
            runtime_options={"pipeline_config": object()},
        ),
        mapping=mapping,
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
        user_inputs=UserInputs(),
        initial_inputs=inference_input_from_replay_inputs(replay_inputs),
        output=output,
        metrics=NullMetricsRecorder(),
    )

    assert len(output.results) == 3
    assert len(pipeline.generate_calls) == 3
    # Each step must receive its own successive slice of the trace. Comparing
    # against the trace directly is the real property; consecutive pose values
    # can repeat because preprocess_example_poses re-expands encoded poses at
    # stride-4 cadence.
    trace_poses = mapping.camera_trace.poses
    received = torch.cat([call["poses"] for call in pipeline.generate_calls])
    assert torch.allclose(received, trace_poses[:6])
