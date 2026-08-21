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
from flashdreams.runtime.types import (
    BATCH_INPUT_FRAME_START_METADATA_KEY,
    StepRequirements,
)

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


def test_session_exposes_shared_step_requirements_without_user_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    runtime, session = _session(tmp_path, monkeypatch, pipeline)

    requirements = session.next_step_requirements()
    legacy_request = session.next_step_request()

    assert isinstance(requirements, StepRequirements)
    assert requirements.step_index == 0
    assert requirements.input_frame_count == 2
    assert requirements.steady_output_frame_count == 2
    assert requirements.metadata[BATCH_INPUT_FRAME_START_METADATA_KEY] == 0
    assert requirements.metadata["num_frames"] == 2
    assert requirements.metadata["frame_start"] == 0
    assert legacy_request is not None
    assert legacy_request.user_input_window is not None
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
