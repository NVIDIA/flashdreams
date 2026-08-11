# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime API adapter for the T2V demo's integration pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    ModelAdapter,
    StepRequest,
)
from flashdreams.runtime.demo import DemoSpec, PreparedScenario
from flashdreams.runtime.demo.session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

from .backends import T2VBackend, resolve_backend

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VScenario:
    """Prompt and output geometry for a finite text-to-video rollout."""

    prompt: str
    total_blocks: int
    pixel_height: int
    pixel_width: int
    fps: int


class T2VDemoAdapter(ModelAdapter):
    """Model adapter shared by replay and WebRTC T2V launch paths."""

    model_id = "flashdreams-t2v"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name=FIELD_PROMPT),),
        description="Text-to-video prompt and rollout settings.",
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, backend: T2VBackend) -> None:
        self.backend = backend

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "webrtc")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null", "webrtc")

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"Expected model_id={self.model_id!r}, got {config.model_id!r}."
            )
        if config.runtime_options.get("backend") != self.backend.key:
            raise ValueError("T2V runtime backend does not match its demo adapter.")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        scenario = _scenario_from_value(spec.scenario, self.backend)
        return PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={
                    FIELD_PROMPT: scenario.prompt,
                    FIELD_TOTAL_BLOCKS: scenario.total_blocks,
                    FIELD_PIXEL_HEIGHT: scenario.pixel_height,
                    FIELD_PIXEL_WIDTH: scenario.pixel_width,
                    FIELD_FPS: scenario.fps,
                }
            )
        )

    def create_runtime(self, config: InferenceConfig) -> "T2VRuntime":
        self.validate_config(config)
        return T2VRuntime(config=config, backend=self.backend)

    def create_model_input_provider(
        self, spec: DemoSpec, scenario: PreparedScenario
    ) -> "T2VInputProvider":
        """Supply fixed prompt conditioning to every shared-demo step."""
        del spec
        return T2VInputProvider(initial_inputs=scenario.initial_inputs)


class T2VInputProvider:
    """No-control input provider for finite prompt-only generation."""

    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        supports_recorded_input=True,
        deterministic_given_inputs=True,
    )

    def __init__(self, *, initial_inputs: InferenceInput) -> None:
        self._initial_inputs = initial_inputs

    def prepare_initial_input(self) -> InferenceInput:
        return self._initial_inputs

    def prepare_step(
        self, *, request: Any, user_window: UserInputWindow
    ) -> PreparedStep:
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            self._initial_inputs = inputs

    def close(self) -> None:
        pass


class T2VRuntime:
    """One heavyweight selected pipeline, reusable across demo sessions."""

    def __init__(self, *, config: InferenceConfig, backend: T2VBackend) -> None:
        self.config = config
        self.backend = backend
        runner = backend.resolve_runner(config.preset_id)
        pipeline_config = runner.pipeline
        if config.compile is not None:
            from flashdreams.infra.config import derive_config

            pipeline_config = derive_config(
                base_config=pipeline_config,
                diffusion_model={"transformer": {"compile_network": config.compile}},
            )
        self.pipeline = pipeline_config.setup().to(config.device or "cuda").eval()
        self._latest_artifact: tuple[Path, T2VScenario] | None = None

    def blocks_for_duration(self, duration_s: float, *, fps: int) -> int:
        """Return enough autoregressive chunks to reach the requested duration."""
        target_frames = int(duration_s * fps)
        frames = 0
        index = 0
        while frames < target_frames:
            frames += int(self.pipeline.get_num_output_frames(index))
            index += 1
        return index

    def record_artifact(self, path: Path, scenario: T2VScenario) -> None:
        self._latest_artifact = (path, scenario)

    @property
    def latest_artifact(self) -> tuple[Path, T2VScenario] | None:
        return self._latest_artifact

    def start_session(self, inputs: InferenceInput) -> "T2VSession":
        return T2VSession(
            pipeline=self.pipeline, scenario=_scenario_from_inputs(inputs), runtime=self
        )

    def close(self) -> None:
        close = getattr(self.pipeline, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class T2VSession(InferenceSession):
    """A cache-isolated T2V session that yields chunks as they are generated."""

    def __init__(
        self, *, pipeline: Any, scenario: T2VScenario, runtime: T2VRuntime
    ) -> None:
        self.pipeline = pipeline
        self.scenario = scenario
        self._runtime = runtime
        self._artifact_path = Path("outputs/t2v-webrtc") / f"{uuid4()}.mp4"
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_output = Mp4VideoOutputTarget(
            output_path=self._artifact_path, fps=scenario.fps, output_layout="tchw"
        )
        self._artifact_output.open()
        self._step_index = 0
        self._closed = False
        self._output_stream = VideoOutputStream(
            postprocess_stream=None, output_layout="tchw"
        )
        assert isinstance(pipeline.decoder, StreamingVideoDecoder)
        ratio = pipeline.decoder.spatial_compression_ratio
        if scenario.pixel_height % ratio or scenario.pixel_width % ratio:
            raise ValueError(
                "T2V dimensions must be divisible by the decoder spatial compression ratio."
            )
        self._cache = pipeline.initialize_cache(
            text=[scenario.prompt],
            image=None,
            height=scenario.pixel_height // ratio,
            width=scenario.pixel_width // ratio,
        )

    def next_step_request(self) -> StepRequest | None:
        if self._closed or self._step_index >= self.scenario.total_blocks:
            return None
        return StepRequest(step_index=self._step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        if self._closed:
            raise RuntimeError("T2V session is closed.")
        index = self._step_index
        video = self.pipeline.generate(autoregressive_index=index, cache=self._cache)
        stats = self.pipeline.finalize(autoregressive_index=index, cache=self._cache)
        self._step_index += 1
        result = self._output_stream.process(
            video,
            autoregressive_index=index,
            metrics=stats,
            metadata={"prompt": self.scenario.prompt},
        )
        self._artifact_output.write(result)
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None and _scenario_from_inputs(inputs) != self.scenario:
            raise ValueError(
                "Create a new T2V session to change the prompt or dimensions."
            )
        raise RuntimeError(
            "T2V sessions are finite; create a new session instead of reset()."
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        artifacts = self._artifact_output.close()
        if artifacts:
            self._runtime.record_artifact(self._artifact_path, self.scenario)


def _scenario_from_value(value: Any, backend: T2VBackend) -> T2VScenario:
    runner = backend.resolve_runner()
    source = value if isinstance(value, dict) else {}
    prompt = str(source.get(FIELD_PROMPT, getattr(runner, FIELD_PROMPT, ""))).strip()
    if not prompt:
        raise ValueError("A non-empty text-to-video prompt is required.")
    return T2VScenario(
        prompt=prompt,
        total_blocks=int(
            source.get(FIELD_TOTAL_BLOCKS, getattr(runner, FIELD_TOTAL_BLOCKS, 1))
        ),
        pixel_height=int(
            source.get(FIELD_PIXEL_HEIGHT, getattr(runner, FIELD_PIXEL_HEIGHT, 480))
        ),
        pixel_width=int(
            source.get(FIELD_PIXEL_WIDTH, getattr(runner, FIELD_PIXEL_WIDTH, 832))
        ),
        fps=int(source.get(FIELD_FPS, getattr(runner, FIELD_FPS, 16))),
    )


def _scenario_from_inputs(inputs: InferenceInput) -> T2VScenario:
    source = inputs.global_conditioning
    return T2VScenario(
        prompt=str(source[FIELD_PROMPT]),
        total_blocks=int(source[FIELD_TOTAL_BLOCKS]),
        pixel_height=int(source[FIELD_PIXEL_HEIGHT]),
        pixel_width=int(source[FIELD_PIXEL_WIDTH]),
        fps=int(source[FIELD_FPS]),
    )


def make_adapter(backend: str) -> T2VDemoAdapter:
    """Build an adapter from a CLI/UI backend key."""
    return T2VDemoAdapter(backend=resolve_backend(backend))
