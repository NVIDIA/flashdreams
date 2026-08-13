# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral text-to-video demo shell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from flashdreams.demo import (
    Application,
    DemoAdapterApplication,
    FileOutputSink,
    Runner,
    create_replay_io_handler,
)
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
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    OutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.demo.outputs import SessionInfo
from flashdreams.runtime.demo.run_modes import RunResult
from flashdreams.runtime.demo.session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VModelConfig:
    """One integration-owned prompt-only text-to-video model entry."""

    model_id: str
    preset_id: str | None
    pipeline: Any
    prompt: str
    total_blocks: int
    pixel_height: int
    pixel_width: int
    fps: int
    runtime_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("T2VModelConfig.model_id must be non-empty.")
        if self.preset_id is not None and not self.preset_id.strip():
            raise ValueError("T2VModelConfig.preset_id must be non-empty when set.")
        if not self.prompt.strip():
            raise ValueError("T2VModelConfig.prompt must be non-empty.")
        _validate_positive_int(self.total_blocks, name=FIELD_TOTAL_BLOCKS)
        _validate_positive_int(self.pixel_height, name=FIELD_PIXEL_HEIGHT)
        _validate_positive_int(self.pixel_width, name=FIELD_PIXEL_WIDTH)
        _validate_positive_int(self.fps, name=FIELD_FPS)
        object.__setattr__(
            self, "runtime_options", freeze_mapping(self.runtime_options)
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VRunDefaults:
    """Launch-time overrides shared by T2V replay and WebRTC modes."""

    prompt: str | None = None
    total_blocks: int | None = None
    pixel_height: int | None = None
    pixel_width: int | None = None
    fps: int | None = None
    device: str = "cuda"
    compile: bool | None = None
    runtime_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("T2VRunDefaults.device must be non-empty.")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("T2VRunDefaults.prompt must be non-empty when set.")
        _validate_optional_positive_int(self.total_blocks, name=FIELD_TOTAL_BLOCKS)
        _validate_optional_positive_int(self.pixel_height, name=FIELD_PIXEL_HEIGHT)
        _validate_optional_positive_int(self.pixel_width, name=FIELD_PIXEL_WIDTH)
        _validate_optional_positive_int(self.fps, name=FIELD_FPS)
        object.__setattr__(
            self, "runtime_options", freeze_mapping(self.runtime_options)
        )


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

    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name=FIELD_PROMPT),),
        description="Text-to-video prompt and rollout settings.",
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, model: T2VModelConfig) -> None:
        self.model = model

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "webrtc")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null", "webrtc")

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model.model_id:
            raise ValueError(
                f"Expected model_id={self.model.model_id!r}, got {config.model_id!r}."
            )
        if (
            self.model.preset_id is not None
            and config.preset_id != self.model.preset_id
        ):
            raise ValueError(
                f"Expected preset_id={self.model.preset_id!r}, "
                f"got {config.preset_id!r}."
            )
        for name, expected in self.model.runtime_options.items():
            if config.runtime_options.get(name) != expected:
                raise ValueError(
                    f"Expected runtime option {name}={expected!r}, "
                    f"got {config.runtime_options.get(name)!r}."
                )

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        scenario = _scenario_from_value(spec.scenario, self.model)
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
        return T2VRuntime(config=config, model=self.model)

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

    def __init__(self, *, config: InferenceConfig, model: T2VModelConfig) -> None:
        self.config = config
        self.model = model
        pipeline_config = model.pipeline
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
        # Transitional support for the existing WebRTC download endpoint.
        # Replay modes still deliver primary output through the shared OutputSink.
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
                "T2V dimensions must be divisible by the decoder spatial "
                "compression ratio."
            )
        self._cache = pipeline.initialize_cache(
            text=[scenario.prompt],
            image=None,
            height=scenario.pixel_height // ratio,
            width=scenario.pixel_width // ratio,
        )

    def session_info(self) -> SessionInfo:
        return SessionInfo(
            output_layout="tchw",
            metadata={
                FIELD_PROMPT: self.scenario.prompt,
                FIELD_TOTAL_BLOCKS: self.scenario.total_blocks,
                FIELD_PIXEL_HEIGHT: self.scenario.pixel_height,
                FIELD_PIXEL_WIDTH: self.scenario.pixel_width,
                FIELD_FPS: self.scenario.fps,
            },
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
            metadata={FIELD_PROMPT: self.scenario.prompt},
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


def create_t2v_application(
    *,
    model: T2VModelConfig,
    defaults: T2VRunDefaults | None = None,
    input_mode: str = "replay",
    output: OutputSpec | None = None,
) -> Application:
    """Create a public application for one integration-owned T2V model."""
    output = output or NullOutputSpec()
    spec = create_t2v_spec(
        model=model,
        defaults=defaults,
        input_mode=input_mode,
        output=output,
    )
    return DemoAdapterApplication(adapter=T2VDemoAdapter(model=model), spec=spec)


def model_config_from_runner(
    *,
    model_id: str,
    runner: Any,
    runtime_options: Mapping[str, Any] | None = None,
) -> T2VModelConfig:
    """Create a T2V model config from an integration-owned runner config."""
    return T2VModelConfig(
        model_id=model_id,
        preset_id=str(runner.runner_name),
        pipeline=runner.pipeline,
        prompt=str(getattr(runner, FIELD_PROMPT)),
        total_blocks=_int_value(getattr(runner, FIELD_TOTAL_BLOCKS, 1)),
        pixel_height=_int_value(getattr(runner, FIELD_PIXEL_HEIGHT)),
        pixel_width=_int_value(getattr(runner, FIELD_PIXEL_WIDTH)),
        fps=_int_value(getattr(runner, FIELD_FPS)),
        runtime_options=runtime_options or {},
    )


def create_t2v_spec(
    *,
    model: T2VModelConfig,
    defaults: T2VRunDefaults | None = None,
    input_mode: str,
    output: OutputSpec,
) -> DemoSpec:
    """Build the shared runtime spec for one T2V run."""
    defaults = defaults or T2VRunDefaults()
    return DemoSpec(
        model_id=model.model_id,
        preset_id=model.preset_id,
        input_mode=input_mode,
        scenario=t2v_scenario_mapping(model=model, defaults=defaults),
        output=output,
        config=InferenceConfig(
            model_id=model.model_id,
            preset_id=model.preset_id,
            device=defaults.device,
            compile=defaults.compile,
            runtime_options={
                **model.runtime_options,
                **defaults.runtime_options,
            },
        ),
    )


def run_t2v_replay_application(
    *,
    model: T2VModelConfig,
    defaults: T2VRunDefaults | None,
    output: Mp4OutputSpec | NullOutputSpec,
) -> RunResult:
    """Run finite T2V replay through the public runner and replay IO handler."""
    output_sink = None
    if isinstance(output, Mp4OutputSpec):
        output_sink = FileOutputSink(
            output_path=Path(output.path),
            fps=output.fps,
            output_layout=output.output_layout,
        )
    result = Runner(
        io_handler=create_replay_io_handler(output_sink=output_sink),
        app=create_t2v_application(model=model, defaults=defaults, output=output),
    ).run()
    if result.status != "completed":
        reason = result.reason or str(result.error) or "T2V replay failed."
        raise RuntimeError(reason)
    return result


def t2v_scenario_mapping(
    *, model: T2VModelConfig, defaults: T2VRunDefaults | None = None
) -> dict[str, object]:
    """Return runtime scenario values after applying launch overrides."""
    defaults = defaults or T2VRunDefaults()
    return {
        FIELD_PROMPT: model.prompt if defaults.prompt is None else defaults.prompt,
        FIELD_TOTAL_BLOCKS: (
            model.total_blocks
            if defaults.total_blocks is None
            else defaults.total_blocks
        ),
        FIELD_PIXEL_HEIGHT: (
            model.pixel_height
            if defaults.pixel_height is None
            else defaults.pixel_height
        ),
        FIELD_PIXEL_WIDTH: (
            model.pixel_width if defaults.pixel_width is None else defaults.pixel_width
        ),
        FIELD_FPS: model.fps if defaults.fps is None else defaults.fps,
    }


def _scenario_from_value(value: Any, model: T2VModelConfig) -> T2VScenario:
    source = value if isinstance(value, dict) else {}
    prompt = str(source.get(FIELD_PROMPT, model.prompt)).strip()
    if not prompt:
        raise ValueError("A non-empty text-to-video prompt is required.")
    return T2VScenario(
        prompt=prompt,
        total_blocks=_int_value(source.get(FIELD_TOTAL_BLOCKS, model.total_blocks)),
        pixel_height=_int_value(source.get(FIELD_PIXEL_HEIGHT, model.pixel_height)),
        pixel_width=_int_value(source.get(FIELD_PIXEL_WIDTH, model.pixel_width)),
        fps=_int_value(source.get(FIELD_FPS, model.fps)),
    )


def _scenario_from_inputs(inputs: InferenceInput) -> T2VScenario:
    source = inputs.global_conditioning
    return T2VScenario(
        prompt=str(source[FIELD_PROMPT]),
        total_blocks=_int_value(source[FIELD_TOTAL_BLOCKS]),
        pixel_height=_int_value(source[FIELD_PIXEL_HEIGHT]),
        pixel_width=_int_value(source[FIELD_PIXEL_WIDTH]),
        fps=_int_value(source[FIELD_FPS]),
    )


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Expected an integer, not bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(
        f"Expected an integer-compatible value, got {type(value).__name__}."
    )


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def _validate_optional_positive_int(value: int | None, *, name: str) -> None:
    if value is None:
        return
    _validate_positive_int(value, name=name)


__all__ = [
    "FIELD_FPS",
    "FIELD_PIXEL_HEIGHT",
    "FIELD_PIXEL_WIDTH",
    "FIELD_PROMPT",
    "FIELD_TOTAL_BLOCKS",
    "T2VDemoAdapter",
    "T2VInputProvider",
    "T2VModelConfig",
    "T2VRunDefaults",
    "T2VRuntime",
    "T2VScenario",
    "T2VSession",
    "create_t2v_application",
    "create_t2v_spec",
    "model_config_from_runner",
    "run_t2v_replay_application",
    "t2v_scenario_mapping",
]
