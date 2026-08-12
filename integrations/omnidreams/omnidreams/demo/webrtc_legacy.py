# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Legacy OmniDreams WebRTC compatibility facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from loguru import logger
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.core.distributed.rank_orchestration import distributed_op
from flashdreams.runtime import (
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputCanonicalizer,
    StepRequest,
    StepRequirements,
    StepResult,
    TimeWindow,
    step_requirements_from_request,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    PreparedScenario,
    SessionInfo,
    UserInputWindow,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.serving.webrtc.demo import (
    CreateWebRTCApp,
    RunWebRTCServer,
    serve_webrtc_demo,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import (
    ThreadAffineDistributedWebRTCRuntime,
    WebRTCControlSignal,
)
from flashdreams.serving.webrtc.services import WEBRTC_USER_INPUT_SCHEMA

from .controls import (
    SPARSE_KEY_SEGMENTS_METADATA_KEY,
    WSAD_SUPPORTED_KEYS,
    KeyboardResampler,
    PoseSegment,
)
from .providers import LudusSceneConditioningProvider
from .runtime import OmnidreamsRuntime, OmnidreamsRuntimeOptions
from .spec import (
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    OMNIDREAMS_MODEL_ID,
    OmnidreamsLudusReplayScenario,
)
from .webrtc_config import OmnidreamsWebRTCModelRuntimeConfig

WebRTCRuntimeFactory = Callable[..., Any]
_WEBRTC_SESSION_TOTAL_BLOCKS = 2_147_483_647
_WEBRTC_STEP_REQUEST_KEY = "omnidreams_webrtc_step_request"


class OmnidreamsWebRTCModelRuntimeError(RuntimeError):
    """Raised when the OmniDreams demo runtime is used incorrectly."""


class OmnidreamsWebRTCModelRuntime(
    ThreadAffineDistributedWebRTCRuntime[
        OmnidreamsWebRTCModelRuntimeConfig,
        None,
    ]
):
    """Compatibility WebRTC facade over the shared OmniDreams runtime/session."""

    def __init__(self, *, config: OmnidreamsWebRTCModelRuntimeConfig) -> None:
        super().__init__(
            config=config,
            runtime_error_type=OmnidreamsWebRTCModelRuntimeError,
            thread_name="omnidreams-demo-runtime",
        )
        # The shared WebRTC input source emits normalized runtime events
        # (``key_down``/``key_up``). The Ludus provider consumes sparse
        # resampler metadata on this transitional path, so keep validation
        # aligned with the WebRTC source rather than the replay trace schema.
        self.input_source_schema = WEBRTC_USER_INPUT_SCHEMA
        self.input_canonicalizer = InputCanonicalizer()
        self.input_mapping = _OmnidreamsWebRTCInputMapping()
        self._runtime: OmnidreamsRuntime | None = None
        self._active_provider: LudusSceneConditioningProvider | None = None
        self._active_session: Any | None = None
        self._debug_session: _OmnidreamsHDMapDebugSession | None = None
        self._steady_output_frame_count_value = 1

    def _is_runtime_initialized(self) -> bool:
        return self._runtime is not None

    def _runtime_step_index(self) -> int:
        requirements = self._next_step_requirements_sync()
        if requirements is None:
            return 0
        return requirements.step_index

    def _next_input_frame_count(self) -> int:
        requirements = self._next_step_requirements_sync()
        if requirements is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC session is complete."
            )
        return requirements.input_frame_count

    def _steady_output_frame_count(self) -> int:
        return self._steady_output_frame_count_value

    def _initialize_sync(self) -> None:
        if self._runtime is not None:
            return
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniDreams WebRTC inference.")
        _validate_single_view_pipeline_config(
            pipeline_config_name=self.config.pipeline_config_name,
            pipeline_config=self.config.pipeline_config,
        )
        logger.info(
            "Setting up shared OmniDreams runtime {} on {} for WebRTC.",
            self.config.pipeline_config_name,
            self._device,
        )
        self._runtime = OmnidreamsRuntime(
            config=self._inference_config(),
            options=OmnidreamsRuntimeOptions(
                pipeline_config=self.config.pipeline_config,
                pipeline_factory=self.config.pipeline_factory,
                # WebRTC warms the same long-lived runtime before real browser
                # sessions. Keep prompt/image encoders available for later
                # peer connections until Phase 14 replaces loopback warmup with
                # first-class model/runtime warmup.
                release_oneshot_encoders_after_cache_init=False,
            ),
        )
        self._initialize_video_encoder_sync()

    def _reset_rollout_sync(self, session_input: None = None) -> None:
        del session_input
        self._close_active_session_sync()
        runtime = self._require_runtime()
        scenario = self._session_scenario()
        prepared = PreparedScenario(
            initial_inputs=InferenceInput(global_conditioning={"scenario": scenario}),
            source_schema=self.input_source_schema,
            metadata={
                "conditioning_mode": "ludus-scene-driving",
                "model_id": OMNIDREAMS_MODEL_ID,
                "preset_id": self.config.pipeline_config_name,
            },
        )
        provider = LudusSceneConditioningProvider(
            scenario=prepared,
            config=self._inference_config(),
        )
        try:
            initial_input = provider.prepare_initial_input()
            session = runtime.start_session(initial_input)
        except Exception:
            provider.close()
            raise
        self._active_provider = provider
        if self.config.debug_serve_hdmaps:
            self._debug_session = _OmnidreamsHDMapDebugSession(
                pipeline=runtime.pipeline,
                scenario=scenario,
            )
            self._active_session = self._debug_session
        else:
            self._debug_session = None
            self._active_session = session
        self._steady_output_frame_count_value = _steady_output_frame_count(
            self._active_session,
            fallback_pipeline=runtime.pipeline,
        )

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult:
        pose_segments = cast(list[PoseSegment], segments)
        request = self._next_step_request_sync()
        if request is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC session is complete."
            )
        inputs = self.input_mapping.map_step_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=InferenceInput(
                metadata={
                    SPARSE_KEY_SEGMENTS_METADATA_KEY: tuple(pose_segments),
                    "frame_times": tuple(frame_times),
                    "window_start_s": request.step_index / float(self.config.fps),
                    "window_end_s": (request.step_index + len(frame_times))
                    / float(self.config.fps),
                }
            ),
            request=request,
        )
        return self._step_active_session_sync(inputs)

    def _close_sync(self) -> None:
        self._close_active_session_sync()
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.close()
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    async def start_inference_session(self) -> "_OmnidreamsWebRTCInferenceSession":
        self._require_open_and_initialized()
        if not await self._worker.call(self._has_active_session_sync):
            await self.reset_for_new_session()
        return _OmnidreamsWebRTCInferenceSession(self)

    def _next_step_request_sync(self) -> StepRequest | None:
        requirements = self._next_step_requirements_sync()
        if requirements is None:
            return None
        metadata = dict(requirements.metadata)
        metadata["input_frame_count"] = requirements.input_frame_count
        if requirements.steady_output_frame_count is not None:
            metadata["steady_output_frame_count"] = (
                requirements.steady_output_frame_count
            )
        return StepRequest(
            step_index=requirements.step_index,
            inference_input_schema=requirements.inference_input_schema,
            metadata=metadata,
        )

    def _next_step_requirements_sync(self) -> StepRequirements | None:
        session = self._require_active_session()
        next_requirements = getattr(session, "next_step_requirements", None)
        if callable(next_requirements):
            result = next_requirements()
        else:
            next_request = session.next_step_request()
            if next_request is None:
                return None
            result = step_requirements_from_request(next_request)
        if result is None:
            return None
        if not isinstance(result, StepRequirements):
            raise TypeError(
                "OmniDreams WebRTC session requirements must be StepRequirements, "
                f"got {type(result).__name__}."
            )
        return result

    def _session_info_sync(self) -> SessionInfo:
        return SessionInfo(
            output_layout="bvtchw",
            steady_output_frame_count=self._steady_output_frame_count(),
            metadata={"model_id": OMNIDREAMS_MODEL_ID},
        )

    def _step_active_session_sync(self, inputs: InferenceInput) -> StepResult:
        provider = self._require_active_provider()
        session = self._require_active_session()
        request = _request_from_step_inputs(inputs)
        requirements = step_requirements_from_request(
            request,
            allow_user_input_window=True,
        )
        window = _user_window_from_step_inputs(
            inputs,
            request=request,
            input_frame_count=requirements.input_frame_count,
        )
        prepared = provider.prepare_step(request=requirements, user_window=window)
        if prepared.control.close_session:
            raise OmnidreamsWebRTCModelRuntimeError(
                prepared.control.reason or "OmniDreams WebRTC input is exhausted."
            )
        if prepared.control.reset:
            reset_input = prepared.control.reset_input
            session.reset(reset_input)
            if not prepared.control.provider_already_reset:
                provider.reset(reset_input)
            raise OmnidreamsWebRTCModelRuntimeError(
                prepared.control.reason or "OmniDreams WebRTC session reset requested."
            )
        if prepared.inference_input is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC provider returned no inference input."
            )
        result = session.step(prepared.inference_input)
        if not isinstance(result, StepResult):
            raise TypeError(
                "OmniDreams WebRTC session steps must produce StepResult, got "
                f"{type(result).__name__}."
            )
        return result

    @distributed_op(WebRTCControlSignal.SESSION_STEP)
    def _step_active_session_sync_all_ranks(
        self,
        inputs: InferenceInput,
    ) -> StepResult:
        return self._step_active_session_sync(inputs)

    @distributed_op(WebRTCControlSignal.SESSION_CLOSE)
    def _close_active_session_sync_all_ranks(self) -> None:
        self._close_active_session_sync()

    def _close_active_session_sync(self) -> None:
        session = self._active_session
        provider = self._active_provider
        self._active_session = None
        self._debug_session = None
        self._active_provider = None
        first_error: Exception | None = None
        close_session = getattr(session, "close", None)
        if callable(close_session):
            try:
                close_session()
            except Exception as exc:
                first_error = exc
        if provider is not None:
            try:
                provider.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _has_active_session_sync(self) -> bool:
        return self._active_session is not None and self._active_provider is not None

    def _require_runtime(self) -> OmnidreamsRuntime:
        if self._runtime is None:
            raise OmnidreamsWebRTCModelRuntimeError("Runtime is not initialized.")
        return self._runtime

    def _require_active_session(self) -> Any:
        if self._active_session is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC session is not initialized."
            )
        return self._active_session

    def _require_active_provider(self) -> LudusSceneConditioningProvider:
        if self._active_provider is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC provider is not initialized."
            )
        return self._active_provider

    def _inference_config(self) -> InferenceConfig:
        return InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=self.config.pipeline_config_name,
            device=str(self.config.device),
            seed=self.config.seed,
            runtime_options={"seed": self.config.seed},
        )

    def _session_scenario(self) -> OmnidreamsLudusReplayScenario:
        return OmnidreamsLudusReplayScenario(
            keyboard_events=(),
            scene_dir=self.config.scene_dir,
            scene_uuid=self.config.scene_uuid or DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
            scene_variant=self.config.scene_variant,
            camera_name=self.config.camera_name,
            total_blocks=_WEBRTC_SESSION_TOTAL_BLOCKS,
            pixel_height=self.config.video_height,
            pixel_width=self.config.video_width,
            fps=self.config.fps,
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
        )


class _OmnidreamsWebRTCInputMapping:
    """Carry shared WebRTC window facts into the OmniDreams session facade."""

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        del canonical_schema, inference_input_schema

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del canonical_inputs
        step = dict(inference_input.step)
        step[_WEBRTC_STEP_REQUEST_KEY] = request
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step=step,
            metadata=inference_input.metadata,
        )


class _OmnidreamsWebRTCInferenceSession:
    """Synchronous session proxy consumed by the shared WebRTC compatibility path."""

    def __init__(self, runtime: OmnidreamsWebRTCModelRuntime) -> None:
        self._runtime = runtime
        self._closed = False

    def session_info(self) -> SessionInfo:
        self._require_open()
        return self._runtime._worker.call_blocking(self._runtime._session_info_sync)

    def next_step_requirements(self) -> StepRequirements | None:
        self._require_open()
        return self._runtime._worker.call_blocking(
            self._runtime._next_step_requirements_sync
        )

    def next_step_request(self) -> StepRequest | None:
        self._require_open()
        return self._runtime._worker.call_blocking(
            self._runtime._next_step_request_sync
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self._require_open()
        return self._runtime._worker.call_blocking(
            self._runtime._step_active_session_sync_all_ranks,
            inputs,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._runtime._worker.call_blocking(self._runtime._reset_rollout_sync_all_ranks)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime._worker.call_blocking(
            self._runtime._close_active_session_sync_all_ranks
        )

    def _require_open(self) -> None:
        if self._closed:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC inference session is closed."
            )


class _OmnidreamsHDMapDebugSession:
    """Session-shaped debug path that streams rendered Ludus HDMaps."""

    def __init__(
        self, *, pipeline: Any, scenario: OmnidreamsLudusReplayScenario
    ) -> None:
        self._pipeline = pipeline
        self._scenario = scenario
        self._step_index = 0
        self._closed = False

    def session_info(self) -> SessionInfo:
        return SessionInfo(
            output_layout="bvtchw",
            steady_output_frame_count=self._steady_output_frame_count(),
            metadata={"stream": "hdmap"},
        )

    def next_step_requirements(self) -> StepRequirements | None:
        if self._closed or self._step_index >= self._scenario.total_blocks:
            return None
        return StepRequirements(
            step_index=self._step_index,
            input_frame_count=self._num_frames(self._step_index),
            steady_output_frame_count=self._steady_output_frame_count(),
        )

    def next_step_request(self) -> StepRequest | None:
        requirements = self.next_step_requirements()
        if requirements is None:
            return None
        return StepRequest(
            step_index=requirements.step_index,
            metadata={
                "input_frame_count": requirements.input_frame_count,
                "steady_output_frame_count": requirements.steady_output_frame_count,
            },
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        requirements = self.next_step_requirements()
        if requirements is None:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC debug session is complete."
            )
        hdmap = inputs.step.get("hdmap")
        if not isinstance(hdmap, torch.Tensor):
            raise TypeError("OmniDreams WebRTC debug session requires step['hdmap'].")
        result = StepResult.from_video_chunk(
            step_index=requirements.step_index,
            video_chunk=hdmap.detach(),
            layout="bvtchw",
            metadata={"stream": "hdmap"},
        )
        self._step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._step_index = 0
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def _steady_output_frame_count(self) -> int:
        return self._num_frames(1)

    def _num_frames(self, step_index: int) -> int:
        get_num_frames = getattr(self._pipeline, "get_num_frames", None)
        if not callable(get_num_frames):
            return 1
        return int(get_num_frames(step_index))


def _request_from_step_inputs(inputs: InferenceInput) -> StepRequest:
    request = inputs.step.get(_WEBRTC_STEP_REQUEST_KEY)
    if not isinstance(request, StepRequest):
        raise TypeError(
            "OmniDreams WebRTC step input is missing the shared StepRequest."
        )
    return request


def _user_window_from_step_inputs(
    inputs: InferenceInput,
    *,
    request: StepRequest,
    input_frame_count: int,
) -> UserInputWindow:
    frame_times = _frame_times_from_metadata(inputs.metadata, input_frame_count)
    segments = _segments_from_metadata(inputs.metadata)
    window = request.user_input_window or TimeWindow(
        start_s=float(inputs.metadata.get("window_start_s", 0.0)),
        end_s=float(inputs.metadata.get("window_end_s", frame_times[-1])),
    )
    return UserInputWindow(
        start_s=window.start_s,
        end_s=window.end_s,
        frame_times=frame_times,
        metadata={SPARSE_KEY_SEGMENTS_METADATA_KEY: segments},
    )


def _frame_times_from_metadata(
    metadata: Mapping[str, object],
    input_frame_count: int,
) -> tuple[float, ...]:
    value = metadata.get("frame_times")
    if not isinstance(value, tuple):
        raise OmnidreamsWebRTCModelRuntimeError(
            "OmniDreams WebRTC step input is missing frame_times metadata."
        )
    frame_times = tuple(
        _float_metadata_value(frame_time, label="frame_times") for frame_time in value
    )
    if len(frame_times) != input_frame_count:
        raise OmnidreamsWebRTCModelRuntimeError(
            "OmniDreams WebRTC frame_times length does not match "
            f"input_frame_count={input_frame_count}."
        )
    return frame_times


def _segments_from_metadata(metadata: Mapping[str, object]) -> tuple[PoseSegment, ...]:
    value = metadata.get(SPARSE_KEY_SEGMENTS_METADATA_KEY)
    if not isinstance(value, tuple):
        raise OmnidreamsWebRTCModelRuntimeError(
            "OmniDreams WebRTC step input is missing resampled key segments."
        )
    segments: list[PoseSegment] = []
    for segment in value:
        if not isinstance(segment, tuple) or len(segment) != 3:
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC key segments must be 3-tuples."
            )
        start_s, end_s, keys = segment
        if not isinstance(keys, frozenset | set | tuple | list):
            raise OmnidreamsWebRTCModelRuntimeError(
                "OmniDreams WebRTC key segment keys must be a sequence."
            )
        segments.append(
            (
                _float_metadata_value(start_s, label="segment start"),
                _float_metadata_value(end_s, label="segment end"),
                frozenset(str(key) for key in keys),
            )
        )
    return tuple(segments)


def _float_metadata_value(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OmnidreamsWebRTCModelRuntimeError(
            f"OmniDreams WebRTC {label} metadata must be numeric."
        )
    return float(value)


def _steady_output_frame_count(session: Any, *, fallback_pipeline: Any) -> int:
    session_info = getattr(session, "session_info", None)
    if callable(session_info):
        value = session_info()
        if isinstance(value, SessionInfo) and value.steady_output_frame_count:
            return int(value.steady_output_frame_count)
    get_num_frames = getattr(fallback_pipeline, "get_num_frames", None)
    if callable(get_num_frames):
        return int(get_num_frames(1))
    return 1


def _validate_single_view_pipeline_config(
    *,
    pipeline_config_name: str,
    pipeline_config: Any,
) -> None:
    diffusion_model = getattr(pipeline_config, "diffusion_model", None)
    transformer_cfg = getattr(diffusion_model, "transformer", None)
    if transformer_cfg is None:
        return
    if not isinstance(transformer_cfg, CosmosTransformerConfig):
        raise TypeError(
            "OmniDreams WebRTC requires a CosmosTransformerConfig pipeline."
        )
    if transformer_cfg.num_views != 1:
        raise ValueError(
            "OmniDreams WebRTC supports only single-view configs; "
            f"{pipeline_config_name!r} has num_views={transformer_cfg.num_views}."
        )


def _serve_legacy_omnidreams_webrtc_demo(
    *,
    spec: DemoSpec,
    output: WebRTCOutputSpec,
    runtime_config: OmnidreamsWebRTCModelRuntimeConfig,
    runtime_factory: WebRTCRuntimeFactory,
    world_rank: int,
    create_app_fn: CreateWebRTCApp,
    server_runner: RunWebRTCServer,
) -> object:
    runtime = runtime_factory(config=runtime_config)
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        identity=runtime_config.pipeline_config_name,
        busy_message="An OmniDreams session is already active.",
        warmup_label="OmniDreams WebRTC",
        supported_control_keys=WSAD_SUPPORTED_KEYS,
        fatal_generation_errors=True,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        legacy_segment_resampler_factory=KeyboardResampler,
    )
    from importlib.resources import files

    return serve_webrtc_demo(
        output=output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("omnidreams.demo").joinpath("web"),
            preload_name="OmniDreams",
        ),
        world_rank=world_rank,
        create_app_fn=create_app_fn,
        server_runner=server_runner,
    )


__all__ = [
    "OmnidreamsWebRTCModelRuntime",
    "OmnidreamsWebRTCModelRuntimeError",
    "WebRTCRuntimeFactory",
    "_serve_legacy_omnidreams_webrtc_demo",
]
