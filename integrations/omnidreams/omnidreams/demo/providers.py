# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams model-input providers for shared demo run modes."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from omnidreams.runner import _load_video

from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    load_first_frame_tensor,
)
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo import (
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.demo.session_inputs import ControlDecision
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
    InputField,
    UserInputCapability,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.types import StepRequirements
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
)

from .controls import (
    WSAD_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    KeyboardResampler,
    PoseSegment,
)
from .spec import (
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    OmnidreamsLudusReplayScenario,
    OmnidreamsReplayScenario,
)


class PrecomputedHDMapProvider:
    """Prepare fixed OmniDreams HDMap conditioning for replay-style runs."""

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        config: InferenceConfig,
    ) -> None:
        self._scenario = _precomputed_scenario_from_prepared(scenario)
        self._device = _device_from_config(config)
        self._dtype = torch.bfloat16
        self._frame_start = 0
        self._closed = False
        self.capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=precomputed_hdmap_inference_input_schema(),
        )
        self._hdmap_videos: torch.Tensor | None = self._load_hdmaps()

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        scenario = self._scenario
        first_frames = [
            load_first_frame_tensor(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self._device,
                dtype=self._dtype,
                allow_video=True,
                install_hint=DEFAULT_RUNNER_INSTALL_HINT,
            )
            for path in scenario.first_frame_paths
        ]
        return InferenceInput(
            global_conditioning={
                "scenario": scenario,
                "prompt": [list(scenario.prompts)],
                "first_frame": torch.stack(first_frames, dim=0).unsqueeze(0),
            },
            metadata={"view_names": tuple(scenario.camera_names)},
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del user_window
        self._require_open()
        hdmap_videos = self._require_hdmaps()
        frame_end = self._frame_start + request.input_frame_count
        if frame_end > hdmap_videos.shape[2]:
            return PreparedStep(
                control=ControlDecision(
                    close_session=True,
                    reason="OmniDreams precomputed HDMap input exhausted.",
                )
            )

        frame_start = self._frame_start
        self._frame_start = frame_end
        return PreparedStep(
            inference_input=InferenceInput(
                step={"hdmap": hdmap_videos[:, :, frame_start:frame_end]},
                metadata={
                    "hdmap_frame_start": frame_start,
                    "hdmap_frame_end": frame_end,
                },
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._frame_start = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hdmap_videos = None

    def _load_hdmaps(self) -> torch.Tensor:
        scenario = self._scenario
        videos = [
            _load_video(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self._device,
                dtype=self._dtype,
            )
            for path in scenario.hdmap_video_paths
        ]
        hdmap_videos = torch.stack(videos, dim=0).unsqueeze(0)
        if _is_rank_zero():
            logger.info(
                "Loaded OmniDreams demo HDMaps shape={} views={}",
                tuple(hdmap_videos.shape),
                len(scenario.camera_names),
            )
        return hdmap_videos

    def _require_hdmaps(self) -> torch.Tensor:
        hdmap_videos = self._hdmap_videos
        if hdmap_videos is None:
            raise RuntimeError("OmniDreams precomputed HDMap provider is closed.")
        return hdmap_videos

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams precomputed HDMap provider is closed.")


class LudusSceneConditioningProvider:
    """Render finite Ludus keyboard-driving traces into OmniDreams HDMaps."""

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        config: InferenceConfig,
    ) -> None:
        self._scenario = _ludus_scenario_from_prepared(scenario)
        self._device = _device_from_config(config)
        self._dtype = torch.bfloat16
        self._closed = False
        self._scene: Any | None = None
        self._rasterizer: Any | None = None
        self._pose_integrator: CameraPoseIntegrator | None = None
        self._keyboard_resampler: KeyboardResampler | None = None
        self._next_timestamp_us = 0
        self._step_index = 0
        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=True,
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=precomputed_hdmap_inference_input_schema(),
        )

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        scene = self._ensure_scene_loaded()
        return InferenceInput(
            global_conditioning={
                "scenario": self._scenario,
                "prompt": [[str(scene.prompt)]],
                "first_frame": _initial_rgb_tensor(
                    scene.initial_rgb,
                    device=self._device,
                    dtype=self._dtype,
                ),
            },
            metadata={
                "view_names": self._scenario.camera_names,
                "scene_id": str(getattr(scene, "scene_id", "")),
            },
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        self._require_open()
        scenario = self._scenario
        if request.step_index >= scenario.total_blocks:
            return PreparedStep(
                control=ControlDecision(
                    close_session=True,
                    reason="OmniDreams Ludus replay input exhausted.",
                )
            )

        self._ensure_scene_loaded()
        pose_integrator = self._require_pose_integrator()
        rasterizer = self._require_rasterizer()
        segments, frame_times = self._sample_controls(
            request=request,
            user_window=user_window,
        )
        rig_poses_world = pose_integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        timestamps_us = self._consume_timestamps(request.input_frame_count)
        raster_chunk = rasterizer.render_chunk(
            rig_poses_world=rig_poses_world,
            timestamps_us=timestamps_us,
        )
        hdmap = _condition_frames_tensor(
            raster_chunk.frames,
            device=self._device,
            dtype=self._dtype,
        )
        self._step_index += 1
        return PreparedStep(
            inference_input=InferenceInput(
                step={"hdmap": hdmap},
                metadata={
                    "frame_timestamps_us": tuple(int(t) for t in timestamps_us),
                    "keyboard_segments": _segments_metadata(segments),
                    "camera_name": scenario.camera_name,
                    "scene_uuid": scenario.scene_uuid,
                },
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        if self._scene is not None:
            self._reset_driving_state(self._scene)
        else:
            self._step_index = 0
            self._next_timestamp_us = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rasterizer = self._rasterizer
        self._rasterizer = None
        self._scene = None
        self._pose_integrator = None
        self._keyboard_resampler = None
        _close_rasterizer(rasterizer)

    def _ensure_scene_loaded(self) -> Any:
        if self._scene is not None:
            return self._scene
        scenario = self._scenario
        scene_path = _resolve_ludus_scene_path(scenario)
        scene = _load_ludus_scene_bundle(scenario, scene_path)
        rasterizer = _new_ludus_rasterizer(scenario)
        try:
            rasterizer.load_scene(scene)
        except Exception:
            with contextlib.suppress(Exception):
                _close_rasterizer(rasterizer)
            raise
        self._scene = scene
        self._rasterizer = rasterizer
        self._reset_driving_state(scene)
        if _is_rank_zero():
            logger.info(
                "Loaded OmniDreams Ludus replay scene={} camera={} trace_events={}",
                scene_path,
                scenario.camera_name,
                len(scenario.keyboard_events),
            )
        return scene

    def _reset_driving_state(self, scene: Any) -> None:
        scenario = self._scenario
        pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=scenario.move_speed_per_s,
            rotate_speed_rad_per_s=scenario.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        pose_integrator.reset(np.asarray(scene.initial_rig_to_world, dtype=np.float32))
        keyboard_resampler = KeyboardResampler(
            fps=float(scenario.fps),
            supported_keys=WSAD_SUPPORTED_KEYS,
        )
        for event in scenario.keyboard_events:
            keyboard_resampler.on_edge(
                arrival_t=event.timestamp_s,
                event=event.event,
                key=event.key,
            )
        self._pose_integrator = pose_integrator
        self._keyboard_resampler = keyboard_resampler
        self._next_timestamp_us = int(scene.initial_timestamp_us)
        self._step_index = 0

    def _sample_controls(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> tuple[list[PoseSegment], list[float]]:
        frame_times = list(user_window.frame_times)
        if frame_times and len(frame_times) != request.input_frame_count:
            raise RuntimeError(
                "OmniDreams Ludus realtime window frame_times length does not "
                "match the requested input frame count."
            )
        resampler = self._require_keyboard_resampler()
        self._advance_skipped_input_state(user_window=user_window, resampler=resampler)
        # Realtime/WebRTC windows carry explicit frame times on the driver's
        # clock. Batch replay windows do not, so the provider-owned trace
        # resampler must keep advancing from its prior chunk instead.
        if frame_times:
            resampler.next_chunk_start_v = user_window.start_s
        self._record_keyboard_events(resampler, user_window.inputs)
        segments, sampled_frame_times = resampler.sample_chunk(
            request.input_frame_count
        )
        if not frame_times:
            frame_times = sampled_frame_times
        return segments, frame_times

    def _advance_skipped_input_state(
        self,
        *,
        user_window: UserInputWindow,
        resampler: KeyboardResampler,
    ) -> None:
        skipped_inputs = user_window.metadata.get(WEBRTC_SKIPPED_INPUTS_METADATA_KEY)
        skipped_window = user_window.metadata.get(WEBRTC_SKIPPED_WINDOW_METADATA_KEY)
        if not isinstance(skipped_inputs, UserInputs):
            return
        if not isinstance(skipped_window, tuple) or len(skipped_window) != 2:
            return
        start_value, end_value = skipped_window
        if not isinstance(start_value, int | float) or not isinstance(
            end_value,
            int | float,
        ):
            return
        start_s = float(start_value)
        end_s = float(end_value)
        if end_s <= start_s:
            return
        resampler.next_chunk_start_v = start_s
        self._record_keyboard_events(resampler, skipped_inputs)
        resampler.advance_to(end_s)

    @staticmethod
    def _record_keyboard_events(
        resampler: KeyboardResampler,
        inputs: UserInputs,
    ) -> None:
        for event in inputs.events:
            if event.event_type not in {"key_down", "key_up", "keydown", "keyup"}:
                continue
            key = event.payload.get("key")
            if not isinstance(key, str):
                continue
            resampler.on_edge(
                arrival_t=event.timestamp_s,
                event="keydown"
                if event.event_type in {"key_down", "keydown"}
                else "keyup",
                key=key,
            )

    def _consume_timestamps(self, num_frames: int) -> np.ndarray:
        step_us = int(round(1_000_000 / float(self._scenario.fps)))
        timestamps = np.array(
            [
                self._next_timestamp_us + frame_index * step_us
                for frame_index in range(num_frames)
            ],
            dtype=np.int64,
        )
        self._next_timestamp_us += num_frames * step_us
        return timestamps

    def _require_rasterizer(self) -> Any:
        if self._rasterizer is None:
            raise RuntimeError("OmniDreams Ludus rasterizer is not initialized.")
        return self._rasterizer

    def _require_pose_integrator(self) -> CameraPoseIntegrator:
        if self._pose_integrator is None:
            raise RuntimeError("OmniDreams Ludus pose integrator is not initialized.")
        return self._pose_integrator

    def _require_keyboard_resampler(self) -> KeyboardResampler:
        if self._keyboard_resampler is None:
            raise RuntimeError(
                "OmniDreams Ludus keyboard resampler is not initialized."
            )
        return self._keyboard_resampler

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams Ludus conditioning provider is closed.")


def keyboard_driving_user_input_schema() -> UserInputSchema:
    return UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="keydown",
                input_modality="keyboard",
                payload_fields=frozenset({"key"}),
                description="Keyboard key press edge.",
            ),
            UserInputCapability(
                event_type="keyup",
                input_modality="keyboard",
                payload_fields=frozenset({"key"}),
                description="Keyboard key release edge.",
            ),
        ),
        description="Recorded or realtime WSAD keyboard driving controls.",
    )


def precomputed_hdmap_inference_input_schema() -> InferenceInputSchema:
    return InferenceInputSchema(
        global_conditioning_fields=(
            InputField(
                name="prompt",
                input_modality="omnidreams/prompt",
                description="OmniDreams prompt batch.",
            ),
            InputField(
                name="first_frame",
                input_modality="video/frame",
                description="Initial OmniDreams conditioning frame tensor.",
            ),
            InputField(
                name="scenario",
                required=False,
                input_modality="omnidreams/replay-scenario",
                description="Resolved OmniDreams replay scenario metadata.",
            ),
        ),
        step_fields=(
            InputField(
                name="hdmap",
                input_modality="omnidreams/hdmap-video",
                frequency_consumed="per_step",
                description="Per-step HDMap conditioning chunk.",
            ),
        ),
    )


def _precomputed_scenario_from_prepared(
    scenario: PreparedScenario,
) -> OmnidreamsReplayScenario:
    value = scenario.initial_inputs.global_conditioning.get("scenario")
    if not isinstance(value, OmnidreamsReplayScenario):
        raise TypeError(
            "OmniDreams precomputed HDMap provider requires "
            "initial_inputs.global_conditioning['scenario'] to be an "
            "OmnidreamsReplayScenario."
        )
    return value


def _ludus_scenario_from_prepared(
    scenario: PreparedScenario,
) -> OmnidreamsLudusReplayScenario:
    value = scenario.initial_inputs.global_conditioning.get("scenario")
    if not isinstance(value, OmnidreamsLudusReplayScenario):
        raise TypeError(
            "OmniDreams Ludus conditioning provider requires "
            "initial_inputs.global_conditioning['scenario'] to be an "
            "OmnidreamsLudusReplayScenario."
        )
    return value


def _resolve_ludus_scene_path(scenario: OmnidreamsLudusReplayScenario) -> Path:
    if scenario.scene_path is not None:
        if not scenario.scene_path.exists():
            raise FileNotFoundError(
                f"OmniDreams Ludus scene_path missing: {scenario.scene_path}"
            )
        return scenario.scene_path
    if scenario.scene_dir is not None:
        return _resolve_local_ludus_scene_path(scenario)

    from omnidreams.scenes import hf_hub_download_scene  # noqa: PLC0415

    return hf_hub_download_scene(
        scenario.scene_uuid or DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
        scenario.scene_variant,
    )


def _resolve_local_ludus_scene_path(scenario: OmnidreamsLudusReplayScenario) -> Path:
    scene_dir = scenario.scene_dir
    if scene_dir is None:
        raise RuntimeError("OmniDreams Ludus scene_dir is unexpectedly unset.")
    if scene_dir.is_file():
        return scene_dir
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"OmniDreams Ludus scene_dir missing: {scene_dir}")

    candidates = _local_ludus_scene_candidates(scenario)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    archives = sorted(scene_dir.glob("*.usdz"))
    if scenario.scene_uuid is None and len(archives) == 1:
        return archives[0]
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"No OmniDreams Ludus USDZ scene archive found in {scene_dir}. "
        f"Expected one of: {expected}."
    )


def _local_ludus_scene_candidates(
    scenario: OmnidreamsLudusReplayScenario,
) -> tuple[Path, ...]:
    scene_dir = scenario.scene_dir
    if scene_dir is None or scenario.scene_uuid is None:
        return ()

    from omnidreams.scenes import (  # noqa: PLC0415
        normalise_scene_uuid,
        scene_variant_suffix,
    )

    bare_uuid = normalise_scene_uuid(scenario.scene_uuid)
    suffix = scene_variant_suffix(scenario.scene_variant)
    stems = [f"clipgt-{bare_uuid}{suffix}", f"{bare_uuid}{suffix}"]
    if suffix:
        stems.extend((f"clipgt-{bare_uuid}", bare_uuid))
    return tuple(scene_dir / f"{stem}.usdz" for stem in dict.fromkeys(stems))


def _load_ludus_scene_bundle(
    scenario: OmnidreamsLudusReplayScenario,
    scene_path: Path,
) -> Any:
    from omnidreams.interactive_drive.scene_loader import (  # noqa: PLC0415
        load_scene_bundle,
    )

    return load_scene_bundle(
        scene_path=scene_path,
        camera_name=scenario.camera_name,
        variant=scenario.scene_variant,
        prompt_override=scenario.prompt,
        raster=_ludus_raster_config(scenario),
    )


def _new_ludus_rasterizer(scenario: OmnidreamsLudusReplayScenario) -> Any:
    from omnidreams.interactive_drive.rasterizer import (  # noqa: PLC0415
        LudusConditionRasterizer,
    )

    return LudusConditionRasterizer(_ludus_raster_config(scenario), bev=None)


def _ludus_raster_config(scenario: OmnidreamsLudusReplayScenario) -> Any:
    from omnidreams.interactive_drive.config import RasterConfig  # noqa: PLC0415

    return RasterConfig(
        width=scenario.pixel_width,
        height=scenario.pixel_height,
    )


def _initial_rgb_tensor(
    frame: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.from_numpy(_rgb_hwc_uint8(frame))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    return _to_model_range(tensor, device=device, dtype=dtype)


def _condition_frames_tensor(
    frames: tuple[object, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cuda_video = _condition_cuda_video(frames)
    if cuda_video is not None:
        tensor = cuda_video.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
        return _to_model_range(tensor, device=device, dtype=dtype)
    video = np.stack(
        [_rgb_hwc_uint8(_frame_rgb(frame)) for frame in frames],
        axis=0,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(video))
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
    return _to_model_range(tensor, device=device, dtype=dtype)


def _condition_cuda_video(frames: tuple[object, ...]) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    for frame in frames:
        to_cuda_tensor = getattr(_frame_rgb(frame), "to_cuda_tensor", None)
        if not callable(to_cuda_tensor):
            return None
        try:
            tensor = to_cuda_tensor()
        except RuntimeError:
            return None
        if (
            not torch.is_tensor(tensor)
            or not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or tensor.ndim != 3
            or tensor.shape[-1] < 3
        ):
            return None
        tensors.append(tensor[..., :3])
    return torch.stack(tensors, dim=0)


def _frame_rgb(frame: object) -> object:
    return getattr(frame, "rgb_host_uint8", frame)


def _rgb_hwc_uint8(frame: object) -> np.ndarray:
    if torch.is_tensor(frame):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            "OmniDreams Ludus rendered frames must be HWC RGB/RGBA uint8 arrays."
        )
    return np.ascontiguousarray(np.array(array[..., :3], dtype=np.uint8, copy=True))


def _to_model_range(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype) / 127.5 - 1.0


def _segments_metadata(
    segments: list[PoseSegment],
) -> tuple[tuple[float, float, tuple[str, ...]], ...]:
    return tuple(
        (float(start), float(end), tuple(sorted(keys))) for start, end, keys in segments
    )


def _close_rasterizer(rasterizer: Any | None) -> None:
    if rasterizer is None:
        return
    close = getattr(rasterizer, "cleanup", None) or getattr(
        rasterizer,
        "close",
        None,
    )
    if callable(close):
        close()


def _device_from_config(config: InferenceConfig) -> torch.device:
    if dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device(config.device or "cuda")


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


__all__ = [
    "LudusSceneConditioningProvider",
    "PrecomputedHDMapProvider",
    "keyboard_driving_user_input_schema",
    "precomputed_hdmap_inference_input_schema",
]
