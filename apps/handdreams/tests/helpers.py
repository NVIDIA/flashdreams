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

"""Model-independent HandDreams application, rollout, and controller test doubles."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from flashdreams_controller_input.events import (
    ControllerFrameEvent,
    ControllerSnapshot,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    PreparedControlWindow,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
)
from handdreams.teleop import PreparedTeleopWindow

from flashdreams.infra.postprocess import (
    VideoPostProcessorConfig,
    VideoSpec,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass(kw_only=True)
class _ScalePostprocessConfig(VideoPostProcessorConfig):
    scale: int = 2

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        return replace(
            input_spec,
            width=input_spec.width * self.scale,
            height=input_spec.height * self.scale,
        )


class _FakePostprocessStream:
    created: list["_FakePostprocessStream"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.prepared: list[VideoSpec] = []
        self.processed: list[int] = []
        self.resets = 0
        self.finished = 0
        self.created.append(self)

    def prepare(self, spec: VideoSpec) -> None:
        self.prepared.append(spec)

    def process(self, output: torch.Tensor, *, autoregressive_index: int):
        self.processed.append(autoregressive_index)
        return output.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    def reset(self) -> None:
        self.resets += 1

    def finish(self):
        self.finished += 1
        return None


def _event(frame_id: int, *, trigger: float = 0.0):
    controller = ControllerSnapshot(
        axes=(0.25, -0.5),
        trigger=trigger,
        grip=0.2,
        position=(float(frame_id), 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        tracked=True,
        battery=0.75,
    )
    return ControllerFrameEvent(
        timestamp=np.uint64(frame_id),
        action="state",
        frame_id=frame_id,
        client_capture_ns=frame_id * 1_000_000 + 1,
        pose_convention=OPENXR_LOCAL_GRIP_POSE_CONVENTION,
        left=replace(controller, primary_button=True),
        right=replace(controller, position=(float(frame_id) + 0.4, 0.0, 0.0)),
    )


def _raw_frame(frame_id: int) -> dict[str, Any]:
    return {
        "source_frame_id": frame_id,
        "client_capture_ns": frame_id * 1_000_000 + 1,
        "left": {
            "position": [float(frame_id), 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "grip": 0.0,
            "tracked": True,
        },
        "right": {
            "position": [0.4, float(frame_id), 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.25,
            "grip": 0.0,
            "tracked": True,
        },
    }


class _FakePipeline:
    def __init__(self) -> None:
        self.starts = 0
        self.resets = 0
        self.closes = 0
        self.calibrations = 0
        self.calibration_bases: list[Any] = []
        self.windows: list[list[int]] = []
        self.commits = 0
        self.aborts = 0

    def start(self) -> None:
        self.starts += 1

    def transform(self, frame: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(frame)
        result["pose_convention"] = HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
        return result

    def calibrate(self, neutral_frame: dict[str, Any], **_kwargs: Any) -> None:
        self.calibrations += 1
        self.calibration_bases.append(_kwargs.get("participant_basis"))

    def prepare_window(
        self, frames: list[dict[str, Any]], *, command_sequence: int
    ) -> PreparedTeleopWindow:
        self.windows.append([int(frame["source_frame_id"]) for frame in frames])
        targets: list[dict[str, Any]] = []
        for frame in frames:
            target: dict[str, Any] = {}
            for side in ("left", "right"):
                controller = frame[side]
                target[side] = {
                    "position": list(controller["position"]),
                    "orientation": list(controller["orientation"]),
                    "gripper_open": 1.0 - float(controller["trigger"]),
                }
            targets.append(target)
        return PreparedTeleopWindow(
            frames=targets,
            receipt=PreparedControlWindow(
                frames=targets,
                epoch=1,
                command_sequence=command_sequence,
                token=f"token-{command_sequence}",
            ),
        )

    def commit(self, prepared: PreparedTeleopWindow) -> None:
        self.commits += 1

    def abort(self, prepared: PreparedTeleopWindow) -> None:
        self.aborts += 1

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closes += 1


class _FakeInferenceCache:
    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []


class _FakeInferencePipeline:
    def __init__(self) -> None:
        self.actions_per_step = 1
        self.prepared_scenes: list[Any] = []
        self.caches: list[_FakeInferenceCache] = []
        self.generate_calls: list[tuple[int, _FakeInferenceCache]] = []
        self.finalize_calls: list[tuple[int, _FakeInferenceCache]] = []
        self.released_caches: list[_FakeInferenceCache | None] = []
        self.reject_before_callback = False
        self.fail_decode_after_callback = False
        self.close_count = 0

    def prepare_scene(self, scene: Any) -> None:
        self.prepared_scenes.append(scene)

    def initialize_cache(
        self,
        *,
        transformer_context: dict[str, Any],
        decoder_context: dict[str, Any],
    ) -> _FakeInferenceCache:
        assert transformer_context["scene"] in self.prepared_scenes
        assert decoder_context == transformer_context
        cache = _FakeInferenceCache()
        self.caches.append(cache)
        return cache

    def generate(
        self,
        autoregressive_index: int,
        cache: _FakeInferenceCache,
        *,
        input: torch.Tensor,
        on_diffusion_complete,
    ) -> torch.Tensor:
        self.generate_calls.append((autoregressive_index, cache))
        cache.actions.append(input.clone())
        if self.reject_before_callback:
            raise RuntimeError("Cosmos rejected action")
        on_diffusion_complete()
        if self.fail_decode_after_callback:
            raise RuntimeError("decoder failed")
        return torch.zeros((1, 3, 4, 6), dtype=torch.uint8)

    def finalize(
        self,
        autoregressive_index: int,
        cache: _FakeInferenceCache,
    ) -> dict[str, float]:
        self.finalize_calls.append((autoregressive_index, cache))
        return {"fake_s": 0.25}

    def release_session(self, cache: _FakeInferenceCache | None = None) -> None:
        self.released_caches.append(cache)

    def close(self) -> None:
        self.close_count += 1


def _scene(name: str | None = "scene-a") -> Any:
    identity = np.eye(3, dtype=np.float32)
    poses = {side: np.eye(4, dtype=np.float32) for side in ("left", "right")}
    poses["left"][:3, 3] = [-0.24, 0.0, 1.0]
    poses["right"][:3, 3] = [0.24, 0.0, 1.0]
    return SimpleNamespace(
        name=name,
        fps=36.0,
        wrist_rotations={"left": identity, "right": identity},
        wrist_poses=poses,
        gripper_open={"left": 1.0, "right": 1.0},
        init_rgb=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def _events_from_raw_frame(
    frame: dict[str, Any],
    *,
    frame_id: int,
) -> UserInputEvents:
    controllers = {}
    for side in ("left", "right"):
        payload = frame[side]
        controllers[side] = ControllerSnapshot(
            trigger=float(payload["trigger"]),
            grip=float(payload["grip"]),
            position=tuple(payload["position"]),
            orientation=tuple(payload["orientation"]),
            tracked=bool(payload["tracked"]),
        )
    return UserInputEvents(
        [
            ControllerFrameEvent(
                timestamp=np.uint64(frame_id),
                action="state",
                frame_id=frame_id,
                client_capture_ns=frame_id + 1,
                pose_convention=OPENXR_LOCAL_GRIP_POSE_CONVENTION,
                left=controllers["left"],
                right=controllers["right"],
            )
        ]
    )


class _FakeRolloutBuilder:
    """Model-free rollout double; codecs are exercised in integration tests."""

    def __init__(self, *, pipeline, scene):
        self.pipeline = pipeline
        self.scene = scene

    @property
    def actions_per_step(self):
        return self.pipeline.actions_per_step

    def prepare_scene(self):
        self.pipeline.prepare_scene(self.scene)

    def build_cache(self):
        context = {"scene": self.scene}
        return self.pipeline.initialize_cache(
            transformer_context=context, decoder_context=context
        )

    def encode_actions(self, frames):
        return torch.zeros((len(frames) - 1, 64), dtype=torch.float32)

    def generate(self, step_index, cache, actions, on_accepted):
        return self.pipeline.generate(
            step_index, cache, input=actions, on_diffusion_complete=on_accepted
        )

    def finalize(self, step_index, cache):
        return self.pipeline.finalize(step_index, cache)

    def release_cache(self, cache):
        self.pipeline.release_session(cache)


def _pipeline_config(**options):
    return SimpleNamespace(**options)


def _application(
    *,
    inference_pipeline_factory=None,
    pipeline_factory=None,
    scene_factory=None,
    scene_catalog=("scene-a", "scene-b"),
):
    from handdreams.app import HandDreamsV2Application
    from handdreams.config import HandDreamsApplicationDefaults

    def add_arguments(parser):
        parser.add_argument("--device", default="cpu")

    defaults = HandDreamsApplicationDefaults(
        video_width=640,
        video_height=360,
        add_model_arguments=add_arguments,
        resolve_pipeline_config=lambda args: _pipeline_config(device=args["device"]),
        inference_pipeline_factory=inference_pipeline_factory
        or (lambda _: _FakeInferencePipeline()),
        discover_scene_catalog=lambda _: scene_catalog,
        scene_previews=lambda _scenario, _catalog: (),
        scene_factory=scene_factory or (lambda scenario, _: _scene(scenario.sample)),
        rollout_builder_factory=_FakeRolloutBuilder,
    )
    return HandDreamsV2Application(
        defaults=defaults,
        pipeline_factory=pipeline_factory or (lambda **_: _FakePipeline()),
    )
