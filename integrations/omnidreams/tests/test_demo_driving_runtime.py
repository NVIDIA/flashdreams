# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    StepRequest,
)
from omnidreams.demo.driving import (
    DRIVING_INPUT_SCHEMA,
    OmnidreamsDriverCommandMapping,
    OmnidreamsDrivingRuntime,
    OmnidreamsDrivingScenario,
)
from omnidreams.interactive_drive.config import AppConfig, ChunkConfig
from omnidreams.interactive_drive.types import FrameChunk, PresentedFrame, SceneBundle

pytestmark = pytest.mark.ci_cpu


class _Backend:
    def __init__(self) -> None:
        self.warmups = 0
        self.loaded: list[object] = []
        self.closed = False

    @property
    def can_prewarm(self) -> bool:
        return True

    def warmup_model(self) -> None:
        self.warmups += 1

    def load_scene(self, scene: object) -> None:
        self.loaded.append(scene)

    def reset_scene_conditioning(self) -> None:
        return

    def reset(self) -> None:
        return

    def set_postprocess_enabled(self, enabled: bool) -> None:
        del enabled

    def render_first_chunk(self, trajectory: Any) -> FrameChunk:
        return _frame_chunk(trajectory)

    def render_next_chunk(self, trajectory: Any) -> FrameChunk:
        return _frame_chunk(trajectory)

    def close(self) -> None:
        self.closed = True


def _scene() -> SceneBundle:
    return cast(
        SceneBundle,
        SimpleNamespace(
            scene_path=SimpleNamespace(stem="test-scene"),
            initial_rig_to_world=np.eye(4, dtype=np.float32),
            initial_yaw_rad=0.0,
            initial_timestamp_us=0,
        ),
    )


def _frame_chunk(trajectory: Any) -> FrameChunk:
    frames = tuple(
        PresentedFrame(
            timestamp_us=int(timestamp),
            rgb_host_uint8=np.zeros((4, 6, 3), dtype=np.uint8),
            depth_host_f32=None,
        )
        for timestamp in trajectory.timestamps_us
    )
    return FrameChunk(
        frames=frames,
        boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
        source_name="fake",
    )


def test_driver_mapping_consumes_canonical_command() -> None:
    mapping = OmnidreamsDriverCommandMapping()
    mapped = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(
            values={
                DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                    {
                        "throttle": 0.75,
                        "brake": 0.0,
                        "steer": 0.25,
                        "stop": False,
                        "reverse": False,
                        "steer_is_direct": False,
                        "manual_control": False,
                    }
                )
            }
        ),
        inference_input=InferenceInput(),
        request=StepRequest(step_index=0),
    )

    assert mapped.step["driver_command"]["throttle"] == 0.75
    assert DRIVER_COMMAND in mapping.mapping_schema.consumes


def test_runtime_warms_once_and_sessions_emit_video_results(tmp_path) -> None:
    backend = _Backend()
    app_config = AppConfig(
        scene_path=tmp_path / "scene.usdz",
        chunk=ChunkConfig(fps=10, initial_chunk_frames=2, chunk_frames=3),
    )
    scenario = OmnidreamsDrivingScenario(
        app_config=app_config,
        scene=_scene(),
        map_bounds=None,
        ground_snapper=None,
    )
    runtime = OmnidreamsDrivingRuntime(
        config=InferenceConfig(model_id="omnidreams"),
        app_config=app_config,
        backend=cast(Any, backend),
    )
    session = runtime.start_session(
        InferenceInput(global_conditioning={"driving_scenario": scenario})
    )

    request = session.next_step_request()
    assert request is not None
    result = session.step(
        InferenceInput(
            step={
                "driver_command": {
                    "throttle": 0.5,
                    "brake": 0.0,
                    "steer": 0.0,
                    "stop": False,
                    "reverse": False,
                }
            }
        )
    )

    assert backend.warmups == 1
    assert backend.loaded == [scenario.scene]
    assert result.frame_count == 2
    assert result.output.layout == "bvtchw"
    assert result.output.metadata["interactive_drive"]["throttle"] == 0.5
    DRIVING_INPUT_SCHEMA.require_step(
        InferenceInput(step={"driver_command": {"throttle": 0.0}})
    )
    session.close()

    second = runtime.start_session(
        InferenceInput(global_conditioning={"driving_scenario": scenario})
    )
    assert second is not session
    assert backend.warmups == 1
    assert backend.loaded == [scenario.scene, scenario.scene]
    second.close()
    runtime.close()
    assert backend.closed
