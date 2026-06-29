# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from omnidreams.interactive_drive._pipeline_fakes import (
    FakeVideoModelBackend,
    make_trajectory,
    minimal_scene,
)
from omnidreams.interactive_drive.runtime.timing import (
    ChunkTimes,
    TraceComponentValue,
    TraceContext,
)
from omnidreams.interactive_drive.types import FrameChunk, PresentedFrame, SceneBundle
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    ChunkRequest,
)


@dataclass(frozen=True)
class _TraceEvent:
    name: str
    depends_on: list[int]
    components: dict[str, TraceComponentValue]


class _RecordingTraceSink:
    def __init__(self) -> None:
        self.threads: list[str] = []
        self.events: list[_TraceEvent] = []

    def add_thread(self, name: str) -> int:
        self.threads.append(name)
        return len(self.threads) - 1

    def add_instant(
        self,
        name: str,
        *,
        thread: int,
        time_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int:
        del thread, time_ns
        return self._append_event(name, depends_on, components)

    def add_range(
        self,
        name: str,
        *,
        thread: int,
        begin_ns: int,
        end_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int:
        del thread
        event_components = dict(components)
        event_components["begin_ns"] = begin_ns
        event_components["end_ns"] = end_ns
        return self._append_event(name, depends_on, event_components)

    def _append_event(
        self,
        name: str,
        depends_on: list[int] | None,
        components: dict[str, TraceComponentValue],
    ) -> int:
        self.events.append(
            _TraceEvent(
                name=name,
                depends_on=[] if depends_on is None else depends_on,
                components=components,
            )
        )
        return len(self.events) - 1


class _GatedBackend:
    """Backend whose render blocks on a gate so a reset can race it."""

    def __init__(self) -> None:
        self.warmup_model_calls = 0
        self.reset_calls = 0
        self.render_started = threading.Event()
        self.release = threading.Event()

    def warmup_model(self) -> None:
        self.warmup_model_calls += 1

    def load_scene(self, scene: SceneBundle) -> None:
        del scene

    def reset(self) -> None:
        self.reset_calls += 1

    def render_chunk(self, trajectory: object) -> FrameChunk:
        self.render_started.set()
        self.release.wait(timeout=5.0)
        frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_host_f32=None,
        )
        return FrameChunk(
            frames=(frame,),
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="gated",
        )


class _GatedWarmupBackend:
    """Backend whose warmup blocks so scene loads can queue behind it."""

    def __init__(self) -> None:
        self.warmup_started = threading.Event()
        self.release_warmup = threading.Event()
        self.scene_loaded = threading.Event()
        self.loaded_prompts: list[str] = []
        self.render_calls = 0

    def warmup_model(self) -> None:
        self.warmup_started.set()
        self.release_warmup.wait(timeout=5.0)

    def load_scene(self, scene: SceneBundle) -> None:
        self.loaded_prompts.append(scene.prompt)
        self.scene_loaded.set()

    def reset(self) -> None:
        return

    def render_chunk(self, trajectory: object) -> FrameChunk:
        self.render_calls += 1
        frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_host_f32=None,
        )
        return FrameChunk(
            frames=(frame,),
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="gated-warmup",
        )


def _wait_for_queue_size(pipeline: ChunkPipeline, size: int) -> None:
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        if pipeline.frame_queue.qsize() >= size:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"timed out waiting for frame queue size {size}; "
        f"actual={pipeline.frame_queue.qsize()}"
    )


def _chunk_times(chunk_size: int) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=0,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        intended_present_times=[
            now + 0.1 + idx * (1.0 / 30.0) for idx in range(chunk_size)
        ],
    )


def test_chunk_pipeline_stamps_timing_and_orders_frames() -> None:
    backend = FakeVideoModelBackend(frames_per_render=3)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    chunk_times = _chunk_times(chunk_size=3)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(3), chunk_times=chunk_times)
    )

    first = pipeline.frame_queue.get(timeout=1.0)
    second = pipeline.frame_queue.get(timeout=1.0)
    third = pipeline.frame_queue.get(timeout=1.0)
    pipeline.shutdown()

    assert [first.frame_index, second.frame_index, third.frame_index] == [0, 1, 2]
    assert first.chunk_times is chunk_times
    assert chunk_times.chunk_render_start_time is not None
    assert chunk_times.chunk_ready_time is not None
    assert chunk_times.frames[0].image_ready_time is not None
    assert backend.warmup_model_calls == 1
    assert backend.load_scene_calls == 1


def test_chunk_pipeline_reset_invokes_backend_reset() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    chunk_times = _chunk_times(chunk_size=1)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=chunk_times)
    )
    pipeline.frame_queue.get(timeout=1.0)
    pipeline.reset()
    pipeline.shutdown()

    assert backend.warmup_model_calls == 1
    assert backend.reset_calls == 1


def test_chunk_pipeline_reuses_model_across_scene_changes() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend)
    assert pipeline.model_ready.wait(timeout=1.0)
    for _ in range(3):
        pipeline.request_scene(minimal_scene())
        chunk_times = _chunk_times(chunk_size=1)
        pipeline.request_pose_chunk(
            ChunkRequest(trajectory=make_trajectory(1), chunk_times=chunk_times)
        )
        pipeline.frame_queue.get(timeout=1.0)
    pipeline.shutdown()

    # The model is warmed exactly once even though the scene changed twice.
    assert backend.warmup_model_calls == 1
    assert backend.load_scene_calls == 3


def test_chunk_pipeline_drops_superseded_render_after_reset() -> None:
    backend = _GatedBackend()
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=_chunk_times(1))
    )
    # Wait until the (gated) render is in flight, then reset to supersede it.
    assert backend.render_started.wait(timeout=1.0)
    pipeline.reset()
    backend.release.set()  # let the superseded render finish
    pipeline.shutdown()

    assert backend.reset_calls == 1
    # The in-flight render belonged to the pre-reset generation, so its
    # frame is dropped rather than queued.
    assert pipeline.frame_queue.qsize() == 0


def test_chunk_pipeline_reset_clears_already_queued_frames() -> None:
    backend = FakeVideoModelBackend(frames_per_render=2)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(2), chunk_times=_chunk_times(2))
    )
    _wait_for_queue_size(pipeline, 2)

    pipeline.reset()
    pipeline.shutdown()

    assert backend.reset_calls == 1
    assert pipeline.frame_queue.qsize() == 0


def test_chunk_pipeline_scene_change_clears_already_queued_frames() -> None:
    backend = FakeVideoModelBackend(frames_per_render=2)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(2), chunk_times=_chunk_times(2))
    )
    _wait_for_queue_size(pipeline, 2)

    pipeline.request_scene(replace(minimal_scene(), prompt="new scene"))
    pipeline.shutdown()

    assert backend.load_scene_calls == 2
    assert pipeline.frame_queue.qsize() == 0


def test_chunk_pipeline_sets_first_chunk_produced_after_first_chunk() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend)
    assert pipeline.model_ready.wait(timeout=1.0)
    # Warmup done, but no generated chunk yet -> still in the optimize phase.
    assert not pipeline.first_chunk_produced.is_set()

    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=_chunk_times(1))
    )
    pipeline.frame_queue.get(timeout=1.0)
    assert pipeline.first_chunk_produced.wait(timeout=1.0)
    pipeline.shutdown()


def test_chunk_pipeline_first_chunk_produced_unset_for_superseded_render() -> None:
    backend = _GatedBackend()
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=_chunk_times(1))
    )
    # Reset supersedes the in-flight render, so its frames are dropped and the
    # optimize-phase latch must not flip on a chunk the user never saw.
    assert backend.render_started.wait(timeout=1.0)
    pipeline.reset()
    backend.release.set()
    pipeline.shutdown()

    assert pipeline.frame_queue.qsize() == 0
    assert not pipeline.first_chunk_produced.is_set()


def test_chunk_pipeline_skips_stale_scene_load_queued_behind_warmup() -> None:
    backend = _GatedWarmupBackend()
    pipeline = ChunkPipeline(backend)
    assert backend.warmup_started.wait(timeout=1.0)

    default_scene = replace(
        minimal_scene(), scene_path=Path("clipgt-scene.usdz"), prompt="clear"
    )
    snow_scene = replace(
        minimal_scene(), scene_path=Path("clipgt-scene-snow.usdz"), prompt="snow"
    )

    pipeline.request_scene(default_scene)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=_chunk_times(1))
    )
    pipeline.request_scene(snow_scene)
    backend.release_warmup.set()
    assert backend.scene_loaded.wait(timeout=1.0)
    pipeline.shutdown()

    assert backend.loaded_prompts == ["snow"]
    assert backend.render_calls == 0


def test_chunk_pipeline_emits_worker_trace_ranges() -> None:
    sink = _RecordingTraceSink()
    trace_context = TraceContext.create(sink)
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend, trace_context=trace_context)
    pipeline.request_scene(minimal_scene())
    chunk_times = _chunk_times(chunk_size=1)
    chunk_times.chunk_index = 1
    pipeline.request_pose_chunk(
        ChunkRequest(
            trajectory=make_trajectory(1),
            chunk_times=chunk_times,
            trace_dependency_event=123,
        )
    )

    queued = pipeline.frame_queue.get(timeout=1.0)
    pipeline.shutdown()

    assert queued.worker_ready_event_id is not None
    names = [event.name for event in sink.events]
    assert "worker_warmup" in names
    assert "queue_wait" in names
    assert "chunk_render" in names
    queue_wait = sink.events[names.index("queue_wait")]
    chunk_render = sink.events[names.index("chunk_render")]
    assert queue_wait.depends_on == [123]
    assert chunk_render.depends_on == [names.index("queue_wait")]
    assert chunk_render.components["chunk_index"] == 1
    assert chunk_render.components["source"] == "fake"
