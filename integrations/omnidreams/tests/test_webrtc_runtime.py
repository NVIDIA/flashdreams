# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omnidreams import scenes
from omnidreams.webrtc import server as webrtc_server
from omnidreams.webrtc import session
from omnidreams.webrtc.session import (
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
    OmnidreamsStepResult,
    OmnidreamsWebRTCSessionManager,
)

from flashdreams.serving.webrtc.controls import CameraPoseIntegrator

pytestmark = pytest.mark.ci_cpu


class _FakeCloseable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _fake_runtime_factory(config: OmnidreamsRuntimeConfig) -> object:
    del config
    return object()


@dataclass
class _FakeOutput:
    state: Any
    condition_frames: torch.Tensor
    rgb_frames: torch.Tensor | None
    finalization_state: dict[str, int]


class _FakeWrapper:
    initial_frame_chunk_size = 2
    frame_chunk_size = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], list[int]]] = []
        self.finalized: list[dict[str, int]] = []
        self.skip_video_generation_flags: list[bool] = []

    def start_generation(self, **kwargs: Any) -> _FakeOutput:
        poses = kwargs["camera_poses_per_view"]["camera_front_wide_120fov"]
        timestamps = kwargs["frame_timestamps_us"]
        self.calls.append(("start", tuple(poses.shape), timestamps))
        skip_video_generation = bool(kwargs.get("skip_video_generation", False))
        self.skip_video_generation_flags.append(skip_video_generation)
        return _FakeOutput(
            state=SimpleNamespace(
                pipeline_cache=None if skip_video_generation else object()
            ),
            condition_frames=torch.full((1, 1, 2, 3, 4, 5), 31, dtype=torch.uint8),
            rgb_frames=(
                None
                if skip_video_generation
                else torch.zeros((1, 1, 2, 3, 4, 5), dtype=torch.uint8)
            ),
            finalization_state={"autoregressive_index": 0},
        )

    def continue_generation(self, **kwargs: Any) -> _FakeOutput:
        poses = kwargs["camera_poses_per_view"]["camera_front_wide_120fov"]
        timestamps = kwargs["frame_timestamps_us"]
        self.calls.append(("continue", tuple(poses.shape), timestamps))
        skip_video_generation = bool(kwargs.get("skip_video_generation", False))
        self.skip_video_generation_flags.append(skip_video_generation)
        return _FakeOutput(
            state=kwargs["state"],
            condition_frames=torch.full((1, 1, 3, 3, 4, 5), 47, dtype=torch.uint8),
            rgb_frames=(
                None
                if skip_video_generation
                else torch.zeros((1, 1, 3, 3, 4, 5), dtype=torch.uint8)
            ),
            finalization_state={"autoregressive_index": 1},
        )

    def finalize_block_generation(
        self, pipeline_cache: object, finalization_state: dict[str, int]
    ) -> None:
        del pipeline_cache
        self.finalized.append(finalization_state)


def _build_fake_runtime() -> tuple[OmnidreamsInferenceRuntime, _FakeWrapper]:
    runtime = OmnidreamsInferenceRuntime(
        config=OmnidreamsRuntimeConfig(device="cpu", fps=30)
    )
    wrapper = _FakeWrapper()
    runtime._wrapper = wrapper  # ty:ignore[invalid-assignment]
    runtime._renderer = object()
    runtime._initial_rgb_frames = torch.zeros((1, 1, 3, 4, 5), dtype=torch.uint8)
    runtime._text_prompts = []
    runtime._camera_to_rig = torch.eye(4)
    runtime._device = torch.device("cpu")
    runtime._next_timestamp_us = 1000
    runtime.pose_integrator = CameraPoseIntegrator()
    runtime.pose_integrator.reset()
    return runtime, wrapper


def test_generate_chunk_dispatches_start_then_continue() -> None:
    runtime, wrapper = _build_fake_runtime()

    result0 = runtime._generate_one_chunk_sync(
        segments=[(0.0, 2 / 30, frozenset({"w"}))],
        frame_times=[1 / 30, 2 / 30],
    )
    result1 = runtime._generate_one_chunk_sync(
        segments=[(2 / 30, 5 / 30, frozenset())],
        frame_times=[3 / 30, 4 / 30, 5 / 30],
    )

    assert result0.chunk_index == 0
    assert result0.num_frames == 2
    assert result1.chunk_index == 1
    assert result1.num_frames == 3
    assert wrapper.calls[0][0] == "start"
    assert wrapper.calls[0][1] == (2, 4, 4)
    assert wrapper.calls[0][2] == [1000, 34333]
    assert wrapper.calls[1][0] == "continue"
    assert wrapper.calls[1][1] == (3, 4, 4)
    assert len(wrapper.finalized) == 2
    assert wrapper.skip_video_generation_flags == [False, False]


def test_generate_chunk_can_stream_debug_hdmaps_without_rgb_frames() -> None:
    runtime, wrapper = _build_fake_runtime()
    runtime.config.debug_serve_hdmaps = True

    result0 = runtime._generate_one_chunk_sync(
        segments=[(0.0, 2 / 30, frozenset({"w"}))],
        frame_times=[1 / 30, 2 / 30],
    )
    result1 = runtime._generate_one_chunk_sync(
        segments=[(2 / 30, 5 / 30, frozenset({"d"}))],
        frame_times=[3 / 30, 4 / 30, 5 / 30],
    )

    assert result0.chunk_index == 0
    assert result0.num_frames == 2
    assert result0.video_chunk.shape == (1, 1, 2, 3, 4, 5)
    assert result0.video_chunk.unique().tolist() == [31]
    assert result1.chunk_index == 1
    assert result1.num_frames == 3
    assert result1.video_chunk.shape == (1, 1, 3, 3, 4, 5)
    assert result1.video_chunk.unique().tolist() == [47]
    assert wrapper.skip_video_generation_flags == [True, True]
    assert wrapper.finalized == []


def test_prepare_clipgt_dir_stages_unprefixed_parquets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clipgt = tmp_path / "clipgt"
    clipgt.mkdir()
    (clipgt / "calibration_estimate.parquet").touch()
    (clipgt / "egomotion_estimate.parquet").touch()
    (clipgt / "lane.parquet").touch()
    runtime = OmnidreamsInferenceRuntime(
        config=OmnidreamsRuntimeConfig(device="cpu", fps=30)
    )

    staged = runtime._prepare_clipgt_dir(clipgt)

    assert staged != clipgt
    assert (staged / "clip.calibration_estimate.parquet").exists()
    assert (staged / "clip.egomotion_estimate.parquet").exists()
    assert (staged / "clip.lane.parquet").exists()

    monkeypatch.chdir(tmp_path)
    staged_from_relative = runtime._prepare_clipgt_dir(Path("clipgt"))
    assert (staged_from_relative / "clip.calibration_estimate.parquet").exists()


def test_prepare_clipgt_dir_stages_nested_unprefixed_parquets(tmp_path: Path) -> None:
    clipgt = tmp_path / "clipgt"
    clipgt.mkdir()
    nested = clipgt / "clipgt"
    nested.mkdir()
    (clipgt / "first_image.png").touch()
    (clipgt / "prompt.txt").touch()
    (nested / "calibration_estimate.parquet").touch()
    (nested / "egomotion_estimate.parquet").touch()
    (nested / "lane.parquet").touch()
    runtime = OmnidreamsInferenceRuntime(
        config=OmnidreamsRuntimeConfig(device="cpu", fps=30)
    )

    staged = runtime._prepare_clipgt_dir(clipgt)

    assert staged != clipgt
    assert (staged / "clip.calibration_estimate.parquet").exists()
    assert (staged / "clip.egomotion_estimate.parquet").exists()
    assert (staged / "clip.lane.parquet").exists()


def test_hf_webrtc_scene_sync_requires_usdz_first_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene_uuid = "065dcac9-ee67-4434-a835-c6b816c88e48"
    archive_repo_path = f"scenes/clipgt-{scene_uuid}.usdz"
    archive_path = tmp_path / "clipgt.usdz"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("calibration_estimate.parquet", "calibration")
        zf.writestr("egomotion_estimate.parquet", "egomotion")
        zf.writestr("prompt.txt", "archive prompt")

    def _fake_hf_hub_download(repo_id: str, repo_type: str, filename: str) -> str:
        assert repo_id == session.hf_scenes_repo_id()
        assert repo_type == "dataset"
        assert filename == archive_repo_path
        return str(archive_path)

    cache_dir = tmp_path / "flashdreams-cache"
    stale_scene_dir = cache_dir / "omnidreams-scenes" / scene_uuid
    stale_scene_dir.mkdir(parents=True)
    (stale_scene_dir / "first_frame.jpeg").write_text(
        "stale first frame", encoding="utf-8"
    )
    (stale_scene_dir / "prompt.txt").write_text("stale prompt", encoding="utf-8")

    monkeypatch.setattr(scenes, "FLASHDREAMS_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download,
    )

    scene_dir = session._ensure_hf_webrtc_scene_synced(scene_uuid)

    with pytest.raises(FileNotFoundError, match="first_image"):
        session._resolve_webrtc_scene_assets(
            scene_dir,
            prompt_filename="prompt.txt",
            clipgt_dirname="clipgt",
        )


def test_hf_webrtc_scene_sync_uses_extracted_first_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene_uuid = "065dcac9-ee67-4434-a835-c6b816c88e48"
    archive_repo_path = f"scenes/clipgt-{scene_uuid}.usdz"
    archive_path = tmp_path / "clipgt.usdz"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("calibration_estimate.parquet", "calibration")
        zf.writestr("egomotion_estimate.parquet", "egomotion")
        zf.writestr("first_image.png", "first image")
        zf.writestr("prompt.txt", "archive prompt")

    def _fake_hf_hub_download(repo_id: str, repo_type: str, filename: str) -> str:
        assert repo_id == session.hf_scenes_repo_id()
        assert repo_type == "dataset"
        assert filename == archive_repo_path
        return str(archive_path)

    cache_dir = tmp_path / "flashdreams-cache"
    stale_scene_dir = cache_dir / "omnidreams-scenes" / scene_uuid
    stale_scene_dir.mkdir(parents=True)
    (stale_scene_dir / "first_frame.jpeg").write_text(
        "stale first frame", encoding="utf-8"
    )
    (stale_scene_dir / "prompt.txt").write_text("stale prompt", encoding="utf-8")

    monkeypatch.setattr(scenes, "FLASHDREAMS_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download,
    )

    scene_dir = session._ensure_hf_webrtc_scene_synced(scene_uuid)

    assert (scene_dir / "clipgt" / "first_image.png").read_text(
        encoding="utf-8"
    ) == "first image"
    assert (scene_dir / "clipgt" / "prompt.txt").read_text(
        encoding="utf-8"
    ) == "archive prompt"

    clipgt_dir, first_frame_path, prompt_path = session._resolve_webrtc_scene_assets(
        scene_dir,
        prompt_filename="prompt.txt",
        clipgt_dirname="clipgt",
    )
    assert clipgt_dir == scene_dir / "clipgt"
    assert first_frame_path == scene_dir / "clipgt" / "first_image.png"
    assert prompt_path == scene_dir / "clipgt" / "prompt.txt"


def test_hf_webrtc_scene_sync_requires_usdz_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene_uuid = "065dcac9-ee67-4434-a835-c6b816c88e48"
    archive_repo_path = f"scenes/clipgt-{scene_uuid}.usdz"
    archive_path = tmp_path / "clipgt.usdz"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("calibration_estimate.parquet", "calibration")
        zf.writestr("egomotion_estimate.parquet", "egomotion")
        zf.writestr("first_image.png", "first image")

    def _fake_hf_hub_download(repo_id: str, repo_type: str, filename: str) -> str:
        assert repo_id == session.hf_scenes_repo_id()
        assert repo_type == "dataset"
        assert filename == archive_repo_path
        return str(archive_path)

    monkeypatch.setattr(scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "flashdreams-cache")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download,
    )

    scene_dir = session._ensure_hf_webrtc_scene_synced(scene_uuid)

    with pytest.raises(FileNotFoundError, match="prompt.txt"):
        session._resolve_webrtc_scene_assets(
            scene_dir,
            prompt_filename="prompt.txt",
            clipgt_dirname="clipgt",
        )


def test_resolved_empty_prompt_keeps_runtime_default_behavior(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    clipgt_dir = scene_dir / "clipgt"
    clipgt_dir.mkdir(parents=True)
    (clipgt_dir / "first_image.png").write_text("first image", encoding="utf-8")
    (clipgt_dir / "prompt.txt").write_text("", encoding="utf-8")

    _, _, prompt_path = session._resolve_webrtc_scene_assets(
        scene_dir,
        prompt_filename="prompt.txt",
        clipgt_dirname="clipgt",
    )

    assert prompt_path == clipgt_dir / "prompt.txt"
    assert (
        prompt_path.read_text(encoding="utf-8").strip() or session.AV_POSITIVE_PROMPT
    ) == session.AV_POSITIVE_PROMPT


def test_build_runtime_config_threads_hf_scene_args(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pipeline_config_name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        scene_dir=tmp_path / "local-scene",
        scene_uuid="scene-123",
        scene_variant="rain",
        seed=123,
        device="cuda:0",
        video_height=360,
        video_width=640,
        fps=24,
        camera_name="camera_front_wide_120fov",
        warmup_chunks=0,
        warmup_timeout_s=30.0,
        debug_serve_hdmaps=True,
    )

    cfg = webrtc_server.build_runtime_config(args, device_override="cuda:7")

    assert cfg.scene_dir == tmp_path / "local-scene"
    assert cfg.scene_uuid == "scene-123"
    assert cfg.scene_variant == "rain"
    assert cfg.device == "cuda:7"
    assert cfg.video_height == 360
    assert cfg.video_width == 640
    assert cfg.debug_serve_hdmaps is True


def test_parse_args_omits_scene_dir_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omnidreams.webrtc.server",
            "--debug_serve_hdmaps",
        ],
    )

    args = webrtc_server.parse_args()

    assert args.scene_dir is None
    assert args.scene_uuid is None
    assert args.debug_serve_hdmaps is True


def test_runtime_uses_default_scene_uuid_when_scene_is_unspecified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staged_scene_dir = tmp_path / "staged-scene"
    calls: list[str] = []

    def _fake_ensure_hf_webrtc_scene_synced(
        scene_uuid: str,
        *,
        variant: str = "default",
        prompt_filename: str,
        clipgt_dirname: str,
    ) -> Path:
        del prompt_filename, clipgt_dirname, variant
        calls.append(scene_uuid)
        return staged_scene_dir

    def _fake_resolve_webrtc_scene_assets(
        scene_dir: Path,
        *,
        prompt_filename: str,
        clipgt_dirname: str,
        camera_name: str = "camera_front_wide_120fov",
        variant: str = "default",
    ) -> tuple[Path, Path, Path]:
        del prompt_filename, clipgt_dirname, camera_name, variant
        clipgt_dir = scene_dir / "clipgt"
        return clipgt_dir, clipgt_dir / "first_image.png", clipgt_dir / "prompt.txt"

    monkeypatch.setattr(
        session,
        "_ensure_hf_webrtc_scene_synced",
        _fake_ensure_hf_webrtc_scene_synced,
    )
    monkeypatch.setattr(
        session,
        "_resolve_webrtc_scene_assets",
        _fake_resolve_webrtc_scene_assets,
    )
    monkeypatch.setattr(session, "load_scene", lambda *args, **kwargs: None)
    runtime = OmnidreamsInferenceRuntime(
        config=OmnidreamsRuntimeConfig(
            pipeline_config_name="missing-config",
            device="cpu",
            scene_dir=None,
            scene_uuid=None,
        )
    )

    with pytest.raises(ValueError, match="Unknown pipeline_config_name"):
        runtime._initialize_sync()

    assert calls == [session.DEFAULT_WEBRTC_SCENE_UUID]


def test_build_runtime_config_clears_scene_uuid_for_local_scene(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pipeline_config_name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        scene_dir=tmp_path / "local-scene",
        scene_uuid=None,
        scene_variant="default",
        seed=123,
        device="cuda:0",
        video_height=360,
        video_width=640,
        fps=24,
        camera_name="camera_front_wide_120fov",
        warmup_chunks=0,
        warmup_timeout_s=30.0,
        debug_serve_hdmaps=True,
    )

    cfg = webrtc_server.build_runtime_config(args)

    assert cfg.scene_dir == tmp_path / "local-scene"
    assert cfg.scene_uuid is None
    assert cfg.scene_variant == "default"


@pytest.mark.asyncio
async def test_session_manager_preload_runs_loopback_warmup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: OmnidreamsRuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.close_calls = 0

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None
    warmup_calls: list[int] = []

    def _fake_runtime_factory(config: OmnidreamsRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    async def _fake_loopback_warmup(
        self: OmnidreamsWebRTCSessionManager, *, num_chunks: int
    ) -> None:
        del self
        warmup_calls.append(num_chunks)

    monkeypatch.setattr(session, "OmnidreamsInferenceRuntime", _fake_runtime_factory)
    monkeypatch.setattr(
        OmnidreamsWebRTCSessionManager,
        "_run_loopback_warmup_session",
        _fake_loopback_warmup,
    )
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(device="cpu", warmup_chunks=2)
    )

    await manager.preload_runtime()
    await manager.preload_runtime()

    assert fake_runtime is not None
    assert fake_runtime.initialize_calls == 1
    assert warmup_calls == [2]
    assert manager.is_runtime_ready()


@pytest.mark.asyncio
async def test_loopback_warmup_drives_session_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: OmnidreamsRuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.reset_calls = 0
            self.close_calls = 0
            self.generated_segments: list[
                list[tuple[float, float, frozenset[str]]]
            ] = []

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def reset_for_new_session(self) -> None:
            self.reset_calls += 1

        def peek_steady_chunk_num_frames(self) -> int:
            return 1

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self,
            *,
            segments: list[tuple[float, float, frozenset[str]]],
            frame_times: list[float],
        ) -> OmnidreamsStepResult:
            del frame_times
            chunk_index = len(self.generated_segments)
            self.generated_segments.append(segments)
            return OmnidreamsStepResult(
                chunk_index=chunk_index,
                num_frames=1,
                video_chunk=torch.zeros((1, 1, 1, 3, 2, 2), dtype=torch.uint8),
                stats=None,
            )

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None

    def _fake_runtime_factory(config: OmnidreamsRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "OmnidreamsInferenceRuntime", _fake_runtime_factory)
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(
            device="cpu",
            fps=30,
            warmup_chunks=2,
        )
    )

    await asyncio.wait_for(manager.preload_runtime(), timeout=10.0)

    assert fake_runtime is not None
    assert fake_runtime.initialize_calls == 1
    assert fake_runtime.reset_calls == 1
    assert len(fake_runtime.generated_segments) == 2
    assert not manager.has_active_session()


@pytest.mark.asyncio
async def test_heartbeat_message_refreshes_client_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "OmnidreamsInferenceRuntime", _fake_runtime_factory)
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    managed_session = session._ManagedOmnidreamsSession(
        runtime=object(),  # ty:ignore[invalid-argument-type]
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=object(),
        last_client_message_at=0.0,
    )
    manager._active_session = managed_session

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"heartbeat"}',
    )

    assert managed_session.last_client_message_at > 0.0
    assert manager.has_active_session()


@pytest.mark.asyncio
async def test_client_liveness_timeout_closes_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "OmnidreamsInferenceRuntime", _fake_runtime_factory)
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(device="cpu", warmup_chunks=0),
        client_liveness_timeout_s=0.01,
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedOmnidreamsSession(
        runtime=object(),  # ty:ignore[invalid-argument-type]
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer_connection,
        resampler=object(),  # ty:ignore[invalid-argument-type]
        last_client_message_at=asyncio.get_running_loop().time() - 1.0,
    )
    manager._active_session = managed_session
    liveness_task = asyncio.create_task(
        manager._client_liveness_watchdog(managed_session=managed_session)
    )
    managed_session.liveness_task = liveness_task

    await asyncio.wait_for(liveness_task, timeout=1.0)

    assert not manager.has_active_session()
    assert managed_session.closed
    assert video_track.closed
    assert peer_connection.closed


@pytest.mark.asyncio
async def test_disconnect_message_closes_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "OmnidreamsInferenceRuntime", _fake_runtime_factory)
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedOmnidreamsSession(
        runtime=object(),  # ty:ignore[invalid-argument-type]
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer_connection,
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=object(),
    )
    manager._active_session = managed_session

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"disconnect"}',
    )

    assert not manager.has_active_session()
    assert managed_session.closed
    assert video_track.closed
    assert peer_connection.closed


@pytest.mark.asyncio
async def test_generation_worker_closes_session_after_generation_failure() -> None:
    class _FailingRuntime:
        def __init__(self) -> None:
            self.generate_calls = 0

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self,
            *,
            segments: list[tuple[float, float, frozenset[str]]],
            frame_times: list[float],
        ) -> OmnidreamsStepResult:
            del segments, frame_times
            self.generate_calls += 1
            raise RuntimeError("boom")

    class _FakeResampler:
        dt = 0.0
        next_chunk_start_v = 0.0

        def sample_chunk(
            self, num_frames: int
        ) -> tuple[list[tuple[float, float, frozenset[str]]], list[float]]:
            assert num_frames == 1
            return [(0.0, 0.0, frozenset({"w"}))], [0.0]

    class _FakeVideoTrack:
        fps = 30

        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

        def qsize(self) -> int:
            return 0

    class _FakePeerConnection:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class _FakeChannel:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def send(self, message: str) -> None:
            self.messages.append(message)

    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime = _FailingRuntime()
    video_track = _FakeVideoTrack()
    peer_connection = _FakePeerConnection()
    control_channel = _FakeChannel()
    first_action_received = asyncio.Event()
    first_action_received.set()
    managed_session = session._ManagedOmnidreamsSession(
        runtime=runtime,  # ty:ignore[invalid-argument-type]
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer_connection,
        resampler=_FakeResampler(),  # ty:ignore[invalid-argument-type]
        control_channel=control_channel,
        first_action_received=first_action_received,
    )
    manager._active_session = managed_session

    task = asyncio.create_task(
        manager._generation_worker(managed_session=managed_session)
    )
    managed_session.generation_task = task

    await task

    assert runtime.generate_calls == 1
    assert not manager.has_active_session()
    assert managed_session.closed
    assert video_track.closed
    assert peer_connection.closed
    assert len(control_channel.messages) == 1
