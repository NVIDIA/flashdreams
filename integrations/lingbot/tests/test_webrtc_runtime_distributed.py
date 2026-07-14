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

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from lingbot.webrtc import session

pytestmark = pytest.mark.ci_gpu


def _write_minimal_assets(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "image.jpg").touch()
    np.save(
        data_dir / "intrinsics.npy", np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 13, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 1.2, poses.shape[0], dtype=np.float32)
    np.save(data_dir / "poses.npy", poses)
    (data_dir / "prompt.txt").write_text("drive through a city\n", encoding="utf-8")


def _patch_cv2(monkeypatch: pytest.MonkeyPatch, *, height: int, width: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    monkeypatch.setattr(session.cv2, "imread", lambda path, flags: image.copy())
    monkeypatch.setattr(session.cv2, "cvtColor", lambda src, code: src)
    monkeypatch.setattr(
        session.cv2, "resize", lambda src, size, interpolation: image.copy()
    )


class _FakePipeline:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.text_encoder = self._encode_text

    def to(self, *, device: torch.device) -> "_FakePipeline":
        self.events.append(("to", str(device)))
        return self

    def _ensure_oneshot_encoders_loaded(self) -> None:
        self.events.append(("ensure_oneshot_encoders_loaded",))

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        self.events.append(("text_encoder", tuple(texts)))
        return torch.zeros((len(texts), 2, 3), dtype=torch.float32)

    def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> object:
        self.events.append(
            ("initialize_cache", tuple(text), tuple(image.shape), str(image.device))
        )
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        self.events.append(("get_num_output_frames", autoregressive_index))
        return 1

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
        input: Any,
    ) -> torch.Tensor:
        self.events.append(("generate", autoregressive_index, str(input.poses.device)))
        return torch.full(
            (1, 1, 1, 3, 2, 2),
            float(input.poses.device.index or 0),
            device=input.poses.device,
        )

    def finalize(self, autoregressive_index: int, cache: object) -> dict[str, float]:
        self.events.append(("finalize", autoregressive_index))
        return {"rank_local_finalize": float(autoregressive_index)}


class _FakePipelineConfig:
    """Minimal stand-in for the ``flashdreams`` pipeline-config object
    returned by :func:`derive_config`; only implements ``setup`` so the
    runtime can call ``setup().to(device=...)``.
    """

    def __init__(
        self,
        derive_kwargs: dict[str, Any],
        events: list[tuple[Any, ...]],
    ) -> None:
        self.derive_kwargs = derive_kwargs
        self.events = events

    def setup(self) -> _FakePipeline:
        self.events.append(("setup", self.derive_kwargs))
        return _FakePipeline(self.events)


def _patch_pipeline_factory(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    derive_calls: list[dict[str, Any]],
    pipeline_events: list[tuple[Any, ...]],
) -> None:
    """Register a fake pipeline config for ``config_name`` and swap
    :func:`session.derive_config` with a capturing stub that
    returns a :class:`_FakePipelineConfig`.

    The runtime path under test is::

        derive_config(base_config=pipeline_configs[name], ...)
            .setup().to(device=...)
    """
    monkeypatch.setattr(session, "_pipeline_configs", lambda: {config_name: object()})

    def _fake_derive_config(**kwargs: Any) -> _FakePipelineConfig:
        derive_calls.append(kwargs)
        return _FakePipelineConfig(kwargs, pipeline_events)

    monkeypatch.setattr(session, "derive_config", _fake_derive_config)


def test_initialize_sync_passes_rank_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_minimal_assets(tmp_path)
    _patch_cv2(monkeypatch, height=4, width=4)
    monkeypatch.setattr(session.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(session.dist, "get_rank", lambda: 2)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(monkeypatch, "TestLingbot", derive_calls, pipeline_events)

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="TestLingbot",
            compile_network=False,
            seed=10,
            context_parallel_size=4,
            device="cpu",
            video_height=4,
            video_width=4,
            example_data_dir=tmp_path,
        )
    )

    runtime._initialize_sync()

    # Under context parallelism the rollout seed must be offset by rank;
    # that lands inside the ``diffusion_model`` nested dict that the
    # runtime hands to :func:`derive_config`.
    assert len(derive_calls) == 1
    call = derive_calls[0]
    assert call["enable_sync_and_profile"] is True
    assert call["diffusion_model"]["seed"] == 12
    assert call["diffusion_model"]["transformer"]["compile_network"] is False
    assert ("to", "cpu") in pipeline_events
    assert any(event[0] == "initialize_cache" for event in pipeline_events)


def test_initialize_sync_keeps_base_seed_without_context_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_minimal_assets(tmp_path)
    _patch_cv2(monkeypatch, height=4, width=4)
    monkeypatch.setattr(session.dist, "is_initialized", lambda: False)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(monkeypatch, "TestLingbot", derive_calls, pipeline_events)

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="TestLingbot",
            seed=10,
            context_parallel_size=1,
            device="cpu",
            video_height=4,
            video_width=4,
            example_data_dir=tmp_path,
        )
    )

    runtime._initialize_sync()

    # Without context parallelism the base seed is used verbatim.
    assert derive_calls[0]["diffusion_model"]["seed"] == 10


def test_initialize_sync_uses_demo_camera_defaults_from_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "image.jpg").touch()
    demo_intrinsics = np.array(
        [502.9115905761719, 503.1081237792969, 415.7778625488281, 239.7777862548828],
        dtype=np.float32,
    )
    np.save(tmp_path / "intrinsics.npy", demo_intrinsics)
    demo_poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 13, axis=0)
    demo_poses[:, 2, 3] = np.linspace(0.0, 1.2, demo_poses.shape[0], dtype=np.float32)
    np.save(tmp_path / "poses.npy", demo_poses)
    (tmp_path / "prompt.txt").write_text("drive through a city\n", encoding="utf-8")
    _patch_cv2(monkeypatch, height=464, width=832)
    monkeypatch.setattr(session.dist, "is_initialized", lambda: False)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(monkeypatch, "TestLingbot", derive_calls, pipeline_events)

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="TestLingbot",
            device="cpu",
            video_height=464,
            video_width=832,
            example_data_dir=tmp_path,
        )
    )

    runtime._initialize_sync()

    assert runtime._base_intrinsics is not None
    expected_intrinsics = session._transform_intrinsics(
        torch.from_numpy(demo_intrinsics).view(1, 4),
        height_org=session._INTRINSICS_REFERENCE_HEIGHT,
        width_org=session._INTRINSICS_REFERENCE_WIDTH,
        height_resize=464,
        width_resize=832,
        height_final=464,
        width_final=832,
    ).view(4)
    torch.testing.assert_close(runtime._base_intrinsics.cpu(), expected_intrinsics)
    _, expected_world_scale = session.preprocess_example_poses(demo_poses)
    assert runtime._world_scale == pytest.approx(expected_world_scale)
    assert any(
        event[0] == "initialize_cache" and event[1] == ("drive through a city",)
        for event in pipeline_events
    )


def test_initialize_sync_uses_demo_world_scale_as_offline_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "image.jpg").touch()
    (tmp_path / "prompt.txt").write_text("drive through a city\n", encoding="utf-8")
    _patch_cv2(monkeypatch, height=464, width=832)
    monkeypatch.setattr(session.dist, "is_initialized", lambda: False)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(monkeypatch, "TestLingbot", derive_calls, pipeline_events)

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="TestLingbot",
            device="cpu",
            video_height=464,
            video_width=832,
            example_data_dir=tmp_path,
            default_image_url=None,
            default_intrinsics_url=None,
            default_poses_url=None,
        )
    )

    runtime._initialize_sync()

    assert runtime._world_scale == pytest.approx(session._DEFAULT_WORLD_SCALE)


def test_reset_rollout_accepts_uploaded_prompt_and_first_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_minimal_assets(tmp_path)
    default_image = np.zeros((4, 4, 3), dtype=np.uint8)
    uploaded_image = np.full((4, 4, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(session.cv2, "imread", lambda path, flags: default_image.copy())
    monkeypatch.setattr(
        session.cv2, "imdecode", lambda data, flags: uploaded_image.copy()
    )
    monkeypatch.setattr(session.cv2, "cvtColor", lambda src, code: src)
    monkeypatch.setattr(
        session.cv2, "resize", lambda src, size, interpolation: src.copy()
    )
    monkeypatch.setattr(session.dist, "is_initialized", lambda: False)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(monkeypatch, "TestLingbot", derive_calls, pipeline_events)

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="TestLingbot",
            device="cpu",
            video_height=4,
            video_width=4,
            example_data_dir=tmp_path,
        )
    )
    runtime._initialize_sync()

    runtime._reset_rollout_sync(
        session_input=session.LingbotSessionInput(
            prompt="follow a coastal highway",
            first_frame_image_bytes=b"uploaded-image",
            first_frame_content_type="image/png",
        )
    )

    assert any(
        event[0] == "initialize_cache" and event[1] == ("follow a coastal highway",)
        for event in pipeline_events
    )


@pytest.mark.manual
def test_runtime_distributed_ops_use_world_cp_and_rank_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("Run under torchrun with WORLD_SIZE>=2.")
    if not torch.cuda.is_available():
        pytest.skip("Distributed Lingbot runtime test requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    _write_minimal_assets(tmp_path)
    _patch_cv2(monkeypatch, height=4, width=4)

    derive_calls: list[dict[str, Any]] = []
    pipeline_events: list[tuple[Any, ...]] = []
    _patch_pipeline_factory(
        monkeypatch, "DistributedTestLingbot", derive_calls, pipeline_events
    )

    runtime = session.LingbotInferenceRuntime(
        config=session.LingbotRuntimeConfig(
            config_name="DistributedTestLingbot",
            seed=100,
            context_parallel_size=world_size,
            device=str(device),
            video_height=4,
            video_width=4,
            example_data_dir=tmp_path,
        )
    )

    exit_sent = False
    result_shape: tuple[int, ...] | None = None
    try:
        if rank == 0:
            runtime._initialize_sync_all_ranks()
            runtime._reset_rollout_sync_all_ranks()
            num_frames = runtime.peek_next_chunk_num_frames()
            per_frame_keys = [frozenset() for _ in range(num_frames)]
            result = runtime._generate_chunk_sync_all_ranks(per_frame_keys)
            result_shape = tuple(result.video_chunk.shape)
            runtime._close_sync_all_ranks()
            runtime.send_exit_signal()
            exit_sent = True
        else:
            runtime.wait_for_termination()

        summaries: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(
            summaries,
            {
                "rank": rank,
                "derive_calls": derive_calls,
                "pipeline_events": pipeline_events,
                "result_shape": result_shape,
            },
        )

        if rank == 0:
            assert len(summaries) == world_size
            assert summaries[0] is not None
            assert summaries[0]["result_shape"] == (1, 1, 1, 3, 2, 2)
            for summary in summaries:
                assert summary is not None
                summary_rank = int(summary["rank"])
                calls = summary["derive_calls"]
                # Every rank must derive its own pipeline-config with a
                # seed shifted by rank so the AR-cache RNG stays unique.
                assert calls[0]["diffusion_model"]["seed"] == 100 + summary_rank
                events = [event[0] for event in summary["pipeline_events"]]
                assert "generate" in events
                assert "finalize" in events
    finally:
        if rank == 0 and not exit_sent and dist.is_initialized():
            runtime.send_exit_signal()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
