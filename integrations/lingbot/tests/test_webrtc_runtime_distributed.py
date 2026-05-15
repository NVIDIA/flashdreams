from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from lingbot.webrtc import session


def _write_minimal_assets(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "image.jpg").touch()
    np.save(
        data_dir / "intrinsics.npy", np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    )
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

    def to(self, *, device: torch.device) -> "_FakePipeline":
        self.events.append(("to", str(device)))
        return self

    def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> object:
        self.events.append(("initialize_cache", tuple(image.shape), str(image.device)))
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
    """Register a fake entry in ``PIPELINE_CONFIGS`` for ``config_name``
    and swap :func:`session.derive_config` with a capturing stub that
    returns a :class:`_FakePipelineConfig`.

    The runtime path under test is::

        derive_config(base_config=PIPELINE_CONFIGS[name], ...)
            .setup().to(device=...)
    """
    monkeypatch.setitem(session.PIPELINE_CONFIGS, config_name, object())

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
