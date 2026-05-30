# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import replace

import numpy as np
import omnidreams.interactive_drive.world_model.flashdreams_adapter as adapter_module
import torch
from omnidreams.interactive_drive.config import WorldModelProfileConfig
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    FlashdreamsWorldModelSession,
    _build_pipeline_config,
    _LazyRGBFrame,
    _select_config_name,
)
from omnidreams.interactive_drive.world_model.manifest import WorldModelManifest


class _FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.initialize_calls: list[dict[str, object]] = []
        self.initialize_from_embeddings_calls: list[dict[str, object]] = []
        self.precompute_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.finalize_calls: list[tuple[int, object]] = []
        self.release_calls = 0

    def get_num_frames(self, autoregressive_index: int) -> int:
        return 5 if autoregressive_index == 0 else 8

    def initialize_cache(self, **kwargs: object) -> str:
        self.initialize_calls.append(kwargs)
        return "cache"

    def initialize_cache_from_embeddings(self, **kwargs: object) -> str:
        self.initialize_from_embeddings_calls.append(kwargs)
        return "cache"

    def precompute_embeddings(self, **kwargs: object) -> dict[str, torch.Tensor | None]:
        self.precompute_calls.append(kwargs)
        return {
            "text_embeddings": torch.ones((1, 1, 2, 3), dtype=torch.float32),
            "image_embeddings": torch.ones((1, 1, 1, 2, 2, 2), dtype=torch.float32),
            "negative_text_embeddings": None,
        }

    def release_oneshot_encoders(self) -> None:
        self.release_calls += 1

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        frame_count = self.get_num_frames(int(kwargs["autoregressive_index"]))
        return torch.zeros((1, 1, frame_count, 3, 2, 3), dtype=torch.float32)

    def finalize(self, autoregressive_index: int, cache: object) -> None:
        self.finalize_calls.append((autoregressive_index, cache))


def _manifest() -> WorldModelManifest:
    return WorldModelManifest()


def test_select_config_name_uses_omnidreams_recipe_slugs() -> None:
    assert (
        _select_config_name(_manifest())
        == "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
    )
    assert (
        _select_config_name(replace(_manifest(), light_vae=False))
        == "omnidreams-sv-2steps-chunk2-loc6-vae-vae"
    )
    assert (
        _select_config_name(
            replace(
                _manifest(),
                encode_with_pixel_shuffle=True,
                local_attn_size=8,
                num_frames_per_block=16,
            )
        )
        == "omnidreams-sv-2steps-chunk4-loc8-pshuffle-lighttae"
    )


def test_build_pipeline_config_uses_manifest_native_dit_overrides() -> None:
    config = _build_pipeline_config(
        replace(
            _manifest(),
            skip_finalize_kv_cache=True,
            native_dit_acceleration="required",
            native_dit_verbose_build=True,
            native_dit_backend="bf16",
            native_dit_attention_backend="sparge",
            native_dit_sparge_topk=0.4,
            native_dit_sparge_hybrid_period=4,
            native_dit_sparge_hybrid_phase=1,
        ),
        profile=WorldModelProfileConfig(),
    )
    transformer_config = config.diffusion_model.transformer

    assert transformer_config.skip_finalize_kv_cache is True
    assert transformer_config.native_dit_acceleration == "required"
    assert transformer_config.native_dit_verbose_build is True
    assert transformer_config.native_dit_backend == "bf16"
    assert transformer_config.native_dit_attention_backend == "sparge"
    assert transformer_config.native_dit_sparge_topk == 0.4
    assert transformer_config.native_dit_sparge_hybrid_period == 4
    assert transformer_config.native_dit_sparge_hybrid_phase == 1


def test_session_uses_flashdreams_pipeline_for_rollout() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]
    next_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(8)]

    first = session.start(initial_rgb, first_condition_frames, "demo prompt")
    assert len(first) == 5
    assert len(fake_pipeline.initialize_calls) == 1
    assert fake_pipeline.initialize_calls[0]["text"] == [["demo prompt"]]
    assert tuple(fake_pipeline.initialize_calls[0]["image"].shape) == (1, 1, 1, 3, 2, 3)
    assert tuple(fake_pipeline.generate_calls[0]["hdmap"].shape) == (1, 1, 5, 3, 2, 3)
    assert fake_pipeline.generate_calls[0]["autoregressive_index"] == 0
    assert fake_pipeline.generate_calls[0]["cache"] == "cache"

    second = session.continue_generation(next_condition_frames)
    assert len(second) == 8
    assert fake_pipeline.finalize_calls == [(0, "cache")]
    assert fake_pipeline.generate_calls[1]["autoregressive_index"] == 1

    session.close()
    assert fake_pipeline.finalize_calls == [(0, "cache"), (1, "cache")]


def test_session_synchronizes_generated_frame_events_before_return(monkeypatch) -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup()
    sync_calls: list[list[object]] = []

    def fake_sync(frames: list[object]) -> None:
        sync_calls.append(frames)

    monkeypatch.setattr(adapter_module, "_synchronize_cuda_frame_event", fake_sync)

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]
    next_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(8)]

    first = session.start(initial_rgb, first_condition_frames, "demo prompt")
    second = session.continue_generation(next_condition_frames)

    assert sync_calls == [first, second]


def test_lazy_rgb_frame_exposes_tensor_before_host_materialization() -> None:
    frames = torch.arange(2 * 2 * 3 * 3, dtype=torch.uint8).reshape(2, 2, 3, 3)
    lazy = _LazyRGBFrame(frames, frame_index=1)

    tensor = lazy.to_cuda_tensor()

    assert torch.equal(tensor, frames[1])
    assert lazy.to_cuda_event() is None
    assert np.array_equal(lazy.to_numpy(), frames[1].numpy())


def test_session_offload_reuses_precomputed_embeddings_after_reset() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        offload_text_encoder=True,
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "demo prompt")
    session.reset()
    session.start(initial_rgb, first_condition_frames, "demo prompt")

    assert fake_pipeline.initialize_calls == []
    assert len(fake_pipeline.precompute_calls) == 1
    assert fake_pipeline.release_calls == 1
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 2
    assert fake_pipeline.initialize_from_embeddings_calls[0]["view_names"] == [
        "camera_front_wide_120fov"
    ]
    assert (
        fake_pipeline.initialize_from_embeddings_calls[1]["text_embeddings"]
        is (fake_pipeline.initialize_from_embeddings_calls[0]["text_embeddings"])
    )

    session.close()
