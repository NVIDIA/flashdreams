# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import omnidreams.interactive_drive.world_model.flashdreams_adapter as adapter_module
import pytest
import torch
from omnidreams.interactive_drive.config import WorldModelProfileConfig
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    FlashdreamsWorldModelSession,
    _build_pipeline_config,
    _LazyRGBFrame,
    _select_config_name,
)
from omnidreams.interactive_drive.world_model.manifest import WorldModelManifest
from omnidreams.interactive_drive.world_model.synthetic_fixture import (
    SyntheticWorldModelAssets,
)


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


class _FakeSyntheticPipeline(_FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(synthetic_text_max_length=7)
        self.decoder = SimpleNamespace(spatial_compression_ratio=8)
        network = SimpleNamespace(
            use_crossattn_projection=True,
            crossattn_proj_in_channels=11,
            crossattn_emb_channels=13,
            in_channels=5,
        )
        transformer_config = SimpleNamespace(
            network=network,
            batch_shape=(1,),
            num_views=1,
            dtype=torch.float32,
            requires_negative_text_embeddings=False,
        )
        transformer = SimpleNamespace(config=transformer_config)
        self.diffusion_model = SimpleNamespace(transformer=transformer)
        self.V_size = 1


def _manifest() -> WorldModelManifest:
    return WorldModelManifest()


def _contains_hf_url(value: object) -> bool:
    if isinstance(value, str):
        return "huggingface.co" in value
    if isinstance(value, (list, tuple, set)):
        return any(_contains_hf_url(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_hf_url(key) or _contains_hf_url(item)
            for key, item in value.items()
        )
    if hasattr(value, "__dict__"):
        return any(_contains_hf_url(item) for item in vars(value).values())
    return False


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


def test_build_pipeline_config_can_select_native_vae_encoder() -> None:
    config = _build_pipeline_config(
        replace(
            _manifest(),
            native_vae_encoder="fp8",
            native_vae_fp8_state_path=Path("/tmp/lightvae-fp8-state.pt"),
        ),
        profile=WorldModelProfileConfig(),
    )

    assert (
        config.name == "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf"
    )
    assert config.image_encoder.native_vae_acceleration == "required"
    assert config.image_encoder.native_vae_backend == "fp8"
    assert (
        config.image_encoder.native_vae_fp8_state_path == "/tmp/lightvae-fp8-state.pt"
    )
    assert config.encoder.native_vae_acceleration == "required"
    assert config.encoder.native_vae_backend == "fp8"


def test_build_pipeline_config_synthetic_swaps_only_weight_sources(
    monkeypatch,
    tmp_path,
) -> None:
    assets = SyntheticWorldModelAssets(
        encoder_checkpoint_path=tmp_path / "synthetic_lightvae_encoder.safetensors",
        decoder_checkpoint_path=tmp_path / "synthetic_lighttae_decoder.safetensors",
    )
    assets.encoder_checkpoint_path.touch()
    assets.decoder_checkpoint_path.touch()

    def fake_assets(*_args: object, **_kwargs: object) -> SyntheticWorldModelAssets:
        return assets

    monkeypatch.setattr(
        adapter_module,
        "build_synthetic_world_model_assets",
        fake_assets,
    )

    manifest = replace(
        _manifest(),
        synthetic_model=True,
        skip_finalize_kv_cache=True,
        native_dit_acceleration="required",
        native_dit_backend="bf16",
        native_dit_attention_backend="cudnn",
    )
    real = _build_pipeline_config(
        replace(manifest, synthetic_model=False),
        profile=WorldModelProfileConfig(),
    )
    synthetic = _build_pipeline_config(manifest, profile=WorldModelProfileConfig())

    real_transformer = real.diffusion_model.transformer
    synthetic_transformer = synthetic.diffusion_model.transformer
    assert synthetic_transformer.checkpoint_path is None
    assert synthetic.text_encoder is None
    assert synthetic.image_encoder is None
    assert synthetic.encoder.checkpoint_path == str(assets.encoder_checkpoint_path)
    assert synthetic.decoder.checkpoint_path == str(assets.decoder_checkpoint_path)
    assert synthetic.decoder.state_dict_transform is None
    assert synthetic.synthetic_text_max_length == real.text_encoder.max_length

    for field in (
        "compile_network",
        "use_cuda_graph",
        "skip_finalize_kv_cache",
        "native_dit_acceleration",
        "native_dit_backend",
        "native_dit_attention_backend",
    ):
        assert getattr(synthetic_transformer, field) == getattr(real_transformer, field)
    assert synthetic.encoder.use_compile == real.encoder.use_compile
    assert synthetic.encoder.use_cuda_graph == real.encoder.use_cuda_graph
    assert synthetic.decoder.use_compile == real.decoder.use_compile
    assert synthetic.decoder.use_cuda_graph == real.decoder.use_cuda_graph
    assert not _contains_hf_url(synthetic)


def test_native_vae_encoder_requires_light_vae_recipe() -> None:
    with pytest.raises(ValueError, match="native_vae_encoder=fp8 requires light_vae"):
        _build_pipeline_config(
            replace(_manifest(), light_vae=False, native_vae_encoder="fp8"),
            profile=WorldModelProfileConfig(),
        )


def test_session_uses_flashdreams_pipeline_for_rollout() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup_model()

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


def test_session_synthetic_model_initializes_cache_from_synthetic_embeddings() -> None:
    fake_pipeline = _FakeSyntheticPipeline()
    manifest = replace(_manifest(), synthetic_model=True, resolution_wh=(64, 32))
    session = FlashdreamsWorldModelSession(
        manifest,
        offload_text_encoder=True,
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    assert session.can_prewarm is True
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "demo prompt")

    assert fake_pipeline.initialize_calls == []
    assert fake_pipeline.precompute_calls == []
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 1
    call = fake_pipeline.initialize_from_embeddings_calls[0]
    assert tuple(call["text_embeddings"].shape) == (1, 1, 7, 11)
    assert tuple(call["image_embeddings"].shape) == (1, 1, 1, 5, 4, 8)
    assert call["negative_text_embeddings"] is None
    assert call["view_names"] == ["camera_front_wide_120fov"]


def test_session_synchronizes_generated_frame_events_before_return(monkeypatch) -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup_model()
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
    session.warmup_model()

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


def test_session_offload_reruns_embeddings_after_scene_conditioning_reset() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _manifest(),
        offload_text_encoder=True,
        pipeline_factory=lambda manifest, profile: fake_pipeline,
    )
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "clear prompt")
    session.reset(clear_precomputed_embeddings=True)
    session.start(initial_rgb, first_condition_frames, "snow prompt")

    assert len(fake_pipeline.precompute_calls) == 2
    assert fake_pipeline.precompute_calls[0]["text"] == [["clear prompt"]]
    assert fake_pipeline.precompute_calls[1]["text"] == [["snow prompt"]]
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 2
    assert (
        fake_pipeline.initialize_from_embeddings_calls[1]["text_embeddings"]
        is not fake_pipeline.initialize_from_embeddings_calls[0]["text_embeddings"]
    )

    session.close()
