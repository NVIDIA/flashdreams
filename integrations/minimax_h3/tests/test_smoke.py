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

"""CPU smoke tests for the MiniMax H3 runner plugin."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import tomli as tomllib
import torch
from flashdreams.infra.diffusion.transformer import Transformer
from flashdreams.infra.runner import RunnerConfig
from minimax_h3 import config as config_mod
from minimax_h3 import pipeline as h3_pipeline
from minimax_h3.config import (
    PIPELINE_MINIMAX_H3_FL2VA,
    PIPELINE_MINIMAX_H3_REF2VA,
    PIPELINE_MINIMAX_H3_T2VA,
    RUNNER_CONFIGS,
    RUNNER_MINIMAX_H3_FL2VA,
    RUNNER_MINIMAX_H3_REF2VA,
    RUNNER_MINIMAX_H3_T2VA,
)
from minimax_h3.constants import align_num_frames, validate_canvas
from minimax_h3.lora import convert_musubi_lora
from minimax_h3.model import MiniMaxH3DenoiseState, MiniMaxH3DiffusionModel
from minimax_h3.pipeline import MiniMaxH3Pipeline
from minimax_h3.references import parse_reference_specs
from minimax_h3.runner import (
    MiniMaxH3FL2VARunner,
    MiniMaxH3Ref2VARunner,
    MiniMaxH3RunnerConfig,
    MiniMaxH3T2VARunner,
)
from minimax_h3.scheduler import MiniMaxH3SchedulerConfig
from minimax_h3.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_name_mirrors_pipeline_name() -> None:
    """Runner names must match pipeline names for CLI discovery."""
    drifted = {
        slug: (config.runner_name, config.pipeline.name)
        for slug, config in RUNNER_CONFIGS.items()
        if config.runner_name != config.pipeline.name
    }
    assert not drifted, f"runner_name != pipeline.name: {drifted}"


def test_runners_have_descriptions() -> None:
    """Every registered runner must have a CLI description."""
    empty = [
        slug
        for slug, config in RUNNER_CONFIGS.items()
        if not config.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_entry_points_match_module_literals() -> None:
    """Package entry points must resolve to every registered runner literal."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        metadata = tomllib.load(handle)
    entries = metadata["project"]["entry-points"][ENTRY_POINT_GROUP]
    assert set(entries) == set(RUNNER_CONFIGS)

    for slug, target in entries.items():
        module_name, attribute = target.split(":", 1)
        assert module_name == "minimax_h3.config"
        config = cast(RunnerConfig, getattr(config_mod, attribute))
        assert config.runner_name == slug


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on importlib.metadata 3.10+ shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    """Installed plugin entry points must expose every registered runner."""
    from importlib.metadata import entry_points

    entries = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {
        entry.name for entry in entries if entry.value.startswith("minimax_h3.")
    }
    if not discovered:
        pytest.skip("plugin not installed; run uv sync from the repository root")
    assert discovered == set(RUNNER_CONFIGS)


def test_pipeline_config_constructs_without_loading_weights() -> None:
    """Construct every runtime pipeline without network or checkpoint access."""
    pipelines = {
        config.workflow: config.setup()
        for config in (
            PIPELINE_MINIMAX_H3_T2VA,
            PIPELINE_MINIMAX_H3_FL2VA,
            PIPELINE_MINIMAX_H3_REF2VA,
        )
    }
    assert set(pipelines) == {"t2va", "fl2va", "ref2va"}
    assert all(
        isinstance(pipeline, MiniMaxH3Pipeline) for pipeline in pipelines.values()
    )
    assert all(
        pipeline.config.model_id == "MiniMaxAI/MiniMax-H3"
        for pipeline in pipelines.values()
    )
    assert RUNNER_MINIMAX_H3_T2VA._target is MiniMaxH3T2VARunner
    assert RUNNER_MINIMAX_H3_FL2VA._target is MiniMaxH3FL2VARunner
    assert RUNNER_MINIMAX_H3_REF2VA._target is MiniMaxH3Ref2VARunner
    for config in (
        PIPELINE_MINIMAX_H3_T2VA,
        PIPELINE_MINIMAX_H3_FL2VA,
        PIPELINE_MINIMAX_H3_REF2VA,
    ):
        assert issubclass(config.diffusion_model.transformer._target, Transformer)


def test_low_ram_is_an_explicit_default_flag() -> None:
    """Default to crash-safe staging without enabling a third-party LoRA."""
    for runner in RUNNER_CONFIGS.values():
        assert isinstance(runner, MiniMaxH3RunnerConfig)
        assert runner.low_ram is True
        assert runner.lora is None


def test_native_bf16_gpu_path_is_default() -> None:
    """Keep the quality-preserving low-host-RAM path on the accelerator."""
    transformer = PIPELINE_MINIMAX_H3_FL2VA.diffusion_model.transformer
    assert isinstance(transformer, MiniMaxH3TransformerConfig)
    assert transformer.device == "cuda"
    assert transformer.sequential_cpu_offload is False


def test_musubi_lora_conversion_targets_native_layers(tmp_path: Path) -> None:
    """Convert all H3 block targets without introducing a default adapter."""
    from safetensors.torch import load_file, save_file

    source = tmp_path / "adapter.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    for block in range(50):
        for module in ("attn_qkv_proj", "attn_out_proj", "mlp_fc1", "mlp_fc2"):
            prefix = f"lora_unet_blocks_{block}_{module}"
            tensors[f"{prefix}.alpha"] = torch.tensor(2.0)
            tensors[f"{prefix}.lora_down.weight"] = torch.ones(2, 3)
            out_features = 6 if module == "attn_qkv_proj" else 4
            tensors[f"{prefix}.lora_up.weight"] = torch.ones(out_features, 2)
    save_file(tensors, source)

    converted = load_file(
        convert_musubi_lora(source, tmp_path / "converted.safetensors")
    )
    assert len(converted) == 600
    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in converted
    assert "transformer.transformer_blocks.0.attn.to_v.lora_B.weight" in converted
    assert "transformer.transformer_blocks.49.ff.net.2.lora_B.weight" in converted


def test_duration_and_canvas_contracts() -> None:
    """Align five seconds and reject non-H3 canvas dimensions."""
    assert align_num_frames(5.0) == 124
    validate_canvas(576, 768)
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_canvas(577, 768)


def test_runtime_cache_uses_stage_specific_checkpoints(tmp_path: Path) -> None:
    """Derive conditioning and denoised checkpoints beside the output."""
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    pipeline = PIPELINE_MINIMAX_H3_FL2VA.setup()
    cache = pipeline.initialize_cache(
        prompt="animate",
        image_path=image,
        last_image_path=None,
        references=(),
        output_path=tmp_path / "out.mp4",
        width=576,
        height=768,
        duration=5.0,
        steps=30,
        seed=42,
        low_ram=True,
        restart=False,
        attention="auto",
        lora=None,
        lora_weight_name=None,
        lora_scale=1.0,
    )
    assert cache.conditioning_checkpoint.name == "out.mp4.conditioning.safetensors"
    assert cache.latent_checkpoint.name == "out.mp4.latents.safetensors"


def test_generate_preserves_non_overlapping_stage_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep conditioning separate from native denoise and checkpoint timings."""
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    pipeline = PIPELINE_MINIMAX_H3_FL2VA.setup()
    cache = pipeline.initialize_cache(
        prompt="animate",
        image_path=image,
        last_image_path=None,
        references=(),
        output_path=tmp_path / "out.mp4",
        width=576,
        height=768,
        duration=5.0,
        steps=30,
        seed=42,
        low_ram=True,
        restart=True,
        attention="auto",
        lora=None,
        lora_weight_name=None,
        lora_scale=1.0,
    )

    def generate_low_ram(_: Any) -> torch.Tensor:
        cache.conditioning_seconds = 10.0
        cache.denoise_seconds = 20.0
        cache.denoise_prepare_seconds = 1.0
        cache.transformer_load_seconds = 4.0
        cache.denoise_compute_seconds = 14.0
        cache.denoise_cleanup_seconds = 1.0
        return torch.zeros(1)

    monkeypatch.setattr(pipeline, "_generate_low_ram", generate_low_ram)
    monkeypatch.setattr(
        pipeline,
        "_decode_video",
        lambda _cache, _latents: torch.zeros(1, 3, 1, 1),
    )
    monkeypatch.setattr(h3_pipeline, "_save_latents", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    pipeline.generate(0, cache)
    metrics = pipeline.finalize(0, cache)

    assert cache.latent_checkpoint_future is None
    assert metrics["conditioning_seconds"] == 10.0
    assert metrics["denoise_seconds"] == 20.0
    assert metrics["denoise_prepare_seconds"] == 1.0
    assert metrics["transformer_load_seconds"] == 4.0
    assert metrics["denoise_compute_seconds"] == 14.0
    assert metrics["denoise_cleanup_seconds"] == 1.0
    assert metrics["latent_checkpoint_seconds"] >= 0.0


def test_generate_overlaps_latent_checkpoint_with_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not put recovery-checkpoint I/O on generate's critical path."""
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    pipeline = PIPELINE_MINIMAX_H3_FL2VA.setup()
    cache = pipeline.initialize_cache(
        prompt="animate",
        image_path=image,
        last_image_path=None,
        references=(),
        output_path=tmp_path / "out.mp4",
        width=576,
        height=768,
        duration=5.0,
        steps=30,
        seed=42,
        low_ram=True,
        restart=True,
        attention="auto",
        lora=None,
        lora_weight_name=None,
        lora_scale=1.0,
    )
    checkpoint_started = threading.Event()
    release_checkpoint = threading.Event()

    def save_latents(*_args: Any) -> None:
        checkpoint_started.set()
        assert release_checkpoint.wait(timeout=5)

    def decode_video(*_args: Any) -> torch.Tensor:
        assert checkpoint_started.wait(timeout=5)
        return torch.zeros(1, 3, 1, 1)

    monkeypatch.setattr(pipeline, "_generate_low_ram", lambda _: torch.zeros(1))
    monkeypatch.setattr(pipeline, "_decode_video", decode_video)
    monkeypatch.setattr(h3_pipeline, "_save_latents", save_latents)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    try:
        pipeline.generate(0, cache)
        assert cache.latent_checkpoint_future is not None
        assert not cache.latent_checkpoint_future.done()
    finally:
        release_checkpoint.set()

    pipeline.finalize(0, cache)
    assert cache.latent_checkpoint_future is None
    cache.output_path.write_bytes(b"mp4")
    pipeline.mark_complete(cache)


def test_reference_parser_preserves_order_and_enforces_limits(tmp_path: Path) -> None:
    """Keep semantic reference order while rejecting unsupported requests."""
    image = tmp_path / "subject.png"
    video = tmp_path / "motion.mp4"
    audio = tmp_path / "voice.wav"
    for path in (image, video, audio):
        path.write_bytes(b"test")
    parsed = parse_reference_specs(
        (f"image:{image}", f"audio:{audio}", f"video:{video}")
    )
    assert [reference.kind for reference in parsed] == ["image", "audio", "video"]
    with pytest.raises(ValueError, match="paired"):
        parse_reference_specs((f"audio:{audio}",))
    with pytest.raises(ValueError, match="at most 3 video"):
        parse_reference_specs(tuple(f"video:{video}" for _ in range(4)))


def test_registered_workflows_validate_their_inputs(tmp_path: Path) -> None:
    """Reject cross-workflow media instead of silently selecting another model."""
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    common = {
        "prompt": "animate",
        "output_path": tmp_path / "out.mp4",
        "width": 512,
        "height": 768,
        "duration": 5.0,
        "steps": 30,
        "seed": 42,
        "low_ram": True,
        "restart": False,
        "attention": "auto",
        "lora": None,
        "lora_weight_name": None,
        "lora_scale": 1.0,
    }
    t2va = PIPELINE_MINIMAX_H3_T2VA.setup()
    cache = t2va.initialize_cache(
        image_path=None, last_image_path=None, references=(), **common
    )
    assert cache.workflow == "t2va"
    fl2va = PIPELINE_MINIMAX_H3_FL2VA.setup()
    cache = fl2va.initialize_cache(
        image_path=None, last_image_path=image, references=(), **common
    )
    assert cache.workflow == "fl2va"
    with pytest.raises(ValueError, match="requires --image-path"):
        fl2va.initialize_cache(
            image_path=None, last_image_path=None, references=(), **common
        )


def test_native_scheduler_matches_official_h3_euler() -> None:
    """Match the released H3 schedule and data-ward Euler update exactly."""
    from diffusers.schedulers.scheduling_minimax_h3 import (
        MiniMaxH3Scheduler as OfficialScheduler,
    )

    official: Any = OfficialScheduler(shift=12.0)
    official.set_timesteps(7, device="cpu")
    native = MiniMaxH3SchedulerConfig(num_inference_steps=7, shift=12.0).setup()
    sigmas, timesteps = native.schedule("cpu")
    torch.testing.assert_close(sigmas, official.sigmas)
    torch.testing.assert_close(timesteps, official.timesteps)

    sample = torch.randn(2, 3)
    flow = torch.randn_like(sample)
    expected = official.step(flow, official.timesteps[0], sample).prev_sample
    actual = native.step(sample, flow, timesteps[0], sigmas[0], sigmas[1])
    torch.testing.assert_close(actual, expected)


def test_row_timestep_plan_preserves_device_and_conditioning_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build packed row timesteps without materializing accelerator scalars."""
    state = MiniMaxH3DenoiseState(
        latents=torch.empty(0),
        audio_latents=torch.empty(0),
        prompt_embeds=torch.empty(0),
        position_ids=torch.empty(0),
        token_tags=torch.empty(0),
        video_indices=torch.tensor([0, 1, 4]),
        audio_indices=torch.tensor([2, 3]),
        text_indices=torch.tensor([5]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_latent_frames=0,
        latent_height=0,
        latent_width=0,
    )

    def reject_scalar_conversion(_tensor: torch.Tensor) -> float:
        raise AssertionError("row timestep construction converted a tensor to float")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__float__", reject_scalar_conversion)
        timesteps, indices = MiniMaxH3DiffusionModel._row_timesteps(
            state, torch.tensor(0.5), torch.tensor(0.25)
        )

    assert timesteps.device == state.video_indices.device
    torch.testing.assert_close(timesteps, torch.tensor([0.25, 0.5, 0.999, 1.0]))
    torch.testing.assert_close(indices, torch.tensor([2, 1, 3, 0, 1, 1]))
