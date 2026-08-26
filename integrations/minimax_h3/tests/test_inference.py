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

"""CPU contracts for staged, Diffusers-free MiniMax H3 inference."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import minimax_h3.inference as inference_module
import pytest
import torch
from minimax_h3.inference import (
    DefaultMiniMaxH3Resources,
    MiniMaxH3InferenceConfig,
    MiniMaxH3InferenceEngine,
    MiniMaxH3InferenceRequest,
    MiniMaxH3Workflow,
    validate_execution_capacity,
)
from minimax_h3.latent_checkpoint import MiniMaxH3LatentCheckpointStore
from minimax_h3.model import (
    MiniMaxH3DenoiseProgress,
    MiniMaxH3JointLatents,
)
from minimax_h3.reference_conditioning import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
)
from PIL import Image
from torch import nn

from flashdreams.runtime_v2.audio_output import AudioOutput

pytestmark = pytest.mark.ci_cpu


class _Tokenizer:
    """Return small deterministic ids for prompts and reference labels."""

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        """Tokenize one presentation fragment without special tokens."""
        assert not add_special_tokens
        if text == "prompt":
            return {"input_ids": [10, 11]}
        return {"input_ids": [12 + len(text) % 100]}

    def convert_tokens_to_ids(self, token: str) -> int:
        """Resolve the four Qwen vision sentinel tokens."""
        return {
            "<|vision_start|>": 900,
            "<|image_pad|>": 901,
            "<|video_pad|>": 902,
            "<|vision_end|>": 903,
        }[token]


class _ImageProcessor:
    """Return one merged Qwen image token per supplied image."""

    merge_size = 2

    def __call__(
        self, *, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        """Create minimal correctly shaped image features."""
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros(len(images), 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _VideoProcessor:
    """Expose the Qwen temporal-patch contract for reference tests."""

    temporal_patch_size = 2


class _Processor:
    """Minimal Qwen multimodal processor used by the fake resources."""

    image_processor = _ImageProcessor()
    video_processor = _VideoProcessor()

    def create_mm_token_type_ids(self, batches: list[list[int]]) -> list[list[int]]:
        """Mark every fake presentation token as text for Qwen."""
        return [[0] * len(value) for value in batches]


class _QwenBase:
    """Return the requested hidden-state depth without a language head."""

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        """Create one fake layer-50 embedding row per input token."""
        rows = kwargs["input_ids"].shape[1]
        hidden = [torch.zeros(1, rows, 5120) for _ in range(51)]
        hidden[50] = torch.full((1, rows, 5120), 50.0)
        return SimpleNamespace(hidden_states=hidden)


class _TextEncoder(nn.Module):
    """Small module matching the Qwen conditioner surface."""

    def __init__(self) -> None:
        """Create a BF16 dtype anchor and 64-layer metadata."""
        super().__init__()
        self.register_parameter(
            "anchor", nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        )
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64))
        self.model = _QwenBase()

    @property
    def dtype(self) -> torch.dtype:
        """Return the conditioner computation dtype."""
        return torch.bfloat16


class _VideoVAE(nn.Module):
    """Shape-preserving fake native video VAE."""

    device = torch.device("cpu")

    def encode_condition_pixels(
        self, pixels: torch.Tensor, *, seed: int = 42
    ) -> torch.Tensor:
        """Encode pixels to the corresponding H3 latent-frame shape."""
        del seed
        frames = pixels.shape[2]
        latent_frames = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
        return torch.zeros(
            1,
            24,
            latent_frames,
            pixels.shape[3] // 16,
            pixels.shape[4] // 16,
        )

    def decode_output(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode H3 latent-frame arithmetic into unit-range pixels."""
        frames = (latents.shape[2] - 2) // 5 * 17 + 5
        return torch.full(
            (
                1,
                3,
                frames,
                latents.shape[3] * 16,
                latents.shape[4] * 16,
            ),
            0.25,
        )


class _AudioVAE(nn.Module):
    """Shape-preserving fake native audio VAE."""

    def encode_condition(self, samples: torch.Tensor) -> torch.Tensor:
        """Encode stereo samples into channel-major latent rows."""
        steps = max(1, (samples.shape[1] + 799) // 800)
        return torch.zeros(2 * steps, 32)

    def decode_output(self, latents: torch.Tensor) -> AudioOutput:
        """Decode every H3 latent step to 800 stereo samples."""
        return AudioOutput(
            samples=torch.zeros(2, latents.shape[2] * 800),
            sample_rate=32000,
        )


class _DiffusionModel(nn.Module):
    """Record resume use and return shape-derived joint latents."""

    def __init__(self, resumes: list[bool]) -> None:
        """Share resume observations with the resource factory."""
        super().__init__()
        self.resumes = resumes

    def generate_joint(
        self,
        state: Any,
        *,
        resume: MiniMaxH3DenoiseProgress | None,
        checkpoint: Any,
    ) -> MiniMaxH3JointLatents:
        """Checkpoint paired state and return generated-only streams."""
        self.resumes.append(resume is not None)
        if checkpoint is not None:
            checkpoint(
                MiniMaxH3DenoiseProgress(
                    video=state.latents,
                    audio=state.audio_latents,
                    next_step=1,
                )
            )
        return MiniMaxH3JointLatents(
            video=torch.zeros(
                1,
                24,
                state.num_latent_frames,
                state.latent_height,
                state.latent_width,
            ),
            audio=torch.zeros(2, 32, state.num_audio_latents),
        )


class _Resources:
    """Record staged heavyweight component lifetime events."""

    tokenizer = _Tokenizer()
    processor = _Processor()

    def __init__(self) -> None:
        """Initialize event and resume logs."""
        self.events: list[str] = []
        self.resumes: list[bool] = []

    def load_text_encoder(self) -> nn.Module:
        """Load the fake Qwen conditioner stage."""
        self.events.append("load:text")
        return _TextEncoder()

    def load_video_vae(self) -> nn.Module:
        """Load a fake native video VAE stage."""
        self.events.append("load:video_vae")
        return _VideoVAE()

    def load_audio_vae(self) -> nn.Module:
        """Load a fake native audio VAE stage."""
        self.events.append("load:audio_vae")
        return _AudioVAE()

    def load_diffusion_model(
        self,
        workflow: MiniMaxH3Workflow,
        num_inference_steps: int,
    ) -> nn.Module:
        """Load the requested fake workflow transformer stage."""
        self.events.append(f"load:transformer:{workflow}:{num_inference_steps}")
        return _DiffusionModel(self.resumes)

    def release(self, module: nn.Module) -> None:
        """Record release of a completed or failed stage."""
        self.events.append(f"release:{type(module).__name__}")

    def close(self) -> None:
        """Record release of shared metadata."""
        self.events.append("close")


class _FailingAudioResources(_Resources):
    """Fail the reference audio-VAE load after the video VAE is live."""

    def load_audio_vae(self) -> nn.Module:
        """Raise a deterministic allocation-style failure."""
        self.events.append("load:audio_vae")
        raise RuntimeError("audio stage failed")


def test_default_resources_resolve_pinned_subfolders_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve metadata and Qwen weights through one pinned local snapshot."""
    import huggingface_hub
    import minimax_h3.inference as inference_module

    snapshot_dir = tmp_path / "snapshot"
    processor_dir = snapshot_dir / "processor"
    processor_dir.mkdir(parents=True)
    (snapshot_dir / "tokenizer").mkdir()
    (snapshot_dir / "text_encoder").mkdir()
    for component in ("vae", "audio_vae", "transformer", "transformer_ref"):
        (snapshot_dir / component).mkdir()
    (processor_dir / "chat_template.json").write_text(
        '{"chat_template": "pinned template"}', encoding="utf-8"
    )
    downloads: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return str(snapshot_dir)

    class _LocalFactory:
        calls: list[tuple[Path, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: Any) -> object:
            cls.calls.append((path, kwargs))
            return object()

    class _TokenizerFactory(_LocalFactory):
        calls: list[tuple[Path, dict[str, Any]]] = []

    class _ImageFactory(_LocalFactory):
        calls: list[tuple[Path, dict[str, Any]]] = []

    class _VideoFactory(_LocalFactory):
        calls: list[tuple[Path, dict[str, Any]]] = []

    class _ProcessorFactory:
        kwargs: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            type(self).kwargs = kwargs

    class _EncoderFactory(nn.Module):
        calls: list[tuple[Path, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: Any) -> _EncoderFactory:
            cls.calls.append((path, kwargs))
            return cls()

    fake_transformers = ModuleType("transformers")
    setattr(fake_transformers, "AutoTokenizer", _TokenizerFactory)
    setattr(fake_transformers, "Qwen2VLImageProcessor", _ImageFactory)
    setattr(fake_transformers, "Qwen3VLVideoProcessor", _VideoFactory)
    setattr(fake_transformers, "Qwen3VLProcessor", _ProcessorFactory)
    setattr(fake_transformers, "Qwen3VLForConditionalGeneration", _EncoderFactory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    disk_checks: list[Path] = []

    def decreasing_disk(path: Path) -> SimpleNamespace:
        disk_checks.append(path)
        free_gib = 151 if len(disk_checks) == 1 else 1
        return SimpleNamespace(free=free_gib * 2**30)

    monkeypatch.setattr(inference_module.shutil, "disk_usage", decreasing_disk)

    resources = DefaultMiniMaxH3Resources(
        MiniMaxH3InferenceConfig(
            workflow="t2va",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
    )
    encoder = resources.load_text_encoder()

    assert len(disk_checks) == 1
    assert downloads[0]["allow_patterns"] == ["tokenizer/*", "processor/*"]
    assert downloads[1]["allow_patterns"] == ["text_encoder/*"]
    assert all(call["revision"] == resources.config.revision for call in downloads)
    assert all(call["cache_dir"] == str(tmp_path / "cache") for call in downloads)
    assert _TokenizerFactory.calls == [
        (snapshot_dir / "tokenizer", {"local_files_only": True})
    ]
    assert _ImageFactory.calls == [(processor_dir, {"local_files_only": True})]
    assert _VideoFactory.calls == [(processor_dir, {"local_files_only": True})]
    assert _ProcessorFactory.kwargs["chat_template"] == "pinned template"
    assert _ProcessorFactory.kwargs["tokenizer"] is resources.tokenizer
    assert _EncoderFactory.calls == [
        (
            snapshot_dir / "text_encoder",
            {
                "local_files_only": True,
                "dtype": torch.bfloat16,
                "device_map": {"": "cpu"},
                "low_cpu_mem_usage": True,
            },
        )
    ]
    assert not encoder.training


def test_native_components_load_from_the_selected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep native weight downloads on the filesystem selected by --cache-dir."""
    import huggingface_hub
    import minimax_h3.inference as inference_module

    snapshot_dir = tmp_path / "snapshot"
    processor_dir = snapshot_dir / "processor"
    processor_dir.mkdir(parents=True)
    (snapshot_dir / "tokenizer").mkdir()
    (processor_dir / "chat_template.json").write_text(
        '{"chat_template": "pinned template"}', encoding="utf-8"
    )
    for component in ("vae", "audio_vae", "transformer", "transformer_ref"):
        (snapshot_dir / component).mkdir()
    downloads: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return str(snapshot_dir)

    class _Factory:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> object:
            del args, kwargs
            return object()

    class _ProcessorFactory:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

    fake_transformers = ModuleType("transformers")
    setattr(fake_transformers, "AutoTokenizer", _Factory)
    setattr(fake_transformers, "Qwen2VLImageProcessor", _Factory)
    setattr(fake_transformers, "Qwen3VLVideoProcessor", _Factory)
    setattr(fake_transformers, "Qwen3VLProcessor", _ProcessorFactory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    calls: dict[str, list[dict[str, Any]]] = {
        "video": [],
        "audio": [],
        "transformer": [],
    }

    def config_factory(kind: str) -> type:
        class _Config:
            def __init__(self, **kwargs: Any) -> None:
                calls[kind].append(kwargs)

            def setup(self) -> nn.Identity:
                return nn.Identity()

        return _Config

    class _DiffusionConfig:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def setup(self) -> nn.Identity:
            return nn.Identity()

    monkeypatch.setattr(
        inference_module, "MiniMaxH3VideoVAEConfig", config_factory("video")
    )
    monkeypatch.setattr(
        inference_module, "MiniMaxH3AudioVAEConfig", config_factory("audio")
    )
    monkeypatch.setattr(
        inference_module,
        "MiniMaxH3TransformerConfig",
        config_factory("transformer"),
    )
    monkeypatch.setattr(
        inference_module, "MiniMaxH3DiffusionModelConfig", _DiffusionConfig
    )
    cache_dir = tmp_path / "selected-cache"
    resources = DefaultMiniMaxH3Resources(
        MiniMaxH3InferenceConfig(device="cpu", cache_dir=cache_dir)
    )

    resources.load_video_vae()
    resources.load_audio_vae()
    resources.load_diffusion_model("t2va", 2)
    resources.load_diffusion_model("ref2va", 2)

    assert calls["video"][0]["checkpoint_path"] == str(
        snapshot_dir / "vae" / "diffusion_pytorch_model.safetensors.index.json"
    )
    assert calls["audio"][0]["checkpoint_path"] == str(
        snapshot_dir / "audio_vae" / "diffusion_pytorch_model.safetensors"
    )
    assert [call["checkpoint_path"] for call in calls["transformer"]] == [
        str(
            snapshot_dir
            / "transformer"
            / "diffusion_pytorch_model.safetensors.index.json"
        ),
        str(
            snapshot_dir
            / "transformer_ref"
            / "diffusion_pytorch_model.safetensors.index.json"
        ),
    ]
    assert all(download["cache_dir"] == str(cache_dir) for download in downloads)


@pytest.mark.parametrize("stage", ["text", "video", "audio", "diffusion"])
def test_component_load_failure_reclaims_unreachable_allocations(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial constructor failure must not poison the next staged load."""
    import minimax_h3.inference as inference_module

    resources = object.__new__(DefaultMiniMaxH3Resources)
    resources.config = MiniMaxH3InferenceConfig(device="cpu")
    resources._snapshot_dir = tmp_path
    monkeypatch.setattr(
        DefaultMiniMaxH3Resources,
        "_component_dir",
        lambda _self, component: tmp_path / component,
    )
    cleanup: list[str] = []
    monkeypatch.setattr(inference_module.gc, "collect", lambda: cleanup.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: cleanup.append("empty_cache")
    )

    class _FailSetup:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def setup(self) -> nn.Module:
            raise RuntimeError(f"{stage} setup failed")

    if stage == "text":

        class _FailEncoder:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> nn.Module:
                del cls, args, kwargs
                raise RuntimeError("text setup failed")

        fake_transformers = ModuleType("transformers")
        setattr(fake_transformers, "Qwen3VLForConditionalGeneration", _FailEncoder)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        load = resources.load_text_encoder
    elif stage == "video":
        monkeypatch.setattr(inference_module, "MiniMaxH3VideoVAEConfig", _FailSetup)
        load = resources.load_video_vae
    elif stage == "audio":
        monkeypatch.setattr(inference_module, "MiniMaxH3AudioVAEConfig", _FailSetup)
        load = resources.load_audio_vae
    else:
        monkeypatch.setattr(
            inference_module,
            "MiniMaxH3TransformerConfig",
            lambda **_kwargs: object(),
        )
        monkeypatch.setattr(
            inference_module, "MiniMaxH3DiffusionModelConfig", _FailSetup
        )

        def load_diffusion() -> nn.Module:
            return resources.load_diffusion_model("t2va", 2)

        load = load_diffusion

    with pytest.raises(RuntimeError, match=f"{stage} setup failed"):
        load()

    assert cleanup == ["gc", "empty_cache"]


def test_staged_t2va_returns_synchronized_v2_media() -> None:
    """Keep every heavyweight stage disjoint and return one finite clip."""
    resources = _Resources()
    engine = MiniMaxH3InferenceEngine(
        MiniMaxH3InferenceConfig(device="cpu"), resources=resources
    )
    result = engine.generate(
        MiniMaxH3InferenceRequest(
            workflow="t2va",
            prompt="prompt",
            width=32,
            height=32,
            duration=15.0,
            num_inference_steps=2,
            seed=7,
        )
    )

    assert result.video.shape == (362, 3, 32, 32)
    assert float(result.video.mean()) == -0.5
    assert result.audio.samples.shape == (2, 482_400)
    assert result.audio.sample_rate == 32000
    assert result.metrics["aligned_frame_count"] == 362
    assert result.metrics["audio_sample_count"] == 482_400
    assert {
        "conditioning_s",
        "prepare_s",
        "denoise_s",
        "transformer_load_s",
        "denoise_compute_s",
        "transformer_release_s",
        "video_decode_s",
        "audio_decode_s",
        "total_s",
        "generated_fps",
        "peak_gpu_memory_gib",
    } <= result.metrics.keys()
    assert resources.events == [
        "load:text",
        "release:_TextEncoder",
        "load:transformer:t2va:2",
        "release:_DiffusionModel",
        "load:video_vae",
        "release:_VideoVAE",
        "load:audio_vae",
        "release:_AudioVAE",
    ]
    engine.close()
    engine.close()
    assert resources.events[-1] == "close"
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(
            MiniMaxH3InferenceRequest(
                workflow="t2va", prompt="prompt", width=32, height=32
            )
        )


def test_engine_rejects_a_workflow_that_was_not_preflighted() -> None:
    """Do not load a transformer omitted from the workflow capacity check."""
    resources = _Resources()
    engine = MiniMaxH3InferenceEngine(
        MiniMaxH3InferenceConfig(device="cpu", workflow="ref2va"),
        resources=resources,
    )

    with pytest.raises(ValueError, match="configured for ref2va, not t2va"):
        engine.generate(
            MiniMaxH3InferenceRequest(
                workflow="t2va",
                prompt="prompt",
                width=32,
                height=32,
                num_inference_steps=2,
            )
        )

    assert resources.events == []


def test_staged_engine_resumes_both_packed_streams(tmp_path: Path) -> None:
    """Publish one paired record and pass it back to the next denoise stage."""
    resources = _Resources()
    engine = MiniMaxH3InferenceEngine(
        MiniMaxH3InferenceConfig(device="cpu"), resources=resources
    )
    store = MiniMaxH3LatentCheckpointStore(work_dir=tmp_path, job_id="job-7")
    request = MiniMaxH3InferenceRequest(
        workflow="t2va",
        prompt="prompt",
        width=32,
        height=32,
        num_inference_steps=2,
        checkpoint_store=store,
    )

    engine.generate(request)
    engine.generate(request)

    assert store.path.is_file()
    assert resources.resumes == [False, True]


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_staged_conditioning_workflows_use_native_vaes(workflow: str) -> None:
    """Encode FL2VA and REF2VA media before loading their transformer."""
    resources = _Resources()
    engine = MiniMaxH3InferenceEngine(
        MiniMaxH3InferenceConfig(device="cpu"), resources=resources
    )
    image = Image.new("RGB", (32, 32), color=(64, 128, 192))
    options: dict[str, Any]
    if workflow == "fl2va":
        options = {"first_image": image}
    else:
        options = {"references": (MiniMaxH3ImageReference(image),)}

    result = engine.generate(
        MiniMaxH3InferenceRequest(
            workflow=workflow,  # ty: ignore[invalid-argument-type]
            prompt="prompt",
            width=32,
            height=32,
            duration=5.0,
            num_inference_steps=2,
            **options,
        )
    )

    assert result.video.shape == (124, 3, 32, 32)
    assert resources.events == [
        "load:text",
        "release:_TextEncoder",
        "load:video_vae",
        "release:_VideoVAE",
        f"load:transformer:{workflow}:2",
        "release:_DiffusionModel",
        "load:video_vae",
        "release:_VideoVAE",
        "load:audio_vae",
        "release:_AudioVAE",
    ]


def test_reference_audio_load_failure_releases_live_video_vae() -> None:
    """Release a live reference video VAE when the paired audio load fails."""
    resources = _FailingAudioResources()
    engine = MiniMaxH3InferenceEngine(
        MiniMaxH3InferenceConfig(device="cpu"), resources=resources
    )
    request = MiniMaxH3InferenceRequest(
        workflow="ref2va",
        prompt="prompt",
        width=32,
        height=32,
        num_inference_steps=2,
        references=(
            MiniMaxH3ImageReference(Image.new("RGB", (32, 32))),
            MiniMaxH3AudioReference(torch.zeros(2, 800), sample_rate=32000),
        ),
    )

    with pytest.raises(RuntimeError, match="audio stage failed"):
        engine.generate(request)

    assert resources.events[-3:] == [
        "load:video_vae",
        "load:audio_vae",
        "release:_VideoVAE",
    ]


def test_request_and_capacity_reject_unsupported_work_before_weights() -> None:
    """Validate workflows and devices without constructing default resources."""
    with pytest.raises(ValueError, match="does not accept keyframes"):
        MiniMaxH3InferenceRequest(
            workflow="t2va",
            prompt="prompt",
            width=32,
            height=32,
            first_image=Image.new("RGB", (32, 32)),
        )
    with pytest.raises(ValueError, match="first_image must be a PIL image"):
        MiniMaxH3InferenceRequest(
            workflow="fl2va",
            prompt="prompt",
            width=32,
            height=32,
            first_image=object(),  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="requires a first image"):
        MiniMaxH3InferenceRequest(
            workflow="fl2va", prompt="prompt", width=32, height=32
        )
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        validate_execution_capacity(MiniMaxH3InferenceConfig(device="cpu"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        MiniMaxH3InferenceConfig(checkpoint_min_free_gb=float("nan"))
    with pytest.raises(ValueError, match="unsupported MiniMax H3 workflow"):
        MiniMaxH3InferenceConfig(workflow="unknown")  # ty: ignore[invalid-argument-type]


def test_complete_cached_component_skips_download_capacity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not reject a usable pinned component because the disk is nearly full."""
    config = MiniMaxH3InferenceConfig(
        cache_dir=tmp_path,
        checkpoint_min_free_gb=150,
    )
    component_dir = (
        tmp_path
        / "models--MiniMaxAI--MiniMax-H3"
        / "snapshots"
        / config.revision
        / "audio_vae"
    )
    component_dir.mkdir(parents=True)
    (component_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"cached")
    monkeypatch.setattr(
        inference_module.shutil,
        "disk_usage",
        lambda _path: pytest.fail("complete cache must not inspect free disk"),
    )

    inference_module._validate_component_download_capacity(config, ("audio_vae",))


def test_missing_or_partial_component_requires_configured_free_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the selected cache filesystem before a missing shard is downloaded."""
    config = MiniMaxH3InferenceConfig(
        cache_dir=tmp_path,
        checkpoint_min_free_gb=2,
    )
    component_dir = (
        tmp_path
        / "models--MiniMaxAI--MiniMax-H3"
        / "snapshots"
        / config.revision
        / "transformer"
    )
    component_dir.mkdir(parents=True)
    (component_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"missing.safetensors"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        inference_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1 * 2**30),
    )

    with pytest.raises(RuntimeError, match=r"transformer.*2 GiB.*found 1.0 GiB"):
        inference_module._validate_component_download_capacity(config, ("transformer",))
