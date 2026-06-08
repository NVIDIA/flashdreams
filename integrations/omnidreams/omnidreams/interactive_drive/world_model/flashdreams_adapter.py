# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from loguru import logger
from omnidreams.interactive_drive.config import WorldModelProfileConfig
from omnidreams.interactive_drive.cuda_host_prefetch import CudaHostPrefetch
from omnidreams.interactive_drive.world_model.manifest import WorldModelManifest

PipelineFactory = Callable[[WorldModelManifest, WorldModelProfileConfig], Any]
_VIEW_NAMES = ["camera_front_wide_120fov"]
_LIGHTVAE_RECIPE = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
_LIGHTVAE_PERF_RECIPE = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
_LIGHTVAE_NATIVE_PERF_RECIPE = (
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf"
)


def _select_config_name(manifest: WorldModelManifest) -> str:
    """Map interactive-drive's single-view manifest knobs to a flashdreams recipe slug.

    Returns a key from ``omnidreams.config.OMNIDREAMS_CONFIGS``
    (i.e. the same slug ``flashdreams-run`` accepts as its first positional arg).
    """
    if manifest.upsampling_enabled:
        raise NotImplementedError(
            "flashdreams interactive-drive path does not support upsampling."
        )
    if manifest.sink_size != 0:
        raise NotImplementedError(
            "flashdreams interactive-drive path currently supports sink_size=0 only."
        )

    if manifest.encode_with_pixel_shuffle:
        if manifest.num_frames_per_block != 16:
            raise ValueError(
                "Single-view pixel-shuffle flashdreams checkpoints require 16-frame chunks."
            )
        if manifest.local_attn_size != 8:
            raise ValueError(
                "Single-view pixel-shuffle flashdreams checkpoints require local_attn_size=8."
            )
        return "omnidreams-sv-2steps-chunk4-loc8-pshuffle-lighttae"

    if manifest.local_attn_size != 6:
        raise ValueError(
            "Single-view VAE flashdreams checkpoints require local_attn_size=6."
        )
    if manifest.light_vae:
        if manifest.num_frames_per_block != 8:
            raise ValueError(
                "The light-VAE flashdreams recipe currently supports 8-frame chunks."
            )
        return _LIGHTVAE_RECIPE
    if manifest.num_frames_per_block == 8:
        return "omnidreams-sv-2steps-chunk2-loc6-vae-vae"
    if manifest.num_frames_per_block == 12:
        return "omnidreams-sv-2steps-chunk3-loc6-vae-vae"
    raise ValueError("Full-VAE flashdreams recipes support 8- or 12-frame chunks.")


def _pipeline_config_log_line(
    config: Any,
    *,
    config_name: str,
    base_config_name: str,
) -> str:
    """Summarize resolved pipeline knobs without dumping the full config tree."""
    transformer = config.diffusion_model.transformer
    scheduler = config.diffusion_model.scheduler
    encoder = config.encoder
    image_encoder = config.image_encoder
    return (
        "[flashdreams-session] resolved pipeline config "
        f"selected_recipe={config_name} "
        f"base_recipe={base_config_name} "
        f"pipeline_name={config.name} "
        f"native_dit={transformer.native_dit_acceleration} "
        f"dit_backend={transformer.native_dit_backend} "
        f"dit_attn={transformer.native_dit_attention_backend} "
        f"compile_network={transformer.compile_network} "
        f"use_cuda_graph={transformer.use_cuda_graph} "
        f"denoising_steps={list(scheduler.denoising_timesteps)} "
        f"encoder_native_vae={encoder.native_vae_acceleration} "
        f"image_encoder_native_vae={image_encoder.native_vae_acceleration} "
        f"native_vae_backend={encoder.native_vae_backend}"
    )


def _build_pipeline_config(
    manifest: WorldModelManifest, profile: WorldModelProfileConfig
) -> Any:
    try:
        from omnidreams.config import OMNIDREAMS_CONFIGS

        from flashdreams.infra.config import derive_config
        from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The flashdreams and flashdreams-omnidreams packages are required "
            "for the omnidreams backend. Run `uv sync --package "
            "omnidreams-interactive-drive` from the flashdreams workspace "
            "root, or otherwise install an environment where "
            "`import flashdreams` and `import omnidreams` succeed."
        ) from exc

    config_name = _select_config_name(manifest)
    seed = (
        42
        if manifest.seed_for_every_rollout is None
        else int(manifest.seed_for_every_rollout)
    )

    # The lightvae chassis maps to the perf preset (use_compile + cuda_graph
    # on every encoder/decoder). ``OMNIDREAMS_CONFIGS`` values are shared
    # global instances, so use ``derive_config`` to get a deep-copied
    # override-applied instance instead of mutating the global.
    transformer_overrides = _transformer_overrides(manifest)
    base_config_name = _base_config_name(config_name, manifest)
    base = OMNIDREAMS_CONFIGS[base_config_name]
    config = derive_config(
        base,
        enable_sync_and_profile=bool(profile.enabled),
        **_native_vae_overrides(manifest),
        diffusion_model=dict(
            seed=seed,
            transformer=transformer_overrides,
        ),
    )
    scheduler_uses_manifest_steps = False

    if not scheduler_uses_manifest_steps and hasattr(
        config.diffusion_model, "scheduler"
    ):
        scheduler = config.diffusion_model.scheduler
        if isinstance(scheduler, FlowMatchSchedulerConfig):
            config = derive_config(
                config,
                diffusion_model=dict(
                    scheduler=dict(
                        denoising_timesteps=list(manifest.denoising_steps),
                        num_inference_steps=len(manifest.denoising_steps),
                    ),
                ),
            )
            scheduler_uses_manifest_steps = True
    if not scheduler_uses_manifest_steps and manifest.denoising_steps != [1000, 450]:
        raise NotImplementedError(
            f"{config_name} uses flashdreams default denoising steps [1000, 450]; "
            f"got {manifest.denoising_steps}."
        )
    logger.info(
        _pipeline_config_log_line(
            config,
            config_name=config_name,
            base_config_name=base_config_name,
        ),
    )
    return config


def _base_config_name(config_name: str, manifest: WorldModelManifest) -> str:
    if manifest.native_vae_encoder != "disabled":
        if config_name != _LIGHTVAE_RECIPE:
            raise ValueError("native_vae_encoder=fp8 requires light_vae=true.")
        return _LIGHTVAE_NATIVE_PERF_RECIPE
    if config_name == _LIGHTVAE_RECIPE:
        return _LIGHTVAE_PERF_RECIPE
    return config_name


def _native_vae_overrides(manifest: WorldModelManifest) -> dict[str, object]:
    if manifest.native_vae_encoder == "disabled":
        return {}
    if manifest.native_vae_encoder != "fp8":
        raise ValueError(
            f"Unsupported native_vae_encoder={manifest.native_vae_encoder!r}"
        )

    common: dict[str, object] = {
        "native_vae_acceleration": "required",
        "native_vae_backend": "fp8",
    }
    if manifest.native_vae_fp8_state_path is not None:
        common["native_vae_fp8_state_path"] = str(manifest.native_vae_fp8_state_path)
    return {
        "image_encoder": dict(common),
        "encoder": dict(common),
    }


def _transformer_overrides(manifest: WorldModelManifest) -> dict[str, object]:
    return {
        "skip_finalize_kv_cache": manifest.skip_finalize_kv_cache,
        "compile_network": manifest.compile_net,
        "native_dit_acceleration": manifest.native_dit_acceleration,
        "native_dit_build_root": manifest.native_dit_build_root,
        "native_dit_max_jobs": manifest.native_dit_max_jobs,
        "native_dit_verbose_build": manifest.native_dit_verbose_build,
        "native_dit_backend": manifest.native_dit_backend,
        "native_dit_attention_backend": manifest.native_dit_attention_backend,
        "native_dit_sparge_topk": manifest.native_dit_sparge_topk,
        "native_dit_sparge_hybrid_period": manifest.native_dit_sparge_hybrid_period,
        "native_dit_sparge_hybrid_phase": manifest.native_dit_sparge_hybrid_phase,
    }


def _setup_pipeline_from_config(config: Any, manifest: WorldModelManifest) -> Any:
    pipeline = config.setup().to(device=torch.device(manifest.device))
    if manifest.seed_for_every_rollout is None:
        # Let repeated fresh rollouts vary when the manifest does not pin a seed.
        pipeline.diffusion_model.config.seed = None
    return pipeline


def _precompute_embeddings_from_config(
    config: Any,
    manifest: WorldModelManifest,
    *,
    initial_rgb: object,
    prompt: str,
) -> dict[str, torch.Tensor | None]:
    text_encoder_config = getattr(config, "text_encoder", None)
    image_encoder_config = getattr(config, "image_encoder", None)
    if text_encoder_config is None or image_encoder_config is None:
        raise RuntimeError(
            "--offload-text-encoder requires flashdreams text_encoder and "
            "image_encoder configs, but one of those slots is None."
        )

    try:
        from omnidreams.constants import NEGATIVE_PROMPT
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The flashdreams-omnidreams package is required for --offload-text-encoder."
        ) from exc

    device = torch.device(manifest.device)
    image = _initial_rgb_tensor(initial_rgb, device=device)
    text = [[prompt]]
    transformer_config = getattr(config.diffusion_model, "transformer", None)
    needs_negative_text = bool(
        getattr(transformer_config, "requires_negative_text_embeddings", False)
    )

    start = time.perf_counter()
    text_encoder = text_encoder_config.setup().to(device=device)
    image_encoder = image_encoder_config.setup().to(device=device)
    with torch.no_grad():
        text_embeddings = torch.stack(
            [text_encoder(prompt_row) for prompt_row in text], dim=0
        )
        image_embeddings = image_encoder(image)
        negative_text_embeddings = (
            torch.stack(
                [
                    text_encoder([NEGATIVE_PROMPT for _ in prompt_row])
                    for prompt_row in text
                ],
                dim=0,
            )
            if needs_negative_text
            else None
        )

    embeddings = {
        "text_embeddings": text_embeddings.cpu(),
        "image_embeddings": image_embeddings.cpu(),
        "negative_text_embeddings": (
            negative_text_embeddings.cpu()
            if negative_text_embeddings is not None
            else None
        ),
    }
    del (
        text_encoder,
        image_encoder,
        text_embeddings,
        image_embeddings,
        negative_text_embeddings,
    )
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "[flashdreams-session] offloaded one-shot encoders "
        f"precompute_ms={elapsed_ms:.1f} "
        f"text_shape={tuple(embeddings['text_embeddings'].shape)} "
        f"image_shape={tuple(embeddings['image_embeddings'].shape)}",
    )
    return embeddings


def _default_pipeline_factory(
    manifest: WorldModelManifest, profile: WorldModelProfileConfig
) -> Any:
    config = _build_pipeline_config(manifest, profile)
    return _setup_pipeline_from_config(config, manifest)


def _initial_rgb_tensor(frame: object, *, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(_rgb_hwc_uint8(frame))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    return _to_model_range(tensor, device=device)


def _to_model_range(tensor: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=torch.bfloat16)
    return tensor / 127.5 - 1.0


class FlashdreamsWorldModelSession:
    """Thin adapter from interactive-drive chunking to flashdreams AlpadreamsPipeline."""

    def __init__(
        self,
        manifest: WorldModelManifest,
        profile: WorldModelProfileConfig | None = None,
        *,
        offload_text_encoder: bool = False,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.manifest = manifest
        self._profile_config = profile or WorldModelProfileConfig()
        self._offload_text_encoder = bool(offload_text_encoder)
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._cache: Any | None = None
        self._precomputed_embeddings: dict[str, torch.Tensor | None] | None = None
        self._pending_finalization_index: int | None = None
        self._next_block_index = 0

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError(
                "warmup() must be called before rendering world-model chunks"
            )
        return self._pipeline

    @property
    def can_prewarm(self) -> bool:
        # The non-factory offload path defers its build to the first
        # prepare_for_scene so the one-shot encoders are freed before the
        # AR pipeline is allocated (peak-VRAM ordering); every other path
        # builds the pipeline eagerly with no scene needed.
        return self._pipeline_factory is not None or not self._offload_text_encoder

    def warmup_model(self) -> None:
        """Build the scene-independent diffusion pipeline (weights + compile).

        Called once per process. The non-factory offload path returns here
        and builds lazily in :meth:`prepare_for_scene` instead, so per-scene
        embeddings are computed and the one-shot encoders freed before the
        AR pipeline is allocated.
        """
        start = time.perf_counter()
        if self._pipeline_factory is not None:
            self._pipeline = self._pipeline_factory(self.manifest, self._profile_config)
        elif self._offload_text_encoder:
            return
        else:
            config = _build_pipeline_config(self.manifest, self._profile_config)
            self._pipeline = _setup_pipeline_from_config(config, self.manifest)
        self._validate_chunk_sizes()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"[flashdreams-session] model warmup runtime_ms={elapsed_ms:.1f}",
        )

    def prepare_for_scene(
        self, *, initial_rgb: object | None = None, prompt: str | None = None
    ) -> None:
        """Per-scene conditioning prep, run on every scene (re)load.

        Default path: a no-op. The pipeline keeps its text/image encoders
        and re-embeds the current prompt in ``initialize_cache`` on every
        rollout, so switching scenes needs no model-side work here.

        Offload path: the one-shot encoders are freed to save VRAM, so a
        new scene's prompt/first-frame cannot reuse the previous scene's
        cached embeddings. The factory (test) path recomputes them lazily
        on the next ``start``; the real path rebuilds the pipeline per
        scene (precompute embeddings -> free encoders -> build pipeline) to
        keep peak VRAM low. This is the only path that does not keep the
        model resident across scene changes.
        """
        if not self._offload_text_encoder:
            return
        self._precomputed_embeddings = None
        if self._pipeline_factory is not None:
            return
        if initial_rgb is None or prompt is None:
            raise RuntimeError(
                "offload_text_encoder requires the scene initial_rgb and prompt."
            )
        self._release_pipeline()
        config = _build_pipeline_config(self.manifest, self._profile_config)
        self._precomputed_embeddings = _precompute_embeddings_from_config(
            config,
            self.manifest,
            initial_rgb=initial_rgb,
            prompt=prompt,
        )
        config = replace(config, text_encoder=None, image_encoder=None)
        self._pipeline = _setup_pipeline_from_config(config, self.manifest)
        self._validate_chunk_sizes()

    def _validate_chunk_sizes(self) -> None:
        first_chunk_frames = self.pipeline.get_num_frames(0)
        # Flashdreams indexes the first post-initial chunk as AR step 1; this
        # is the steady-state frame count that interactive-drive loops over.
        steady_chunk_frames = self.pipeline.get_num_frames(1)
        if first_chunk_frames != 5:
            raise ValueError(
                "flashdreams initial chunk size does not match interactive-drive's first chunk: "
                f"{first_chunk_frames} vs 5"
            )
        if steady_chunk_frames != self.manifest.num_frames_per_block:
            raise ValueError(
                "flashdreams steady-state chunk size does not match the manifest: "
                f"{steady_chunk_frames} vs {self.manifest.num_frames_per_block}"
            )

    def _release_pipeline(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        device = torch.device(self.manifest.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def start(
        self,
        initial_rgb: object,
        condition_frames: list[object],
        prompt: str,
    ) -> list[object]:
        expected_frames = self.pipeline.get_num_frames(0)
        if len(condition_frames) != expected_frames:
            raise ValueError(
                "First condition chunk length does not match flashdreams initial chunk size: "
                f"{len(condition_frames)} vs {expected_frames}"
            )

        start = time.perf_counter()
        with torch.no_grad():
            self._cache = self._initialize_cache(initial_rgb, prompt)
            video = self.pipeline.generate(
                autoregressive_index=0,
                cache=self._cache,
                hdmap=self._condition_tensor(condition_frames),
            )
            model_frames = self._video_tensor_to_frames(video)
            _synchronize_cuda_frame_event(model_frames)
        self._pending_finalization_index = 0
        self._next_block_index = 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"[flashdreams-session] start total_ms={elapsed_ms:.1f}")
        return model_frames

    def continue_generation(self, condition_frames: list[object]) -> list[object]:
        if self._cache is None:
            raise RuntimeError("start() must be called before continue_generation()")
        expected_frames = self.pipeline.get_num_frames(self._next_block_index)
        if len(condition_frames) != expected_frames:
            raise ValueError(
                "Condition chunk length does not match flashdreams steady-state chunk size: "
                f"{len(condition_frames)} vs {expected_frames}"
            )

        start = time.perf_counter()
        with torch.no_grad():
            if self._pending_finalization_index is not None:
                self.pipeline.finalize(self._pending_finalization_index, self._cache)
                self._pending_finalization_index = None
            video = self.pipeline.generate(
                autoregressive_index=self._next_block_index,
                cache=self._cache,
                hdmap=self._condition_tensor(condition_frames),
            )
            model_frames = self._video_tensor_to_frames(video)
            _synchronize_cuda_frame_event(model_frames)
        block_index = self._next_block_index
        self._pending_finalization_index = block_index
        self._next_block_index += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if block_index <= 3 or elapsed_ms > 500.0:
            logger.info(
                f"[flashdreams-session] continue block_index={block_index} total_ms={elapsed_ms:.1f}",
            )
        return model_frames

    def reset(self, *, clear_precomputed_embeddings: bool = False) -> None:
        self._cache = None
        self._pending_finalization_index = None
        self._next_block_index = 0
        if clear_precomputed_embeddings:
            self._precomputed_embeddings = None
            logger.info(
                "[flashdreams-session] reset scene conditioning; "
                "will rerun text/image encoders for the next scene",
            )

    def close(self) -> None:
        if self._cache is not None and self._pending_finalization_index is not None:
            self.pipeline.finalize(self._pending_finalization_index, self._cache)
            self._pending_finalization_index = None
        self._cache = None
        self._pipeline = None

    def _initialize_cache(self, initial_rgb: object, prompt: str) -> Any:
        if self._offload_text_encoder:
            embeddings = self._ensure_precomputed_embeddings(initial_rgb, prompt)
            initialize_cache_from_embeddings = getattr(
                self.pipeline, "initialize_cache_from_embeddings", None
            )
            if not callable(initialize_cache_from_embeddings):
                raise RuntimeError(
                    "offload_text_encoder requires flashdreams initialize_cache_from_embeddings()."
                )
            return initialize_cache_from_embeddings(
                text_embeddings=embeddings["text_embeddings"],
                image_embeddings=embeddings["image_embeddings"],
                negative_text_embeddings=embeddings["negative_text_embeddings"],
                view_names=_VIEW_NAMES,
            )

        return self.pipeline.initialize_cache(
            text=[[prompt]],
            image=self._initial_rgb_tensor(initial_rgb),
            view_names=_VIEW_NAMES,
        )

    def _ensure_precomputed_embeddings(
        self, initial_rgb: object, prompt: str
    ) -> dict[str, torch.Tensor | None]:
        if self._precomputed_embeddings is not None:
            return self._precomputed_embeddings

        precompute_embeddings = getattr(self.pipeline, "precompute_embeddings", None)
        if not callable(precompute_embeddings):
            raise RuntimeError(
                "offload_text_encoder requires flashdreams precompute_embeddings()."
            )

        embeddings = precompute_embeddings(
            text=[[prompt]],
            image=self._initial_rgb_tensor(initial_rgb),
        )
        self._precomputed_embeddings = {
            "text_embeddings": embeddings["text_embeddings"].cpu(),
            "image_embeddings": embeddings["image_embeddings"].cpu(),
            "negative_text_embeddings": (
                embeddings["negative_text_embeddings"].cpu()
                if embeddings.get("negative_text_embeddings") is not None
                else None
            ),
        }
        release_oneshot_encoders = getattr(
            self.pipeline, "release_oneshot_encoders", None
        )
        if callable(release_oneshot_encoders):
            release_oneshot_encoders()
            logger.info("[flashdreams-session] release_oneshot_encoders done")
        return self._precomputed_embeddings

    def _initial_rgb_tensor(self, initial_rgb: object) -> torch.Tensor:
        return _initial_rgb_tensor(initial_rgb, device=self.pipeline.device)

    def _condition_tensor(self, condition_frames: Sequence[object]) -> torch.Tensor:
        cuda_video = _condition_cuda_video(condition_frames)
        if cuda_video is not None:
            tensor = cuda_video.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
            return self._to_model_range(tensor)
        video = np.stack([_rgb_hwc_uint8(frame) for frame in condition_frames], axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(video))
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
        return self._to_model_range(tensor)

    def _to_model_range(self, tensor: torch.Tensor) -> torch.Tensor:
        return _to_model_range(tensor, device=self.pipeline.device)

    @staticmethod
    def _video_tensor_to_frames(video: torch.Tensor) -> list[object]:
        if video.ndim != 6:
            raise ValueError(
                f"Expected [B,V,T,3,H,W] video tensor, got shape {tuple(video.shape)}"
            )
        frames = video[0, 0]
        if frames.dtype != torch.uint8:
            frames = frames.clamp(-1.0, 1.0)
            frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).contiguous()
        source_event = None
        if frames.is_cuda:
            source_event = torch.cuda.Event()
            source_event.record(torch.cuda.current_stream(frames.device))
        return [
            _LazyRGBFrame(frames, frame_index, source_event=source_event)
            for frame_index in range(frames.shape[0])
        ]


class _LazyRGBFrame:
    """Defer GPU-to-host copies until the presenter consumes each frame."""

    def __init__(
        self,
        frames_hwc_uint8: torch.Tensor,
        frame_index: int,
        *,
        source_event: object | None = None,
    ) -> None:
        self._frames_hwc_uint8: torch.Tensor | None = frames_hwc_uint8
        self._frame_index = int(frame_index)
        self._source_event = source_event
        self._host: np.ndarray | None = None
        self._prefetch: CudaHostPrefetch | None = None

    def prefetch_to_numpy(self) -> None:
        if (
            self._host is not None
            or self._prefetch is not None
            or self._frames_hwc_uint8 is None
        ):
            return
        frame = self._frames_hwc_uint8[self._frame_index].detach()
        prefetch = CudaHostPrefetch(frame, source_event=self._source_event)
        if prefetch.start():
            self._prefetch = prefetch

    def to_numpy(self) -> np.ndarray:
        if self._host is None:
            if self._prefetch is not None:
                self._host = self._prefetch.to_numpy()
                self._prefetch = None
                self._frames_hwc_uint8 = None
                return self._host
            if self._frames_hwc_uint8 is None:
                raise RuntimeError(
                    "Lazy RGB frame lost its source tensor before materialization."
                )
            frame = self._frames_hwc_uint8[self._frame_index].detach().cpu().numpy()
            self._host = np.ascontiguousarray(frame, dtype=np.uint8)
            self._frames_hwc_uint8 = None
        return self._host

    def to_cuda_tensor(self) -> torch.Tensor:
        if self._frames_hwc_uint8 is None:
            raise RuntimeError("Lazy RGB frame was already materialized on the host.")
        return self._frames_hwc_uint8[self._frame_index]

    def to_cuda_event(self) -> object | None:
        if self._frames_hwc_uint8 is None:
            return None
        return self._source_event

    def __array__(
        self,
        dtype: object | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        array = self.to_numpy()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            return np.array(array, copy=True)
        return array


def _rgb_hwc_uint8(frame: object) -> np.ndarray:
    return np.ascontiguousarray(
        np.array(np.asarray(frame, dtype=np.uint8)[..., :3], copy=True)
    )


def _condition_cuda_video(condition_frames: Sequence[object]) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    device: torch.device | None = None
    for frame in condition_frames:
        to_cuda_tensor = getattr(frame, "to_cuda_tensor", None)
        if not callable(to_cuda_tensor):
            return None
        try:
            tensor = to_cuda_tensor()
        except RuntimeError:
            return None
        if (
            not torch.is_tensor(tensor)
            or not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or tensor.ndim != 3
            or tensor.shape[-1] < 3
        ):
            return None
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            return None

        to_cuda_event = getattr(frame, "to_cuda_event", None)
        event = to_cuda_event() if callable(to_cuda_event) else None
        if event is not None:
            torch.cuda.current_stream(tensor.device).wait_event(event)
        rgb = tensor[..., :3]
        tensors.append(rgb if rgb.is_contiguous() else rgb.contiguous())

    if not tensors:
        return None
    return torch.stack(tensors, dim=0)


def _synchronize_cuda_frame_event(frames: Sequence[object]) -> None:
    for frame in frames:
        to_cuda_event = getattr(frame, "to_cuda_event", None)
        event = to_cuda_event() if callable(to_cuda_event) else None
        if event is None:
            continue
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()
