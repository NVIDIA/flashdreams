# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import AbstractSet, Any, Callable, TypeVar

import cv2
import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from loguru import logger
from omnidreams.conditioning.conditioning_wrapper import (
    AV_POSITIVE_PROMPT,
    OmnidreamsConditioningState,
    OmnidreamsConditioningWrapper,
    TextPrompt,
)
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.scenes import (
    HF_DATASET_BROWSER_URL,
    SCENE_CLIPGT_DIRNAME,
    SCENE_FRAME_SUFFIXES,
    SCENE_FRAMES_DIRNAME,
    SCENE_IMAGE_SUFFIXES,
    SCENE_PROMPT_FILENAME,
    SCENE_VARIANT_DEFAULT,
    hf_hub_download_scene,
    hf_scenes_repo_id,
    prompt_variant_for_scene_variant,
    scenes_cache_root,
)
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    PoseSegment,
)
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
    WebRTCControlSignal,
    WebRTCStepResult,
)

_T = TypeVar("_T")
# Default scene (clear-weather base archive). Weather siblings are selected
# via OmnidreamsRuntimeConfig.scene_variant / the server's --scene-variant.
DEFAULT_WEBRTC_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
# Back-compat aliases for ``omnidreams.scenes`` constants used by external imports.
WEBRTC_SCENES_HF_BROWSER_URL = HF_DATASET_BROWSER_URL
WEBRTC_SCENE_IMAGE_SUFFIXES = SCENE_IMAGE_SUFFIXES


def _resolve_cuda_device(device_spec: str | torch.device) -> torch.device:
    """Resolve a device spec, filling in the active CUDA index when unspecified."""
    device = torch.device(device_spec)
    if device.type == "cuda" and device.index is None:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cuda:0"
        )
    return device


def _choose_existing_asset(
    directory: Path,
    *,
    exact_name: str | None = None,
    fallback_stems: tuple[str, ...] = (),
    fallback_prefixes: tuple[str, ...] = (),
    allowed_suffixes: AbstractSet[str] | None = None,
    preferred_stems: tuple[str, ...] = (),
) -> Path | None:
    if not directory.is_dir():
        return None

    if exact_name is not None:
        exact_path = directory / exact_name
        if exact_path.is_file() and (
            allowed_suffixes is None or exact_path.suffix.lower() in allowed_suffixes
        ):
            return exact_path

    candidates = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        if (
            path.stem in preferred_stems
            or path.stem in fallback_stems
            or any(path.stem.startswith(f"{prefix}-") for prefix in fallback_prefixes)
        ):
            candidates.append(path)

    if not candidates:
        return None

    preferred_order = {stem: index for index, stem in enumerate(preferred_stems)}
    return sorted(
        candidates,
        key=lambda path: (
            preferred_order.get(path.stem, len(preferred_order)),
            path.name,
        ),
    )[0]


def _camera_name_candidates(camera_name: str) -> tuple[str, ...]:
    """Colon/underscore spellings of ``camera_name`` (dataset uses underscores)."""
    underscore = camera_name.replace(":", "_")
    colon = camera_name.replace("_", ":")
    return tuple(dict.fromkeys((camera_name, underscore, colon)))


def _first_frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (2**63 - 1, path.name)


def _resolve_webrtc_first_frame(clipgt_dir: Path, camera_name: str) -> Path | None:
    """Earliest GT frame under ``clipgt/frames/<camera>/``, else ``None``.

    ``None`` when the bundle ships no such frames, so the caller can fall back
    to ``first_image.*``.
    """
    frames_root = clipgt_dir / SCENE_FRAMES_DIRNAME
    if not frames_root.is_dir():
        return None
    candidate_dirs = [
        frames_root / name
        for name in _camera_name_candidates(camera_name)
        if (frames_root / name).is_dir()
    ]
    if not candidate_dirs:
        # Fall back to any single camera directory present.
        candidate_dirs = [
            path for path in sorted(frames_root.iterdir()) if path.is_dir()
        ]
    for directory in candidate_dirs:
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SCENE_FRAME_SUFFIXES
        ]
        if frames:
            return sorted(frames, key=_first_frame_sort_key)[0]
    return None


def _resolve_webrtc_scene_assets(
    scene_dir: Path,
    *,
    prompt_filename: str,
    clipgt_dirname: str,
    camera_name: str = "camera_front_wide_120fov",
    variant: str = SCENE_VARIANT_DEFAULT,
) -> tuple[Path, Path, Path]:
    missing_assets = []
    clipgt_dir = scene_dir / clipgt_dirname
    if not clipgt_dir.is_dir():
        missing_assets.append(str(scene_dir / clipgt_dirname))
        clipgt_dir = None

    # Prefer the GT camera frame; fall back to ``first_image.*`` for bundles
    # with no per-camera frames.
    first_frame_path = (
        None
        if clipgt_dir is None
        else _resolve_webrtc_first_frame(clipgt_dir, camera_name)
    )
    if first_frame_path is None and clipgt_dir is not None:
        first_frame_path = _choose_existing_asset(
            clipgt_dir,
            fallback_stems=("first_image_1",),
            allowed_suffixes=WEBRTC_SCENE_IMAGE_SUFFIXES,
            preferred_stems=("first_image",),
        )
    if first_frame_path is None:
        missing_assets.append(
            f"frames/<camera>/*.jpeg or first_image.* under {clipgt_dir}/"
        )

    # Prompt matching the weather variant (``promptN.txt``); fall back to a
    # bare ``prompt.txt`` for older bundles.
    weather_prompt_stem = f"prompt{prompt_variant_for_scene_variant(variant)}"
    prompt_path = (
        None
        if clipgt_dir is None
        else _choose_existing_asset(
            clipgt_dir,
            fallback_stems=("prompt1", "prompt2", "prompt3", "prompt"),
            allowed_suffixes={".txt"},
            preferred_stems=(weather_prompt_stem, "prompt"),
        )
    )
    if prompt_path is None:
        missing_assets.append(f"{prompt_filename} under {clipgt_dir}/")

    if missing_assets:
        raise FileNotFoundError(
            "Missing Omnidreams WebRTC scene assets: " + ", ".join(missing_assets)
        )

    assert clipgt_dir is not None
    assert first_frame_path is not None
    assert prompt_path is not None
    return clipgt_dir, first_frame_path, prompt_path


def _safe_extract_zip(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(source) as zf:
        for member in zf.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or not member_path.parts
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise ValueError(
                    f"Unsafe archive member in {source}: {member.filename}"
                )
            target = destination / Path(*member_path.parts)
            target_resolved = target.resolve()
            if destination_root != target_resolved and destination_root not in (
                target_resolved.parents
            ):
                raise ValueError(
                    f"Archive member escapes destination: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _variant_dir_suffix(variant: str | None) -> str:
    """Cache subdir / filename suffix for ``variant`` (``""`` for default)."""
    slug = (variant or SCENE_VARIANT_DEFAULT).strip()
    return "" if slug in ("", SCENE_VARIANT_DEFAULT) else f"-{slug}"


def _extract_local_webrtc_scene_if_needed(
    scene_dir: Path,
    *,
    scene_uuid: str | None,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str,
) -> Path:
    """Extract the ``scene_uuid`` (+ variant) archive into the local layout."""
    if scene_uuid is None:
        return scene_dir

    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be non-empty when provided."
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    suffix = _variant_dir_suffix(variant)
    expected_names = (
        f"clipgt-{scene_uuid}{suffix}.usdz",
        f"{scene_uuid}{suffix}.usdz",
    )
    archive_path = _choose_existing_asset(scene_dir, exact_name=expected_names[0]) or (
        _choose_existing_asset(scene_dir, exact_name=expected_names[1])
    )
    if archive_path is None:
        # Prefer the variant suffix but accept the base archive too.
        archive_path = _choose_existing_asset(
            scene_dir,
            fallback_prefixes=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
            allowed_suffixes={".usdz"},
            preferred_stems=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
        )
    if archive_path is None:
        raise FileNotFoundError(
            "scene_uuid is set but no local USDZ archive was found in "
            f"{scene_dir}. Expected one of: {', '.join(expected_names)}."
        )

    normalized_scene_dir = scene_dir / f"{scene_uuid}{suffix}"
    normalized_clipgt_root = normalized_scene_dir / clipgt_dirname
    _safe_extract_zip(archive_path, normalized_clipgt_root)
    return normalized_scene_dir


def _ensure_hf_webrtc_scene_synced(
    scene_uuid: str,
    *,
    variant: str = SCENE_VARIANT_DEFAULT,
    prompt_filename: str = SCENE_PROMPT_FILENAME,
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME,
) -> Path:
    """Stage an HF scene variant into the WebRTC cache layout.

    Downloads ``scenes/clipgt-<uuid>[-<variant>].usdz`` and extracts it under
    ``FLASHDREAMS_CACHE_DIR/omnidreams-scenes/<uuid>[-<variant>]/clipgt/``. The
    per-uuid+variant directory coexists with the desktop demo's archive files
    in the same root.
    """
    del prompt_filename  # accepted for call-site symmetry; assets resolved later
    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be set."
    suffix = _variant_dir_suffix(variant)
    cache_root = scenes_cache_root()
    scene_dir = cache_root / f"{scene_uuid}{suffix}"
    lock_path = cache_root / ".locks" / f"{scene_uuid}{suffix}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        archive_path = hf_hub_download_scene(scene_uuid, variant)
        _safe_extract_zip(archive_path, scene_dir / clipgt_dirname)

    logger.info(
        "Synced Omnidreams WebRTC scene {} (variant {}) from Hugging Face ({}) to {}",
        scene_uuid,
        variant,
        hf_scenes_repo_id(),
        scene_dir,
    )
    return scene_dir


def _summarize_sdp_candidates(sdp: str) -> str:
    candidates = [
        line.removeprefix("a=candidate:")
        for line in sdp.splitlines()
        if line.startswith("a=candidate:")
    ]
    if not candidates:
        return "0 candidates"

    protocols: dict[str, int] = {}
    addresses: set[str] = set()
    endpoints: list[str] = []
    for candidate in candidates:
        parts = candidate.split()
        if len(parts) >= 5:
            protocols[parts[2].lower()] = protocols.get(parts[2].lower(), 0) + 1
            addresses.add(parts[4])
        if len(parts) >= 6:
            endpoints.append(f"{parts[2].lower()}://{parts[4]}:{parts[5]}")
    protocol_summary = ",".join(
        f"{key}={value}" for key, value in sorted(protocols.items())
    )
    address_summary = ",".join(sorted(addresses)[:8])
    if len(addresses) > 8:
        address_summary += f",+{len(addresses) - 8} more"
    endpoint_summary = ",".join(endpoints[:12])
    if len(endpoints) > 12:
        endpoint_summary += f",+{len(endpoints) - 12} more"
    return (
        f"{len(candidates)} candidates protocols=[{protocol_summary}] "
        f"addresses=[{address_summary}] endpoints=[{endpoint_summary}]"
    )


class OmnidreamsRuntimeError(RuntimeError):
    """Raised when the Omnidreams WebRTC runtime is used incorrectly."""


@dataclass(slots=True)
class OmnidreamsRuntimeConfig:
    pipeline_config_name: str = (
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    )
    scene_dir: Path | None = None
    scene_uuid: str | None = None
    # Weather variant slug (default/rain/snow): picks the sibling USDZ + prompt.
    scene_variant: str = SCENE_VARIANT_DEFAULT
    seed: int | None = 42
    device: str = "cuda:0"
    video_height: int = 704
    video_width: int = 1280
    fps: int = 30
    camera_name: str = "camera_front_wide_120fov"
    prompt_filename: str = SCENE_PROMPT_FILENAME
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME
    move_speed_per_s: float = 6.0
    rotate_speed_rad_per_s: float = float(np.deg2rad(35.0))
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0
    debug_serve_hdmaps: bool = False


class OmnidreamsInferenceRuntime:
    """Single-scene, single-view Omnidreams runtime for WebRTC control."""

    def __init__(self, config: OmnidreamsRuntimeConfig | None = None) -> None:
        self.config = config or OmnidreamsRuntimeConfig()
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        control_device = _resolve_cuda_device(self.config.device)

        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._wrapper: OmnidreamsConditioningWrapper | None = None
        self._state: OmnidreamsConditioningState | None = None
        self._renderer: Any | None = None
        self._scene_data: Any | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._camera_to_rig: torch.Tensor | None = None
        self._initial_ego_pose: np.ndarray | None = None
        self._next_timestamp_us: int = 0
        self._closed = False
        self._clipgt_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        # Pin every blocking runtime call to one OS thread: Omnidreams' CUDA
        # graph capture/replay state is thread-local, so spreading calls across
        # workers (e.g. asyncio.to_thread) crashes capture after a few chunks.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omnidreams-webrtc-runtime",
        )

        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=control_device,
            signal_type=WebRTCControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=WebRTCControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=WebRTCControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._wrapper is not None:
            return
        await self._run_on_runtime_thread(self._initialize_sync_all_ranks)

    async def reset_for_new_session(self) -> None:
        if self._closed:
            raise OmnidreamsRuntimeError("Runtime is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        await self._run_on_runtime_thread(self._reset_rollout_sync_all_ranks)

    async def close(self) -> None:
        self._closed = True
        try:
            await self._run_on_runtime_thread(self._close_sync_all_ranks)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        if self._closed:
            raise OmnidreamsRuntimeError("Session is closed.")
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")

        async with self._step_lock:
            if self._closed:
                raise OmnidreamsRuntimeError("Session is closed.")
            return await self._run_on_runtime_thread(
                self._generate_chunk_sync_all_ranks,
                segments,
                frame_times,
            )

    async def _run_on_runtime_thread(
        self,
        func: Callable[..., _T],
        *args: Any,
    ) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._runtime_thread_entry,
            func,
            args,
        )

    def _runtime_thread_entry(
        self,
        func: Callable[..., _T],
        args: tuple[Any, ...],
    ) -> _T:
        device = self._device
        if device is None:
            device = _resolve_cuda_device(self.config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return func(*args)

    def peek_next_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._state is None:
            return int(self._wrapper.initial_frame_chunk_size)
        return int(self._wrapper.frame_chunk_size)

    def peek_steady_chunk_num_frames(self) -> int:
        if self._wrapper is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        return int(self._wrapper.frame_chunk_size)

    @distributed_op(WebRTCControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(WebRTCControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(self) -> None:
        self._reset_rollout_sync()

    @distributed_op(WebRTCControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(WebRTCControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        self._close_sync()

    def _initialize_sync(self) -> None:
        if self._wrapper is not None:
            return

        init_t0 = time.perf_counter()
        cfg = self.config
        if cfg.scene_dir is None:
            scene_uuid = cfg.scene_uuid or DEFAULT_WEBRTC_SCENE_UUID
            scene_dir = _ensure_hf_webrtc_scene_synced(
                scene_uuid,
                variant=cfg.scene_variant,
                prompt_filename=cfg.prompt_filename,
                clipgt_dirname=cfg.clipgt_dirname,
            )
        else:
            scene_dir = _extract_local_webrtc_scene_if_needed(
                cfg.scene_dir,
                scene_uuid=cfg.scene_uuid,
                variant=cfg.scene_variant,
                clipgt_dirname=cfg.clipgt_dirname,
            )

        cfg.scene_dir = scene_dir
        clipgt_dir, first_frame_path, prompt_path = _resolve_webrtc_scene_assets(
            scene_dir,
            prompt_filename=cfg.prompt_filename,
            clipgt_dirname=cfg.clipgt_dirname,
            camera_name=cfg.camera_name,
            variant=cfg.scene_variant,
        )
        if cfg.pipeline_config_name not in OMNIDREAMS_CONFIGS:
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unknown pipeline_config_name={cfg.pipeline_config_name!r}. "
                f"Supported: {supported}"
            )

        pipeline_cfg = OMNIDREAMS_CONFIGS[cfg.pipeline_config_name]
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        if not isinstance(transformer_cfg, CosmosTransformerConfig):
            raise TypeError(
                "Omnidreams WebRTC requires a CosmosTransformerConfig pipeline."
            )
        if transformer_cfg.num_views != 1:
            raise ValueError(
                "Omnidreams WebRTC v1 only supports single-view configs; "
                f"{cfg.pipeline_config_name!r} has num_views={transformer_cfg.num_views}."
            )

        self._device = torch.device(cfg.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Omnidreams WebRTC runtime.")

        logger.info("Loading Omnidreams first frame from {}", first_frame_path)
        image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read first frame from {first_frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (cfg.video_width, cfg.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        self._initial_rgb_frames = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.uint8)
        )

        prompt = prompt_path.read_text(encoding="utf-8").strip() or AV_POSITIVE_PROMPT
        self._text_prompts = [TextPrompt(positive=prompt)]

        loadable_clipgt_dir = self._prepare_clipgt_dir(clipgt_dir)
        logger.info("Loading Omnidreams scene data from {}", loadable_clipgt_dir)
        scene_t0 = time.perf_counter()
        scene_data = load_scene(
            loadable_clipgt_dir,
            camera_names=[cfg.camera_name],
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=(cfg.video_height, cfg.video_width),
        )
        logger.info(
            "Loaded Omnidreams scene data in {:.1f}s; attaching Ludus scene.",
            time.perf_counter() - scene_t0,
        )
        ludus_t0 = time.perf_counter()
        scene_data = load_and_attach_ludus_scene(
            loadable_clipgt_dir,
            scene_data,
            device=self._device,
        )
        logger.info(
            "Attached Omnidreams Ludus scene in {:.1f}s.",
            time.perf_counter() - ludus_t0,
        )
        if not scene_data.ego_poses:
            raise ValueError(f"Scene {loadable_clipgt_dir} has no ego poses.")
        if cfg.camera_name not in scene_data.camera_models:
            raise ValueError(
                f"Camera {cfg.camera_name!r} was not loaded from {loadable_clipgt_dir}."
            )
        if cfg.camera_name not in scene_data.camera_extrinsics:
            raise ValueError(
                f"Camera {cfg.camera_name!r} has no extrinsics in {loadable_clipgt_dir}."
            )

        logger.info(
            "Setting up Omnidreams pipeline {} on {}. This may load checkpoints, "
            "compile modules, and initialize CUDA graphs.",
            cfg.pipeline_config_name,
            self._device,
        )
        pipeline_t0 = time.perf_counter()
        self._wrapper = OmnidreamsConditioningWrapper(
            pipeline_config_name=cfg.pipeline_config_name,
            resolution_wh=(cfg.video_width, cfg.video_height),
            seed_for_every_rollout=cfg.seed,
            device=self._device,
        )
        logger.info(
            "Omnidreams pipeline setup complete in {:.1f}s.",
            time.perf_counter() - pipeline_t0,
        )
        self._scene_data = scene_data
        logger.info("Creating Omnidreams renderer for camera {}", cfg.camera_name)
        renderer_t0 = time.perf_counter()
        self._renderer = self._wrapper.create_renderer(scene_data, [cfg.camera_name])
        logger.info(
            "Omnidreams renderer ready in {:.1f}s.",
            time.perf_counter() - renderer_t0,
        )
        self._camera_to_rig = torch.as_tensor(
            scene_data.camera_extrinsics[cfg.camera_name],
            device=self._device,
            dtype=torch.float32,
        )
        self._initial_ego_pose = scene_data.ego_poses[0].transformation_matrix
        self._next_timestamp_us = int(scene_data.ego_poses[0].timestamp)
        self._reset_rollout_sync()
        logger.info(
            "Omnidreams runtime initialization complete in {:.1f}s.",
            time.perf_counter() - init_t0,
        )

    def _prepare_clipgt_dir(self, clipgt_dir: Path) -> Path:
        def _has_prefixed_parquets(path: Path) -> bool:
            return any(path.glob("*.calibration_estimate.parquet"))

        def _has_unprefixed_parquets(path: Path) -> bool:
            return (path / "calibration_estimate.parquet").exists()

        if _has_prefixed_parquets(clipgt_dir):
            return clipgt_dir

        parquet_source_dir: Path | None = None
        if _has_unprefixed_parquets(clipgt_dir):
            parquet_source_dir = clipgt_dir
        else:
            # Some HF scenes extract into ``clipgt/clipgt`` (or another single
            # nested directory) while first_image/prompt stay one level up.
            # Discover that nested parquet root and normalize it for loader use.
            nested_candidates = [
                child for child in clipgt_dir.iterdir() if child.is_dir()
            ]
            for candidate in nested_candidates:
                if _has_prefixed_parquets(candidate):
                    return candidate
                if _has_unprefixed_parquets(candidate):
                    parquet_source_dir = candidate
                    break

        if parquet_source_dir is None:
            return clipgt_dir

        self._clipgt_temp_dir = tempfile.TemporaryDirectory(prefix="omnidreams-clipgt-")
        staged = Path(self._clipgt_temp_dir.name)
        for source in parquet_source_dir.glob("*.parquet"):
            target = staged / f"clip.{source.name}"
            os.symlink(source.resolve(), target)
        return staged

    def _reset_rollout_sync(self) -> None:
        if self._wrapper is None or self._renderer is None:
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._initial_ego_pose is None or self._scene_data is None:
            raise OmnidreamsRuntimeError("Scene state is not initialized.")

        if self._state is not None and self._state.pipeline_cache is not None:
            del self._state.pipeline_cache
        self._state = None
        self.pose_integrator = CameraPoseIntegrator(
            move_speed_per_s=self.config.move_speed_per_s,
            rotate_speed_rad_per_s=self.config.rotate_speed_rad_per_s,
            coordinate_system="FLU",
        )
        self.pose_integrator.reset(self._initial_ego_pose)
        self.autoregressive_index = 0
        self._next_timestamp_us = int(self._scene_data.ego_poses[0].timestamp)
        self._wrapper.set_rollout_seed(self.config.seed)

    def _close_sync(self) -> None:
        state = self._state
        wrapper = self._wrapper
        self._state = None
        self._wrapper = None
        self._renderer = None
        self._scene_data = None
        self._initial_rgb_frames = None
        self._text_prompts = None
        self._camera_to_rig = None
        self._initial_ego_pose = None

        if state is not None and wrapper is not None:
            wrapper.cleanup(state)
        if wrapper is not None:
            del wrapper
        if self._clipgt_temp_dir is not None:
            self._clipgt_temp_dir.cleanup()
            self._clipgt_temp_dir = None

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        if (
            self._wrapper is None
            or self._renderer is None
            or self._initial_rgb_frames is None
            or self._text_prompts is None
            or self._camera_to_rig is None
        ):
            raise OmnidreamsRuntimeError("Runtime is not initialized.")
        if self._device is None:
            raise OmnidreamsRuntimeError("Runtime device is not initialized.")

        num_frames = self.peek_next_chunk_num_frames()
        if len(frame_times) != num_frames:
            raise OmnidreamsRuntimeError(
                f"Expected {num_frames} frame_times for chunk={self.autoregressive_index}, "
                f"got {len(frame_times)}."
            )
        if not segments:
            raise OmnidreamsRuntimeError(
                f"Chunk={self.autoregressive_index} received empty segments."
            )

        ego_poses = self.pose_integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        ego_poses_t = torch.from_numpy(ego_poses).to(
            device=self._device, dtype=torch.float32
        )
        camera_poses = torch.einsum("nij,jk->nik", ego_poses_t, self._camera_to_rig)
        frame_timestamps_us = self._consume_timestamps(num_frames)

        camera_names = [self.config.camera_name]
        camera_poses_per_view = {self.config.camera_name: camera_poses}
        serve_hdmaps = self.config.debug_serve_hdmaps
        if self._state is None:
            output = self._wrapper.start_generation(
                text_prompts=self._text_prompts,
                initial_rgb_frames=self._initial_rgb_frames,
                renderer=self._renderer,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state
        else:
            output = self._wrapper.continue_generation(
                state=self._state,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
            self._state = output.state

        if self._state.pipeline_cache is not None:
            self._wrapper.finalize_block_generation(
                self._state.pipeline_cache,
                output.finalization_state,
            )

        if serve_hdmaps:
            video_chunk = output.condition_frames
        elif output.rgb_frames is None:
            raise OmnidreamsRuntimeError("Omnidreams WebRTC received no RGB frames.")
        else:
            video_chunk = output.rgb_frames

        result = WebRTCStepResult(
            chunk_index=self.autoregressive_index,
            num_frames=int(video_chunk.shape[2]),
            video_chunk=video_chunk.detach().cpu(),
            stats=None,
        )
        self.autoregressive_index += 1
        return result

    def _consume_timestamps(self, num_frames: int) -> list[int]:
        step_us = int(round(1_000_000 / self.config.fps))
        timestamps = [self._next_timestamp_us + i * step_us for i in range(num_frames)]
        self._next_timestamp_us += num_frames * step_us
        return timestamps


_ManagedOmnidreamsSession = ManagedWebRTCSession


class OmnidreamsWebRTCSessionManager(BaseWebRTCSessionManager):
    """Owns one active WebRTC session and forwards WSAD actions."""

    _busy_message = "An Omnidreams session is already active."
    _warmup_label = "Omnidreams WebRTC"
    _runtime_error_types = (OmnidreamsRuntimeError,)
    # A chunk-generation failure here is fatal to the rollout, so tear the
    # session down instead of retrying on the next tick.
    _close_session_on_generation_error = True
    _resampler_supported_keys = WSAD_SUPPORTED_KEYS

    def __init__(
        self,
        *,
        runtime_config: OmnidreamsRuntimeConfig | None = None,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        runtime_config = runtime_config or OmnidreamsRuntimeConfig()
        super().__init__(
            runtime=OmnidreamsInferenceRuntime(config=runtime_config),
            runtime_config=runtime_config,
            fps=runtime_config.fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )

    def _model_name(self) -> str:
        return self.runtime_config.pipeline_config_name

    def _chunk_done_extra(self) -> dict[str, Any]:
        return {"stream": "hdmap" if self.runtime_config.debug_serve_hdmaps else "rgb"}

    def _register_extra_peer_handlers(self, peer_connection: Any) -> None:
        @peer_connection.on("iceconnectionstatechange")
        def on_iceconnectionstatechange() -> None:
            logger.info(
                "Peer ICE connection state changed: {}",
                peer_connection.iceConnectionState,
            )

        @peer_connection.on("icegatheringstatechange")
        def on_icegatheringstatechange() -> None:
            logger.debug(
                "Peer ICE gathering state changed: {}",
                peer_connection.iceGatheringState,
            )

    def _on_offer_received(self, offer_sdp: str) -> None:
        logger.info(
            "Received WebRTC offer with {}.", _summarize_sdp_candidates(offer_sdp)
        )

    def _on_answer_created(self, answer_sdp: str) -> None:
        logger.info(
            "Created WebRTC answer with {}.", _summarize_sdp_candidates(answer_sdp)
        )
