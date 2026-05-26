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

"""Omnidreams HDMap-conditioned I2V runner classes (single- + multi-view).

Pure implementation module. The per-slug ``*_RUNNER`` literals + the
``OMNIDREAMS_RUNNERS`` aggregating dict live in
:mod:`omnidreams.config`, alongside the matching
pipeline configs.

:meth:`OmnidreamsRunner.run` dispatches across three modes:

- Default: encode then AR rollout, write MP4 + per-step stats.
- ``--save_embeddings_path``: run only the one-shot encoders,
  ``torch.save`` the embeddings, exit before the AR loop.
- ``--embeddings_path``: hydrate the cache from precomputed
  embeddings and skip the one-shot encoder forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from loguru import logger
from omnidreams.pipeline import (
    OmnidreamsPipeline,
    OmnidreamsPipelineCache,
)
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.core.io.internal import use_internal_storage
from flashdreams.core.io.s3_sync import sync_s3_dir_to_local
from flashdreams.infra.runner import Runner, RunnerConfig

DEFAULT_VIDEO_HEIGHT = 704
"""Pixel-space rollout height (matches the trained 720p chassis)."""

DEFAULT_VIDEO_WIDTH = 1280
"""Pixel-space rollout width (matches the trained 720p chassis)."""

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

_REPO_ROOT = Path(__file__).resolve().parents[4]

EXAMPLE_DATA_HF_REPO = "nvidia/omni-dreams-samples"
"""Single-view HDMap clips + first frames in the NVIDIA HF dataset."""

EXAMPLE_DATA_HF_BROWSER_URL = (
    "https://huggingface.co/datasets/nvidia/omni-dreams-samples/tree/main/"
    "data/single_view"
)
"""Browser URL for the public single-view sample list."""

DEFAULT_EXAMPLE_DATA_UUID_1V = "239560dc-33d1-11ef-9720-00044bcbccac"
"""Arbitrary first-alphabetically pick from the 32 single-view clips
the dataset ships. Override with ``--example-data-uuid <uuid>``; see
the NVIDIA Omni Dreams HF dataset's ``data/single_view`` directory."""

EXAMPLE_DATA_DIR_S3 = "s3://flashdreams/assets/example_data/omnidreams"
"""Internal-team source for both views; also the external fallback for
multi-view (no HF mirror yet)."""

EXAMPLE_DATA_DIR_LOCAL = _REPO_ROOT / "assets/example_data/omnidreams"
"""Local cache the S3 sync writes into."""

S3_CREDENTIAL_PATH = _REPO_ROOT / "credentials/s3_checkpoint.secret"
"""Required for any S3 sync (internal mode, or external multi-view)."""

_CAMERA_NAMES_1V = ("camera_front_wide_120fov",)
_CAMERA_NAMES_4V = (
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
)


def _example_camera_names(num_views: int) -> tuple[str, ...]:
    """Return the canonical bundled camera-name tuple for ``num_views``."""
    if num_views == 1:
        return _CAMERA_NAMES_1V
    if num_views == 4:
        return _CAMERA_NAMES_4V
    raise ValueError(
        f"example data only ships single-view (1) and 4-camera multi-view (4); "
        f"got num_views={num_views}."
    )


def _ensure_hf_single_view_example_data_synced(
    uuid: str,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Pull ``data/single_view/<uuid>/{*_hdmap.mp4, first_frame.png}``
    from :data:`EXAMPLE_DATA_HF_REPO` (the hdmap filename is per-clip so
    we list the dir first to find it). Returns ``((hdmap,), (first_frame,))``."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    subdir = f"data/single_view/{uuid}"
    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=EXAMPLE_DATA_HF_REPO,
        repo_type="dataset",
        path_in_repo=subdir,
        recursive=False,
    )
    files = [entry.path for entry in entries if isinstance(entry, RepoFile)]
    hdmap_candidates = [f for f in files if f.endswith("_hdmap.mp4")]
    if not hdmap_candidates:
        raise FileNotFoundError(
            f"No '*_hdmap.mp4' under {subdir!r} in HF dataset "
            f"{EXAMPLE_DATA_HF_REPO!r}. Pick a UUID listed at "
            f"{EXAMPLE_DATA_HF_BROWSER_URL} "
            "via --example-data-uuid <uuid>, or supply --hdmap-video-paths / "
            "--first-frame-paths explicitly."
        )
    if len(hdmap_candidates) > 1:
        raise RuntimeError(
            f"Multiple '*_hdmap.mp4' files under {subdir!r} in "
            f"{EXAMPLE_DATA_HF_REPO!r}: {hdmap_candidates}. Expected exactly "
            "one; aborting to avoid an ambiguous demo selection."
        )
    hdmap_local = Path(
        hf_hub_download(
            repo_id=EXAMPLE_DATA_HF_REPO,
            repo_type="dataset",
            filename=hdmap_candidates[0],
        )
    )
    first_frame_local = Path(
        hf_hub_download(
            repo_id=EXAMPLE_DATA_HF_REPO,
            repo_type="dataset",
            filename=f"{subdir}/first_frame.png",
        )
    )
    return (hdmap_local,), (first_frame_local,)


def _ensure_s3_example_data_synced(
    num_views: int, *, is_rank_zero: bool
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Mirror :data:`EXAMPLE_DATA_DIR_S3` to local on rank 0 and return
    per-camera ``(hdmap_paths, first_frame_paths)``. Requires
    :data:`S3_CREDENTIAL_PATH`."""
    if is_rank_zero:
        assert S3_CREDENTIAL_PATH.exists(), (
            f"S3 credential file not found at {S3_CREDENTIAL_PATH}. "
            "Either populate it (see README) or unset --example-data and "
            "pass --hdmap-video-paths / --first-frame-paths explicitly."
        )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=str(S3_CREDENTIAL_PATH),
        cache_dir=str(EXAMPLE_DATA_DIR_LOCAL),
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing omnidreams example data from S3",
    )
    names = _example_camera_names(num_views)
    hdmap = tuple(EXAMPLE_DATA_DIR_LOCAL / f"{n}.mp4" for n in names)
    first = tuple(EXAMPLE_DATA_DIR_LOCAL / f"{n}.png" for n in names)
    return hdmap, first


@dataclass(kw_only=True)
class OmnidreamsRunnerConfig(RunnerConfig):
    """Runner config covering every shipped Omnidreams variant.

    Single-view and 4-camera multi-view share this shape; the wrapped
    pipeline's ``CosmosTransformerConfig.num_views`` decides the
    layout. Per-camera asset tuples are in the canonical camera order.
    """

    _target: type = field(default_factory=lambda: OmnidreamsRunner)

    prompt: str = ""
    """Default text prompt applied to every camera. Override per-camera
    via :attr:`prompts` when the cameras need different prompts."""

    prompts: tuple[str, ...] = ()
    """Optional per-camera prompts. When non-empty must have one entry
    per camera (matches ``num_views`` on the wrapped pipeline) and
    overrides :attr:`prompt`."""

    hdmap_video_paths: tuple[Path, ...] = ()
    """Per-camera HDMap video paths in the canonical camera order.
    Required at ``run()`` time."""

    first_frame_paths: tuple[Path, ...] = ()
    """Per-camera first-frame image (or video) paths in the canonical
    camera order. When a video is provided, frame 0 is used."""

    camera_names: tuple[str, ...] = ()
    """Optional per-camera labels. When non-empty must have one entry
    per camera (used for cross-view bookkeeping); defaults to indexed
    placeholders when omitted."""

    total_blocks: int = 60
    """Number of AR chunks to attempt. The loop stops early once the
    HDMap video is consumed."""

    pixel_height: int = DEFAULT_VIDEO_HEIGHT
    """Resize target height for HDMap videos and first-frame images."""

    pixel_width: int = DEFAULT_VIDEO_WIDTH
    """Resize target width for HDMap videos and first-frame images."""

    output_fps: int = 30
    """Output video frame rate. Omnidreams was trained at 30fps."""

    save_embeddings_path: Path | None = None
    """When set, run only the one-shot encoders, ``torch.save`` text +
    image embeddings to this path, and exit before the AR loop. The
    precompute is rank-0 only (saved tensors are not CP-split)."""

    embeddings_path: Path | None = None
    """When set, hydrate the per-rollout cache from this file and skip
    the one-shot encoder forward pass; the encoders are released right
    after ``__init__``. Mutually exclusive with
    ``--save_embeddings_path``."""

    example_data: bool = False
    """Lazy-fetch a bundled HDMap clip + first frame and fill the empty
    path tuples from the canonical per-view defaults. Use for the README
    demo; pass explicit paths instead for production runs."""

    example_data_uuid: str = DEFAULT_EXAMPLE_DATA_UUID_1V
    """Single-view example clip to pull from :data:`EXAMPLE_DATA_HF_REPO`.
    Ignored for multi-view or when paths are already populated."""


class OmnidreamsRunner(Runner[OmnidreamsRunnerConfig, OmnidreamsPipeline]):
    """Streaming HDMap-conditioned I2V driver."""

    config: OmnidreamsRunnerConfig

    def run(self) -> None:
        """Drive the Omnidreams AR rollout to completion."""
        cfg = self.config
        assert not (cfg.save_embeddings_path and cfg.embeddings_path), (
            "--save_embeddings_path and --embeddings_path are mutually "
            "exclusive: the first writes embeddings, the second reads them."
        )
        if cfg.example_data:
            self._fill_example_data_defaults()
        if cfg.save_embeddings_path is not None:
            self._run_save_embeddings(cfg.save_embeddings_path)
            return
        if cfg.embeddings_path is not None:
            self._run_with_embeddings(cfg.embeddings_path)
            return
        self._run_default()

    def _fill_example_data_defaults(self) -> None:
        """Lazy-fetch bundled assets and fill empty path tuples in-place.
        External 1V uses HF; everything else (internal mode, external 4V)
        uses S3."""
        cfg = self.config
        num_views = self._num_views()
        if not use_internal_storage() and num_views == 1:
            hdmap, first = _ensure_hf_single_view_example_data_synced(
                cfg.example_data_uuid
            )
        else:
            hdmap, first = _ensure_s3_example_data_synced(
                num_views, is_rank_zero=self.is_rank_zero
            )
        if not cfg.hdmap_video_paths:
            cfg.hdmap_video_paths = hdmap
        if not cfg.first_frame_paths:
            cfg.first_frame_paths = first
        if not cfg.camera_names:
            cfg.camera_names = _example_camera_names(num_views)

    ## Run modes

    def _run_default(self) -> None:
        """Encode prompts + first frames, build the cache, run the AR loop."""
        cfg = self.config
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        num_views = self._num_views()
        prompts = self._resolve_prompts(num_views)
        camera_names = self._resolve_camera_names(num_views)
        first_frame_paths = self._resolve_paths(
            cfg.first_frame_paths, num_views, name="first_frame_paths"
        )

        first_frames_t = self._load_first_frames(
            first_frame_paths, device=device, dtype=dtype
        )
        cache = self.pipeline.initialize_cache(
            text=[list(prompts)],
            image=first_frames_t,
            view_names=list(camera_names),
        )
        # Drop the one-shot encoders to free VRAM before the AR loop;
        # long-lived servers that reuse encoders across sessions skip
        # this and call ``release_oneshot_encoders`` on shutdown.
        self.pipeline.release_oneshot_encoders()
        self._rollout_and_save(cache=cache, num_views=num_views)

    def _run_save_embeddings(self, output_path: Path) -> None:
        """Run only the one-shot encoders and ``torch.save`` the embeddings."""
        cfg = self.config
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        num_views = self._num_views()
        prompts = self._resolve_prompts(num_views)
        first_frame_paths = self._resolve_paths(
            cfg.first_frame_paths, num_views, name="first_frame_paths"
        )

        if self.global_rank != 0:
            # Saved tensors are not CP-split; non-zero ranks idle
            # until rank 0 finishes and hits the barrier below.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        first_frames_t = self._load_first_frames(
            first_frame_paths, device=device, dtype=dtype
        )
        embeddings = self.pipeline.precompute_embeddings(
            text=[list(prompts)],
            image=first_frames_t,
        )
        # ``negative_text_embeddings`` is opt-in (``Tensor | None``);
        # text + image are always present.
        text_emb = embeddings["text_embeddings"]
        image_emb = embeddings["image_embeddings"]
        assert text_emb is not None and image_emb is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, output_path)
        logger.info(
            f"[{cfg.runner_name}] saved precomputed embeddings "
            f"text={tuple(text_emb.shape)} "
            f"image={tuple(image_emb.shape)} "
            f"-> {output_path.resolve()}"
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _run_with_embeddings(self, embeddings_path: Path) -> None:
        """Hydrate the cache from precomputed embeddings, run the AR loop."""
        cfg = self.config
        num_views = self._num_views()
        camera_names = self._resolve_camera_names(num_views)

        # Free encoder VRAM before any GPU-heavy work; the loaded
        # embeddings hydrate the cache without an encoder forward pass.
        self.pipeline.release_oneshot_encoders()

        assert embeddings_path.exists(), (
            f"--embeddings_path does not exist: {embeddings_path}"
        )
        embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        if self.is_rank_zero:
            logger.info(
                f"[{cfg.runner_name}] loaded embeddings from {embeddings_path} "
                f"text={tuple(embeddings['text_embeddings'].shape)} "
                f"image={tuple(embeddings['image_embeddings'].shape)}"
            )
        cache = self.pipeline.initialize_cache_from_embeddings(
            text_embeddings=embeddings["text_embeddings"],
            image_embeddings=embeddings["image_embeddings"],
            negative_text_embeddings=embeddings.get("negative_text_embeddings"),
            view_names=list(camera_names),
        )
        self._rollout_and_save(cache=cache, num_views=num_views)

    ## Shared rollout / I/O body

    def _rollout_and_save(
        self, *, cache: OmnidreamsPipelineCache, num_views: int
    ) -> None:
        """Run the AR loop against ``cache`` and write video + stats."""
        cfg = self.config
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        hdmap_paths = self._resolve_paths(
            cfg.hdmap_video_paths, num_views, name="hdmap_video_paths"
        )
        hdmap_videos: list[torch.Tensor] = [
            _load_video(
                hdmap_paths[i],
                pixel_height=cfg.pixel_height,
                pixel_width=cfg.pixel_width,
                device=device,
                dtype=dtype,
            )
            for i in range(num_views)
        ]
        hdmap_videos_t = torch.stack(hdmap_videos, dim=0).unsqueeze(0)
        # Shape: [B=1, V, T, C, H, W]
        hdmap_num_frames = hdmap_videos_t.shape[2]
        if self.is_rank_zero:
            logger.info(
                f"[{cfg.runner_name}] loaded hdmap_videos="
                f"{tuple(hdmap_videos_t.shape)}, num_views={num_views}"
            )

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        start = 0
        for i in range(cfg.total_blocks):
            num_frames = self.pipeline.get_num_frames(i)
            end = start + num_frames
            if end > hdmap_num_frames:
                break
            if self.is_rank_zero:
                logger.info(
                    f"[{cfg.runner_name}] AR step {i}/{cfg.total_blocks}, "
                    f"num_frames={num_frames}, frames=[{start}, {end})"
                )
            video_chunk = self.pipeline.generate(
                autoregressive_index=i,
                cache=cache,
                hdmap=hdmap_videos_t[:, :, start:end],
            )
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())
            start = end

        video = torch.cat(chunks, dim=2)  # [B, V, T, C, H, W]
        generated_num_frames = video.shape[2]
        if not self.is_rank_zero:
            return

        # HDMap + generated stacked vertically per camera, cameras laid
        # out horizontally: ``[T, 2*H, V*W, C]``.
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        condition = hdmap_videos_t[:, :, :generated_num_frames].cpu()
        canvas = rearrange(
            torch.cat([condition, video], dim=-2),
            "1 v t c h w -> t h (v w) c",
        )
        video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        _write_video(canvas, video_path, fps=cfg.output_fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(video.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )

    ## Helpers

    def _num_views(self) -> int:
        """Recover the global ``num_views`` (per-rank ``num_views`` x ``V_size``).

        ``OmnidreamsPipeline.__init__`` divides ``transformer.config.num_views``
        by the CP ``V_size`` for the per-rank shard, so reading the field
        directly after ``setup()`` returns ``1`` on a 4-GPU run with 4 cameras.
        Multiply by ``self.pipeline.V_size`` to get the unsplit count.
        """
        transformer_cfg = self.config.pipeline.diffusion_model.transformer
        assert isinstance(transformer_cfg, CosmosTransformerConfig)
        return transformer_cfg.num_views * self.pipeline.V_size

    def _load_first_frames(
        self,
        first_frame_paths: tuple[Path, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load the per-camera first-frame seeds as ``[B=1, V, 1, C, H, W]``."""
        cfg = self.config
        first_frames = [
            _load_first_frame(
                p,
                pixel_height=cfg.pixel_height,
                pixel_width=cfg.pixel_width,
                device=device,
                dtype=dtype,
            )
            for p in first_frame_paths
        ]
        return torch.stack(first_frames, dim=0).unsqueeze(0)

    def _resolve_prompts(self, num_views: int) -> tuple[str, ...]:
        cfg = self.config
        if cfg.prompts:
            assert len(cfg.prompts) == num_views, (
                f"--prompts has {len(cfg.prompts)} entries but pipeline "
                f"expects {num_views}; pass one prompt per camera or use "
                "--prompt for a shared default."
            )
            return cfg.prompts
        assert cfg.prompt, (
            "either --prompt or --prompts must be set "
            "(both empty resolved to no text input)."
        )
        return (cfg.prompt,) * num_views

    def _resolve_camera_names(self, num_views: int) -> tuple[str, ...]:
        cfg = self.config
        if cfg.camera_names:
            assert len(cfg.camera_names) == num_views, (
                f"--camera_names has {len(cfg.camera_names)} entries but "
                f"pipeline expects {num_views}."
            )
            return cfg.camera_names
        return tuple(f"view_{i}" for i in range(num_views))

    @staticmethod
    def _resolve_paths(
        paths: tuple[Path, ...], num_views: int, *, name: str
    ) -> tuple[Path, ...]:
        assert paths, (
            f"--{name} is required: pass {num_views} comma-separated "
            "path(s) in the canonical camera order."
        )
        assert len(paths) == num_views, (
            f"--{name} has {len(paths)} entries but pipeline expects "
            f"{num_views}; pass one path per camera."
        )
        return paths


__all__ = [
    "OmnidreamsRunner",
    "OmnidreamsRunnerConfig",
    "DEFAULT_VIDEO_HEIGHT",
    "DEFAULT_VIDEO_WIDTH",
]


## I/O helpers (``cv2`` / ``mediapy`` lazy-imported; live under the ``runners`` extras).


def _read_first_frame_np(path: Path) -> np.ndarray:
    """Read a first-frame image (or frame 0 of a video) as ``[H, W, 3]``."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Loading the first-frame asset needs mediapy. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    if path.suffix.lower() in IMAGE_SUFFIXES:
        return media.read_image(str(path))[..., :3]
    video = media.read_video(str(path))
    assert video.shape[0] > 0, f"video has no frames: {path}"
    return video[0, ..., :3]


def _load_first_frame(
    path: Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resize a first-frame asset and return ``[1, C, H, W]`` in ``[-1, 1]``."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Resizing the first-frame asset needs opencv. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = _read_first_frame_np(path)
    arr = cv2.resize(arr, (pixel_width, pixel_height))
    tensor = (
        torch.from_numpy(arr).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # [H, W, C]
    return rearrange(tensor, "h w c -> 1 c h w")


def _load_video(
    path: Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load + resize an HDMap video to ``[T, C, H, W]`` in ``[-1, 1]``."""
    try:
        import cv2  # noqa: PLC0415
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Loading HDMap videos needs mediapy + opencv. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    video_np = media.read_video(str(path))[..., :3]
    if video_np.shape[1:3] != (pixel_height, pixel_width):
        video_np = np.stack(
            [cv2.resize(f, (pixel_width, pixel_height)) for f in video_np], axis=0
        )
    tensor = (
        torch.from_numpy(video_np).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # [T, H, W, C]
    return rearrange(tensor, "t h w c -> t c h w")


def _write_video(canvas: torch.Tensor, path: Path, *, fps: int) -> None:
    """Save a ``[T, H, W, C]`` ``[-1, 1]`` tensor as an MP4."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Writing the output video needs mediapy. Install the runner "
            "extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = (canvas.float().numpy() + 1.0) / 2.0
    arr = (arr * 255).clip(0, 255).astype("uint8")
    media.write_video(str(path), arr, fps=fps)
