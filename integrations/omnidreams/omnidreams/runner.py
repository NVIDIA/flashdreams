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

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from loguru import logger
from omnidreams.hf import omni_dreams_hf_repo, omni_dreams_hf_url
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

_BATCH_PROMPT_KEYS = ("prompt", "text")
_BATCH_PROMPTS_KEYS = ("prompts",)
_BATCH_PROMPT_PATH_KEYS = ("prompt_path", "text_path")
_BATCH_PROMPT_PATHS_KEYS = ("prompt_paths", "text_paths")
_BATCH_HDMAP_KEYS = (
    "hdmap_video_paths",
    "hdmap_paths",
    "hdmap_video_path",
    "hdmap_path",
)
_BATCH_FIRST_FRAME_KEYS = (
    "first_frame_paths",
    "first_frame_path",
    "image_paths",
    "image_path",
)
_BATCH_CAMERA_KEYS = ("camera_names", "camera_name")

_REPO_ROOT = Path(__file__).resolve().parents[4]

EXAMPLE_DATA_HF_REPO = omni_dreams_hf_repo("omni-dreams-samples")
"""Single-view HDMap clips + first frames under the configured HF org."""

DEFAULT_EXAMPLE_DATA_UUID_1V = "23599139-948f-4681-b7f4-74794113086d"
"""Arbitrary first-alphabetically pick from the 32 single-view clips
the dataset ships. Override with ``--example-data-uuid <uuid>``; see
the configured Omni Dreams HF dataset's ``data/single_view`` directory."""

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
            f"{omni_dreams_hf_url('omni-dreams-samples', 'tree/main/data/single_view', repo_type='dataset')} "
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
class _BatchItem:
    """One manifest row normalized enough for the runner loop."""

    index: int
    source: dict[str, Any]
    clip_id: str
    dataset: str | None = None
    prompt_id: str | None = None
    prompt_source: str | None = None
    seed: int | None = None
    prompt: str | None = None
    prompts: tuple[str, ...] | None = None
    hdmap_video_paths: tuple[Path, ...] = ()
    first_frame_paths: tuple[Path, ...] = ()
    camera_names: tuple[str, ...] | None = None
    output_dir: Path | None = None
    output_video_path: Path | None = None
    stats_path: Path | None = None
    meta_path: Path | None = None
    video_filename: str | None = None
    stats_filename: str | None = None
    embeddings_path: Path | None = None
    total_blocks: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for batch metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Convert Paths/tuples/nested records into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _record_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value for any of ``keys``."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _coerce_str_tuple(value: Any, *, split_commas: bool) -> tuple[str, ...]:
    """Parse JSON arrays, Python lists, or delimiter-separated strings."""
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            parsed = json.loads(text)
            return _coerce_str_tuple(parsed, split_commas=split_commas)
        if "|" in text:
            return tuple(part.strip() for part in text.split("|") if part.strip())
        if split_commas and "," in text:
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return (text,)
    return (str(value),)


def _coerce_path_tuple(value: Any) -> tuple[Path, ...]:
    return tuple(Path(v) for v in _coerce_str_tuple(value, split_commas=True))


def _coerce_optional_path(value: Any) -> Path | None:
    values = _coerce_path_tuple(value)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"expected one path, got {values}")
    return values[0]


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError("metadata JSON must decode to an object")
            return parsed
        return {"metadata": text}
    raise TypeError(f"metadata must be an object or JSON object string, got {value!r}")


def _load_batch_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, JSON-array/object, or CSV batch records."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    text = path.read_text(encoding="utf-8")
    if suffix in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        stripped = text.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("items", "rollouts", "records"):
                    if key in data:
                        data = data[key]
                        break
            records = data
        else:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

    if not isinstance(records, list):
        raise TypeError(
            f"batch input {path} must contain a list of records, got {type(records)}"
        )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"batch input record {index} must be an object, got {type(record)}"
            )
        normalized.append({str(k).strip(): v for k, v in record.items()})
    return normalized


def _parse_batch_item(record: dict[str, Any], *, index: int) -> _BatchItem:
    """Normalize one manifest record into an internal item."""
    clip_id = _coerce_optional_str(
        _record_value(record, ("clip_id", "uuid", "scene_id", "id"))
    )
    if clip_id is None:
        clip_id = f"item_{index:06d}"

    prompts: tuple[str, ...] | None = None
    prompt: str | None = None
    prompt_paths = _coerce_path_tuple(
        _record_value(record, _BATCH_PROMPT_PATHS_KEYS)
    )
    prompt_path = _coerce_optional_path(_record_value(record, _BATCH_PROMPT_PATH_KEYS))
    if prompt_paths:
        prompt_texts = tuple(
            p.read_text(encoding="utf-8").strip() for p in prompt_paths
        )
        if len(prompt_texts) == 1:
            prompt = prompt_texts[0]
        else:
            prompts = prompt_texts
    elif prompt_path is not None:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt_values = _coerce_str_tuple(
            _record_value(record, _BATCH_PROMPTS_KEYS),
            split_commas=False,
        )
        if prompt_values:
            prompts = prompt_values
        else:
            prompt = _coerce_optional_str(_record_value(record, _BATCH_PROMPT_KEYS))

    output_video_path = _coerce_optional_path(
        _record_value(record, ("output_video_path", "output_video", "video_path"))
    )
    stats_path = _coerce_optional_path(
        _record_value(record, ("stats_path", "stats_json", "output_stats_path"))
    )
    meta_path = _coerce_optional_path(
        _record_value(record, ("meta_path", "meta_json", "metadata_path"))
    )

    return _BatchItem(
        index=index,
        source=dict(record),
        clip_id=clip_id,
        dataset=_coerce_optional_str(_record_value(record, ("dataset", "dataset_id"))),
        prompt_id=_coerce_optional_str(_record_value(record, ("prompt_id",))),
        prompt_source=_coerce_optional_str(_record_value(record, ("prompt_source",))),
        seed=_coerce_optional_int(_record_value(record, ("seed",)), name="seed"),
        prompt=prompt,
        prompts=prompts,
        hdmap_video_paths=_coerce_path_tuple(
            _record_value(record, _BATCH_HDMAP_KEYS)
        ),
        first_frame_paths=_coerce_path_tuple(
            _record_value(record, _BATCH_FIRST_FRAME_KEYS)
        ),
        camera_names=(
            _coerce_str_tuple(
                _record_value(record, _BATCH_CAMERA_KEYS),
                split_commas=True,
            )
            or None
        ),
        output_dir=_coerce_optional_path(_record_value(record, ("output_dir",))),
        output_video_path=output_video_path,
        stats_path=stats_path,
        meta_path=meta_path,
        video_filename=_coerce_optional_str(
            _record_value(record, ("video_filename", "video_name"))
        ),
        stats_filename=_coerce_optional_str(
            _record_value(record, ("stats_filename", "stats_name"))
        ),
        embeddings_path=_coerce_optional_path(
            _record_value(record, ("embeddings_path", "embedding_path"))
        ),
        total_blocks=_coerce_optional_int(
            _record_value(record, ("total_blocks",)), name="total_blocks"
        ),
        metadata=_coerce_metadata(_record_value(record, ("metadata", "meta"))),
    )


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

    pad_final_hdmap_chunk: bool = False
    """When True, pad the final partial HDMap chunk by repeating the last
    conditioning frame, run one final AR step, and crop the saved video back
    to the original HDMap frame count. This is useful for batch clips whose
    frame counts are not exactly representable by the model's AR chunk size."""

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

    batch_inputs_path: Path | None = None
    """Optional JSONL/JSON/CSV manifest of rollout inputs. Batch mode keeps
    the instantiated pipeline alive across every record, precomputes raw
    prompt/first-frame embeddings before releasing one-shot encoders, then
    runs each rollout without reloading the model process."""

    batch_results_path: Path | None = None
    """Optional CSV path for batch results. Defaults to
    ``output_dir / "manifest.csv"`` in batch mode."""

    batch_skip_existing: bool = True
    """Skip manifest records whose resolved output video already exists
    and is non-empty."""

    batch_continue_on_error: bool = True
    """Continue batch execution after a failed record. Set to ``False`` for
    fail-fast sweeps."""

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
        if cfg.batch_inputs_path is not None:
            assert cfg.save_embeddings_path is None, (
                "--batch_inputs_path and --save_embeddings_path are mutually "
                "exclusive: batch mode owns per-record embedding handling."
            )
            self._run_batch(cfg.batch_inputs_path)
            return
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

    ## Batch mode

    def _run_batch(self, batch_inputs_path: Path) -> None:
        """Run a manifest of rollouts without rebuilding the pipeline."""
        cfg = self.config
        records = _load_batch_records(batch_inputs_path)
        items = [_parse_batch_item(record, index=i) for i, record in enumerate(records)]
        if self.is_rank_zero:
            logger.info(
                f"[{cfg.runner_name}] loaded {len(items)} batch input(s) "
                f"from {batch_inputs_path}"
            )

        runnable: list[_BatchItem] = []
        for item in items:
            video_path = self._batch_output_video_path(item)
            if (
                cfg.batch_skip_existing
                and video_path.exists()
                and video_path.stat().st_size > 0
            ):
                if self.is_rank_zero:
                    logger.info(
                        f"[{cfg.runner_name}] batch item {item.index} "
                        f"clip={item.clip_id!r} skipped; output exists: {video_path}"
                    )
                self._write_batch_metadata_and_result(
                    item=item,
                    status="skipped",
                    started_at=None,
                    finished_at=_utc_now_iso(),
                    exit_code=0,
                    error=None,
                )
                continue
            runnable.append(item)

        if not runnable:
            if self.is_rank_zero:
                logger.info(f"[{cfg.runner_name}] no batch items to run")
            return

        precomputed_embeddings: dict[int, dict[str, torch.Tensor | None]] = {}
        ready: list[_BatchItem] = []
        num_views = self._num_views()
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        for item in runnable:
            if self._batch_embeddings_path(item) is not None:
                ready.append(item)
                continue

            started_at = _utc_now_iso()
            try:
                prompts = self._resolve_item_prompts(item, num_views)
                first_frame_paths = self._resolve_item_paths(
                    item.first_frame_paths,
                    cfg.first_frame_paths,
                    num_views,
                    name="first_frame_paths",
                )
                if self.is_rank_zero:
                    logger.info(
                        f"[{cfg.runner_name}] precomputing embeddings for "
                        f"batch item {item.index}/{len(runnable)} "
                        f"clip={item.clip_id!r}"
                    )
                first_frames_t = self._load_first_frames(
                    first_frame_paths, device=device, dtype=dtype
                )
                precomputed_embeddings[item.index] = (
                    self.pipeline.precompute_embeddings(
                        text=[list(prompts)],
                        image=first_frames_t,
                    )
                )
                ready.append(item)
                del first_frames_t
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if self.is_rank_zero:
                    logger.exception(
                        f"[{cfg.runner_name}] failed precomputing batch item "
                        f"{item.index} clip={item.clip_id!r}"
                    )
                self._write_batch_metadata_and_result(
                    item=item,
                    status="failed",
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    exit_code=1,
                    error=error,
                )
                if not cfg.batch_continue_on_error:
                    raise

        # All raw prompt/image inputs have been encoded; keep the heavy
        # diffusion/decoder stack alive and release only the one-shot encoders.
        self.pipeline.release_oneshot_encoders()

        for item in ready:
            started_at = _utc_now_iso()
            try:
                self._reset_rollout_seed(item)
                item_num_views = self._num_views()
                camera_names = self._resolve_item_camera_names(item, item_num_views)
                embeddings_path = self._batch_embeddings_path(item)
                if embeddings_path is not None:
                    assert embeddings_path.exists(), (
                        f"batch item {item.index} embeddings_path does not "
                        f"exist: {embeddings_path}"
                    )
                    embeddings = torch.load(
                        embeddings_path, map_location="cpu", weights_only=True
                    )
                else:
                    embeddings = precomputed_embeddings[item.index]

                cache = self.pipeline.initialize_cache_from_embeddings(
                    text_embeddings=embeddings["text_embeddings"],
                    image_embeddings=embeddings["image_embeddings"],
                    negative_text_embeddings=embeddings.get(
                        "negative_text_embeddings"
                    ),
                    view_names=list(camera_names),
                )
                self._rollout_and_save(
                    cache=cache,
                    num_views=item_num_views,
                    hdmap_paths=self._resolve_item_paths(
                        item.hdmap_video_paths,
                        cfg.hdmap_video_paths,
                        item_num_views,
                        name="hdmap_video_paths",
                    ),
                    total_blocks=self._resolve_item_total_blocks(item),
                    output_video_path=self._batch_output_video_path(item),
                    stats_path=self._batch_stats_path(item),
                )
                self._write_batch_metadata_and_result(
                    item=item,
                    status="completed",
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    exit_code=0,
                    error=None,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if self.is_rank_zero:
                    logger.exception(
                        f"[{cfg.runner_name}] failed batch item {item.index} "
                        f"clip={item.clip_id!r}"
                    )
                self._write_batch_metadata_and_result(
                    item=item,
                    status="failed",
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    exit_code=1,
                    error=error,
                )
                if not cfg.batch_continue_on_error:
                    raise

    def _batch_embeddings_path(self, item: _BatchItem) -> Path | None:
        """Per-record embeddings path, falling back to the CLI-level path."""
        return item.embeddings_path or self.config.embeddings_path

    def _batch_seed_for_metadata(self, item: _BatchItem) -> int | None:
        if item.seed is not None:
            return item.seed
        return self.config.pipeline.diffusion_model.seed

    def _batch_output_dir(self, item: _BatchItem) -> Path:
        """Resolve the output directory for one batch item."""
        if item.output_dir is not None:
            return item.output_dir
        if item.output_video_path is not None:
            return item.output_video_path.parent

        cfg = self.config
        output_dir = cfg.output_dir
        if item.dataset:
            output_dir = output_dir / item.dataset
        output_dir = output_dir / item.clip_id / cfg.runner_name
        if item.prompt_id:
            output_dir = output_dir / item.prompt_id
            seed = self._batch_seed_for_metadata(item)
            return output_dir / (str(seed) if seed is not None else "seed_default")
        seed = self._batch_seed_for_metadata(item)
        if seed is not None:
            return output_dir / f"seed_{seed}"
        return output_dir / f"item_{item.index:06d}"

    def _batch_output_video_path(self, item: _BatchItem) -> Path:
        if item.output_video_path is not None:
            return item.output_video_path
        return self._batch_output_dir(item) / (item.video_filename or "video.mp4")

    def _batch_stats_path(self, item: _BatchItem) -> Path:
        if item.stats_path is not None:
            return item.stats_path
        return self._batch_output_dir(item) / (item.stats_filename or "stats.json")

    def _batch_meta_path(self, item: _BatchItem) -> Path:
        if item.meta_path is not None:
            return item.meta_path
        return self._batch_output_dir(item) / "meta.json"

    def _resolve_item_prompts(
        self, item: _BatchItem, num_views: int
    ) -> tuple[str, ...]:
        if item.prompts is not None:
            assert len(item.prompts) == num_views, (
                f"batch item {item.index} prompts has {len(item.prompts)} "
                f"entries but pipeline expects {num_views}."
            )
            return item.prompts
        if item.prompt is not None:
            assert item.prompt, f"batch item {item.index} prompt is empty"
            return (item.prompt,) * num_views
        return self._resolve_prompts(num_views)

    def _resolve_item_camera_names(
        self, item: _BatchItem, num_views: int
    ) -> tuple[str, ...]:
        if item.camera_names is not None:
            assert len(item.camera_names) == num_views, (
                f"batch item {item.index} camera_names has "
                f"{len(item.camera_names)} entries but pipeline expects {num_views}."
            )
            return item.camera_names
        return self._resolve_camera_names(num_views)

    def _resolve_item_paths(
        self,
        item_paths: tuple[Path, ...],
        default_paths: tuple[Path, ...],
        num_views: int,
        *,
        name: str,
    ) -> tuple[Path, ...]:
        paths = item_paths or default_paths
        return self._resolve_paths(paths, num_views, name=name)

    def _resolve_item_total_blocks(self, item: _BatchItem) -> int:
        if item.total_blocks is not None:
            return item.total_blocks
        return self.config.total_blocks

    def _reset_rollout_seed(self, item: _BatchItem) -> None:
        """Reset the diffusion RNG so each batch row matches one CLI process."""
        seed = item.seed
        if seed is not None and self.config.offset_seed_by_global_rank:
            seed += self.global_rank
        if seed is None:
            seed = self.config.pipeline.diffusion_model.seed
        self.pipeline.diffusion_model.config.seed = seed
        self.pipeline.diffusion_model._rng = None

    def _write_batch_metadata_and_result(
        self,
        *,
        item: _BatchItem,
        status: str,
        started_at: str | None,
        finished_at: str,
        exit_code: int,
        error: str | None,
    ) -> None:
        """Persist per-item ``meta.json`` and append the batch result CSV."""
        if not self.is_rank_zero:
            return

        video_path = self._batch_output_video_path(item)
        stats_path = self._batch_stats_path(item)
        meta_path = self._batch_meta_path(item)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        seed = self._batch_seed_for_metadata(item)
        metadata = {
            **item.metadata,
            "status": status,
            "error": error,
            "clip_id": item.clip_id,
            "dataset": item.dataset,
            "model": self.config.runner_name,
            "prompt_id": item.prompt_id,
            "prompt_source": item.prompt_source,
            "seed": seed,
            "effective_seed": self.pipeline.diffusion_model.config.seed,
            "camera_names": item.camera_names or self.config.camera_names,
            "hdmap_video_paths": item.hdmap_video_paths
            or self.config.hdmap_video_paths,
            "first_frame_paths": item.first_frame_paths
            or self.config.first_frame_paths,
            "output_video": video_path,
            "stats_json": stats_path,
            "meta_json": meta_path,
            "total_blocks": self._resolve_item_total_blocks(item),
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "source_record": item.source,
        }
        meta_path.write_text(
            json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        results_path = self.config.batch_results_path or (
            self.config.output_dir / "manifest.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "status",
            "clip_id",
            "dataset",
            "model",
            "prompt_id",
            "seed",
            "output_video",
            "stats_json",
            "meta_json",
            "started_at",
            "finished_at",
            "exit_code",
            "error",
        )
        write_header = not results_path.exists() or results_path.stat().st_size == 0
        with results_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "status": status,
                    "clip_id": item.clip_id,
                    "dataset": item.dataset or "",
                    "model": self.config.runner_name,
                    "prompt_id": item.prompt_id or "",
                    "seed": "" if seed is None else seed,
                    "output_video": str(video_path),
                    "stats_json": str(stats_path),
                    "meta_json": str(meta_path),
                    "started_at": started_at or "",
                    "finished_at": finished_at,
                    "exit_code": exit_code,
                    "error": error or "",
                }
            )

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
        self,
        *,
        cache: OmnidreamsPipelineCache,
        num_views: int,
        hdmap_paths: tuple[Path, ...] | None = None,
        total_blocks: int | None = None,
        output_video_path: Path | None = None,
        stats_path: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Run the AR loop against ``cache`` and write video + stats."""
        cfg = self.config
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        resolved_hdmap_paths = self._resolve_paths(
            hdmap_paths if hdmap_paths is not None else cfg.hdmap_video_paths,
            num_views,
            name="hdmap_video_paths",
        )
        hdmap_videos: list[torch.Tensor] = [
            _load_video(
                resolved_hdmap_paths[i],
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
        resolved_total_blocks = total_blocks or cfg.total_blocks
        target_num_frames = hdmap_num_frames
        for i in range(resolved_total_blocks):
            num_frames = self.pipeline.get_num_frames(i)
            end = start + num_frames
            is_padded_final_chunk = False
            if end > hdmap_num_frames:
                if not cfg.pad_final_hdmap_chunk or start >= hdmap_num_frames:
                    break
                pad_frames = end - hdmap_num_frames
                hdmap_chunk = hdmap_videos_t[:, :, start:hdmap_num_frames]
                tail = hdmap_videos_t[:, :, hdmap_num_frames - 1 : hdmap_num_frames]
                pad = tail.expand(*tail.shape[:2], pad_frames, *tail.shape[3:])
                hdmap_chunk = torch.cat([hdmap_chunk, pad], dim=2)
                is_padded_final_chunk = True
            else:
                hdmap_chunk = hdmap_videos_t[:, :, start:end]
            if self.is_rank_zero:
                msg = (
                    f"[{cfg.runner_name}] AR step {i}/{resolved_total_blocks}, "
                    f"num_frames={num_frames}, frames=[{start}, {end})"
                )
                if is_padded_final_chunk:
                    msg += (
                        f" padded_final_chunk target_frames={target_num_frames}"
                    )
                logger.info(msg)
            video_chunk = self.pipeline.generate(
                autoregressive_index=i,
                cache=cache,
                hdmap=hdmap_chunk,
            )
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())
            start = end
            if is_padded_final_chunk:
                break

        if not chunks:
            raise RuntimeError(
                f"HDMap videos have {hdmap_num_frames} frame(s), which is too "
                "short for the first rollout chunk."
            )
        video = torch.cat(chunks, dim=2)  # [B, V, T, C, H, W]
        if cfg.pad_final_hdmap_chunk:
            video = video[:, :, :target_num_frames]
        generated_num_frames = video.shape[2]
        if not self.is_rank_zero:
            return None, None

        # HDMap + generated stacked vertically per camera, cameras laid
        # out horizontally: ``[T, 2*H, V*W, C]``.
        if output_video_path is None:
            output_video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        if stats_path is None:
            stats_path = output_video_path.parent / f"stats_{cfg.runner_name}.json"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        condition = hdmap_videos_t[:, :, :generated_num_frames].cpu()
        canvas = rearrange(
            torch.cat([condition, video], dim=-2),
            "1 v t c h w -> t h (v w) c",
        )
        _write_video(canvas, output_video_path, fps=cfg.output_fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(video.shape)} "
            f"-> {output_video_path.resolve()}"
        )

        if stats_history:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )
        return output_video_path, stats_path

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
