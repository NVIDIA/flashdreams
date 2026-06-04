# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-clip quality regression test for Omnidreams rollouts.

The test is intentionally fixture-gated: normal GPU CI skips it unless the
reference clip and input assets are provided through environment variables.
When wired in CI, it generates a short deterministic rollout and compares the
generated half of the output MP4 against the reference clip.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from omnidreams.config import OMNIDREAMS_RUNNERS

from flashdreams.infra.config import derive_config
from flashdreams.quality.clip_compare import (
    ClipComparisonThresholds,
    assert_clip_within_thresholds,
    bottom_half,
    format_clip_comparison,
    parse_frame_indices,
    read_video_rgb,
)

pytestmark = pytest.mark.ci_gpu

_ENV_PREFIX = "FLASHDREAMS_OMNIDREAMS_QUALITY_"
_DEFAULT_RUNNER = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
_DEFAULT_TOTAL_BLOCKS = 4  # ~1 second for chunk2 + 30fps output.


def test_omnidreams_generated_clip_matches_reference(tmp_path: Path) -> None:
    """Run a short Omnidreams rollout and compare sampled generated frames."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Omnidreams quality regression requires CUDA.")

    reference_clip = _env_existing_path("REFERENCE_CLIP", required=True)
    assert reference_clip is not None
    cfg = _quality_runner_config(tmp_path)

    cfg.setup().run()

    candidate_clip = tmp_path / f"{cfg.runner_name}.mp4"
    assert candidate_clip.exists(), f"runner did not write {candidate_clip}"

    reference = read_video_rgb(reference_clip)
    candidate = read_video_rgb(candidate_clip)
    if _env_bool("EXTRACT_GENERATED_REGION_FROM_REFERENCE", default=False):
        reference = bottom_half(reference)
    if _env_bool("EXTRACT_GENERATED_REGION_FROM_CANDIDATE", default=True):
        candidate = bottom_half(candidate)
    _write_artifacts(
        reference_clip=reference_clip,
        candidate_clip=candidate_clip,
        reference_compare_region=reference,
        candidate_compare_region=candidate,
    )

    thresholds = ClipComparisonThresholds(
        max_mean_abs=_env_float("MAX_MEAN_ABS", 4.0),
        max_rmse=_env_float("MAX_RMSE", 8.0),
        min_psnr_db=_env_float("MIN_PSNR_DB", 30.0),
        max_frame_mean_abs=_env_float("MAX_FRAME_MEAN_ABS", 8.0),
        max_frame_rmse=_env_float("MAX_FRAME_RMSE", 14.0),
        max_mean_flip=_env_optional_float("MAX_MEAN_FLIP", 0.070),
        max_frame_flip=_env_optional_float("MAX_FRAME_FLIP", 0.080),
        require_same_frame_count=_env_bool("REQUIRE_SAME_FRAME_COUNT", default=True),
    )
    result = assert_clip_within_thresholds(
        reference,
        candidate,
        thresholds=thresholds,
        frame_indices=parse_frame_indices(os.environ.get(_env("FRAME_INDICES"))),
        sample_count=_env_int("SAMPLE_COUNT", 8),
    )
    print(format_clip_comparison(result))


def _quality_runner_config(tmp_path: Path) -> Any:
    runner_name = os.environ.get(_env("RUNNER"), _DEFAULT_RUNNER)
    if runner_name not in OMNIDREAMS_RUNNERS:
        pytest.fail(
            f"{_env('RUNNER')}={runner_name!r} is not an Omnidreams runner. "
            f"Known runners: {sorted(OMNIDREAMS_RUNNERS)}"
        )

    example_data = _env_bool("EXAMPLE_DATA", default=False)
    hdmap_video_paths = _env_existing_paths(
        "HDMAP_VIDEO_PATHS", required=not example_data
    )
    first_frame_paths = _env_existing_paths("FIRST_FRAME_PATHS", required=False)
    embeddings_path = _env_existing_path("EMBEDDINGS_PATH", required=False)
    if embeddings_path is None and not first_frame_paths and not example_data:
        pytest.skip(
            f"Set {_env('FIRST_FRAME_PATHS')} or {_env('EMBEDDINGS_PATH')} "
            f"or {_env('EXAMPLE_DATA')}=1 to run the Omnidreams quality "
            "regression."
        )

    changes: dict[str, Any] = {
        "output_dir": tmp_path,
        "total_blocks": _env_int("TOTAL_BLOCKS", _DEFAULT_TOTAL_BLOCKS),
        "pixel_height": _env_int("PIXEL_HEIGHT", 704),
        "pixel_width": _env_int("PIXEL_WIDTH", 1280),
        "example_data": example_data,
    }
    if hdmap_video_paths:
        changes["hdmap_video_paths"] = hdmap_video_paths
    if first_frame_paths:
        changes["first_frame_paths"] = first_frame_paths
    if embeddings_path is not None:
        changes["embeddings_path"] = embeddings_path
        changes["pipeline"] = {"text_encoder": None, "image_encoder": None}
    example_data_uuid = os.environ.get(_env("EXAMPLE_DATA_UUID"))
    if example_data_uuid:
        changes["example_data_uuid"] = example_data_uuid

    prompt = os.environ.get(_env("PROMPT"))
    if prompt:
        changes["prompt"] = prompt

    camera_names = _env_csv("CAMERA_NAMES")
    if camera_names:
        changes["camera_names"] = camera_names

    return derive_config(OMNIDREAMS_RUNNERS[runner_name], **changes)


def _env(name: str) -> str:
    return f"{_ENV_PREFIX}{name}"


def _env_existing_path(name: str, *, required: bool) -> Path | None:
    value = os.environ.get(_env(name))
    if not value:
        if required:
            pytest.skip(f"Set {_env(name)} to run the Omnidreams quality regression.")
        return None
    path = Path(value).expanduser()
    if not path.exists():
        pytest.skip(f"{_env(name)} does not exist: {path}")
    return path


def _env_existing_paths(name: str, *, required: bool) -> tuple[Path, ...]:
    value = os.environ.get(_env(name))
    if not value:
        if required:
            pytest.skip(f"Set {_env(name)} to run the Omnidreams quality regression.")
        return ()
    paths = tuple(
        Path(part.strip()).expanduser() for part in value.split(",") if part.strip()
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"{_env(name)} path(s) do not exist: {missing}")
    return paths


def _env_csv(name: str) -> tuple[str, ...]:
    value = os.environ.get(_env(name))
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(_env(name), str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(_env(name), str(default)))


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(_env(name))
    if value is None:
        return default
    if value.lower() in {"", "none", "off", "false"}:
        return None
    return float(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(_env(name))
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _write_artifacts(
    *,
    reference_clip: Path,
    candidate_clip: Path,
    reference_compare_region: Any,
    candidate_compare_region: Any,
) -> None:
    artifact_dir_value = os.environ.get(_env("ARTIFACT_DIR"))
    if not artifact_dir_value:
        return

    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - read_video_rgb already imports it
        raise ImportError(
            "Writing quality-regression artifacts requires mediapy."
        ) from exc

    artifact_dir = Path(artifact_dir_value).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_clip, artifact_dir / "reference_original.mp4")
    shutil.copy2(candidate_clip, artifact_dir / "candidate_original.mp4")
    media.write_video(
        str(artifact_dir / "reference_compare_region.mp4"),
        reference_compare_region,
        fps=_env_int("OUTPUT_FPS", 30),
    )
    media.write_video(
        str(artifact_dir / "candidate_compare_region.mp4"),
        candidate_compare_region,
        fps=_env_int("OUTPUT_FPS", 30),
    )
