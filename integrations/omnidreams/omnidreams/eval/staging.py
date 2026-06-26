# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare one evaluation batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from omnidreams.eval.hf_assets import download_asset
from omnidreams.eval.manifest import EvalCase, StagedCase, write_staged_cases_jsonl


def stage_cases(
    cases: Sequence[EvalCase],
    *,
    batch_id: str,
    scratch_root: Path,
    output_manifest: Path,
    token: str | None = None,
    force: bool = False,
) -> list[StagedCase]:
    """Download assets for ``cases`` and extract first-frame seed images."""

    downloads_root = scratch_root / "downloads"
    batch_root = scratch_root / "batches" / batch_id
    staged_cases: list[StagedCase] = []

    for case in cases:
        case_root = batch_root / case.uuid
        case_root.mkdir(parents=True, exist_ok=True)
        reference_video_path = download_asset(
            case.reference_video,
            repo_id=case.dataset_repo,
            revision=case.dataset_revision,
            destination_root=downloads_root,
            token=token,
            force=force,
        )
        hdmap_video_path = download_asset(
            case.hdmap_video,
            repo_id=case.dataset_repo,
            revision=case.dataset_revision,
            destination_root=downloads_root,
            token=token,
            force=force,
        )
        prompt_path = download_asset(
            case.prompt,
            repo_id=case.dataset_repo,
            revision=case.dataset_revision,
            destination_root=downloads_root,
            token=token,
            force=force,
        )
        first_frame_path = case_root / "first_frame.png"
        if force or not first_frame_path.exists():
            extract_first_frame(reference_video_path, first_frame_path)
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        staged = StagedCase(
            case=case,
            reference_video_path=reference_video_path,
            hdmap_video_path=hdmap_video_path,
            prompt_path=prompt_path,
            first_frame_path=first_frame_path,
            prompt_text=prompt_text,
        )
        staged_cases.append(staged)
        (case_root / "input.json").write_text(
            json.dumps(staged.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_staged_cases_jsonl(staged_cases, output_manifest)
    return staged_cases


def extract_first_frame(video_path: Path, output_path: Path) -> None:
    """Extract frame 0 from ``video_path`` into ``output_path``."""

    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "first-frame extraction requires opencv-python-headless"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read first frame from {video_path}")
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"failed to write first frame to {output_path}")
