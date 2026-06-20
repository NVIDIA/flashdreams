# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WorldLens checkout, staging, and command adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omnidreams.eval.manifest import StagedCase

DEFAULT_WORLDLENS_REPO = "https://github.com/worldbench/WorldLens.git"
DEFAULT_WORLDLENS_REVISION = "fa6c5ff177e99d0d5b126f238ca42a430982b19a"
DEFAULT_WORLDLENS_METHOD = "omnidreams"
DEFAULT_WORLDLENS_CONFIG_NAME = "default_run_omnidreams_videogen_consistency"


@dataclass(frozen=True)
class WorldLensCheckout:
    path: Path
    repo_url: str
    revision: str
    resolved_commit: str
    config_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "repo_url": self.repo_url,
            "revision": self.revision,
            "resolved_commit": self.resolved_commit,
            "config_name": self.config_name,
        }


def ensure_worldlens_checkout(
    *,
    cache_dir: Path,
    repo_url: str = DEFAULT_WORLDLENS_REPO,
    revision: str = DEFAULT_WORLDLENS_REVISION,
    fetch: bool = True,
    install_config: bool = True,
    config_name: str = DEFAULT_WORLDLENS_CONFIG_NAME,
) -> WorldLensCheckout:
    """Clone/fetch WorldLens and checkout ``revision``."""

    target = cache_dir / "WorldLens"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        _run(["git", "clone", repo_url, str(target)], cwd=cache_dir)
    elif fetch:
        _run(["git", "fetch", "--tags", "origin"], cwd=target)
    _run(["git", "checkout", revision], cwd=target)
    commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=target).strip()
    if install_config:
        write_worldlens_consistency_config(target, config_name=config_name)
    return WorldLensCheckout(
        path=target,
        repo_url=repo_url,
        revision=revision,
        resolved_commit=commit,
        config_name=config_name if install_config else None,
    )


def write_worldlens_consistency_config(
    worldlens_root: Path,
    *,
    config_name: str = DEFAULT_WORLDLENS_CONFIG_NAME,
) -> Path:
    """Write a lightweight WorldLens videogen config for generated-vs-GT clips."""

    config_path = worldlens_root / "tools" / "configs" / f"{config_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """hydra:
  run:
    dir: ${output_dir}
  output_subdir: ${output_dir}/code/hydra

defaults:
  - _self_
  - experiment

method_name: ???
generated_data_path: generated_results

videogen:
  dimensions:
    generation:
      temporal_consistency:
        - name: temporal_consistency
          method_name: ${method_name}
          generated_data_path: ${generated_data_path}
          need_preprocessing: true
          repeat_times: 1
          local_save_path: pretrained_models/clip/ViT-B-32.pt
      subject_consistency:
        - name: subject_consistency
          method_name: ${method_name}
          generated_data_path: ${generated_data_path}
          need_preprocessing: true
          repeat_times: 1
          local_save_path: pretrained_models/dino/dino_vitbase16_pretrain.pth
          repo_or_dir: worldbench/third_party/dino
""",
        encoding="utf-8",
    )
    return config_path


def stage_worldlens_video_inputs(
    staged_cases: Sequence[StagedCase],
    *,
    generated_root: Path,
    worldlens_root: Path,
    method_name: str = DEFAULT_WORLDLENS_METHOD,
    generation_index: int = 0,
    camera_name: str = "CAM_FRONT",
    force: bool = False,
) -> Path:
    """Stage generated clips and frame-matched references in WorldLens' layout."""

    if generation_index < 0:
        raise ValueError(
            f"generation_index must be non-negative, got {generation_index}"
        )
    method_slug = _validate_slug(method_name, "method_name")
    camera_slug = _validate_camera_name(camera_name)

    stage_root = worldlens_root / "generated_results"
    generated_submission = stage_root / method_slug / "video_submission"
    reference_submission = stage_root / "gt" / "video_submission"
    generated_submission.mkdir(parents=True, exist_ok=True)
    reference_submission.mkdir(parents=True, exist_ok=True)

    manifest_path = stage_root / method_slug / "stage_manifest.json"
    manifest_rows_by_scene = _read_existing_stage_manifest_cases(manifest_path)
    for staged in staged_cases:
        uuid = _validate_slug(staged.case.uuid, "uuid")
        scene_dir_name = f"{uuid}_gen{generation_index}"
        generated_video = generated_root / staged.case.uuid / "generated.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"missing generated video: {generated_video}")
        if not staged.reference_video_path.exists():
            raise FileNotFoundError(
                f"missing reference video: {staged.reference_video_path}"
            )

        generated_scene_dir = generated_submission / scene_dir_name
        reference_scene_dir = reference_submission / scene_dir_name
        generated_scene_dir.mkdir(parents=True, exist_ok=True)
        reference_scene_dir.mkdir(parents=True, exist_ok=True)

        video_filename = f"{uuid}_{camera_slug}.mp4"
        generated_target = generated_scene_dir / video_filename
        reference_target = reference_scene_dir / video_filename
        generated_frame_count = video_frame_count(generated_video)
        _copy_or_link(generated_video, generated_target, force=force)
        reference_frame_count = copy_video_first_frames(
            staged.reference_video_path,
            reference_target,
            max_frames=generated_frame_count,
            force=force,
        )
        manifest_rows_by_scene[scene_dir_name] = {
            "uuid": staged.case.uuid,
            "generation_index": generation_index,
            "camera_name": camera_slug,
            "scene_dir": scene_dir_name,
            "video_filename": video_filename,
            "temporal_policy": "reference_first_n_frames_matching_generated",
            "generated_video": str(generated_video),
            "reference_video": str(staged.reference_video_path),
            "worldlens_generated_video": str(generated_target),
            "worldlens_reference_video": str(reference_target),
            "generated_frame_count": generated_frame_count,
            "reference_frame_count": reference_frame_count,
        }

    manifest_path.write_text(
        json.dumps(
            {
                "kind": "worldlens_stage_manifest",
                "method_name": method_slug,
                "generation_index": generation_index,
                "camera_name": camera_slug,
                "generated_submission": str(generated_submission),
                "reference_submission": str(reference_submission),
                "cases": [
                    manifest_rows_by_scene[key]
                    for key in sorted(manifest_rows_by_scene)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def video_frame_count(video_path: Path) -> int:
    """Return the number of frames in ``video_path`` using OpenCV."""

    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "video frame counting requires opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            return frame_count
        count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
        if count == 0:
            raise RuntimeError(f"failed to count any frames in video: {video_path}")
        return count
    finally:
        cap.release()


def copy_video_first_frames(
    source: Path,
    target: Path,
    *,
    max_frames: int,
    force: bool = False,
) -> int:
    """Write the first ``max_frames`` from ``source`` to ``target`` as MP4."""

    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    if target.exists() or target.is_symlink():
        if not force:
            existing_count = video_frame_count(target)
            if existing_count != max_frames:
                raise RuntimeError(
                    f"existing WorldLens reference video {target} has "
                    f"{existing_count} frames, expected {max_frames}; rerun with --force"
                )
            return existing_count
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("video cropping requires opencv-python-headless") from exc

    cap = cv2.VideoCapture(str(source))
    writer = None
    frames_written = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open reference video: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        while frames_written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")
                fourcc = video_writer_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {target}")
            writer.write(frame)
            frames_written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames_written < max_frames:
        raise RuntimeError(
            f"wrote only {frames_written} frames from {source}, expected {max_frames}"
        )
    return frames_written


def worldlens_evaluate_command(
    *,
    worldlens_root: Path,
    modality: str = "videogen",
    method_name: str = DEFAULT_WORLDLENS_METHOD,
    config_name: str = DEFAULT_WORLDLENS_CONFIG_NAME,
    generated_data_path: Path | str = "generated_results",
    python: str = "python",
    hydra_overrides: Sequence[str] = (),
) -> list[str]:
    """Return the WorldLens Hydra command."""

    _validate_slug(method_name, "method_name")
    command = [
        python,
        "tools/evaluate.py",
        "--config-name",
        config_name,
        f"modality={modality}",
        f"method_name={method_name}",
        f"generated_data_path={generated_data_path}",
    ]
    command.extend(hydra_overrides)
    return command


def run_worldlens_evaluation(
    *,
    worldlens_root: Path,
    modality: str = "videogen",
    method_name: str = DEFAULT_WORLDLENS_METHOD,
    config_name: str = DEFAULT_WORLDLENS_CONFIG_NAME,
    generated_data_path: Path | str = "generated_results",
    python: str = "python",
    exp_root: Path | None = None,
    log_path: Path | None = None,
    output_json: Path | None = None,
    hydra_overrides: Sequence[str] = (),
) -> dict[str, object]:
    """Run WorldLens and capture a small summary JSON."""

    if exp_root is None:
        exp_root = worldlens_root / "tools" / "exp"
    command = worldlens_evaluate_command(
        worldlens_root=worldlens_root,
        modality=modality,
        method_name=method_name,
        config_name=config_name,
        generated_data_path=generated_data_path,
        python=python,
        hydra_overrides=hydra_overrides,
    )
    env = os.environ.copy()
    env["WORLDBENCH_EXP_ROOT"] = str(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        subprocess.run(command, cwd=worldlens_root, check=True, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(command) + "\n\n")
            log_file.flush()
            subprocess.run(
                command,
                cwd=worldlens_root,
                check=True,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

    result_path = latest_worldlens_metric_results(
        exp_root=exp_root,
        modality=modality,
        method_name=method_name,
    )
    metric_results = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path is not None
        else {}
    )
    payload = {
        "modality": modality,
        "method_name": method_name,
        "config_name": config_name,
        "generated_data_path": str(generated_data_path),
        "exp_root": str(exp_root),
        "metric_results_path": str(result_path) if result_path is not None else None,
        "metric_results": metric_results,
        "artifact_results": collect_worldlens_artifact_results(
            worldlens_root=worldlens_root,
            method_name=method_name,
            generated_data_path=generated_data_path,
        ),
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def latest_worldlens_metric_results(
    *,
    exp_root: Path,
    modality: str,
    method_name: str,
) -> Path | None:
    """Return the newest WorldLens top-level ``metric_results.json``."""

    root = exp_root / modality / method_name
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("metric_results.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect_worldlens_artifact_results(
    *,
    worldlens_root: Path,
    method_name: str,
    generated_data_path: Path | str = "generated_results",
) -> dict[str, object]:
    """Collect per-metric JSON artifacts written under ``generated_results``."""

    generated_data_root = Path(generated_data_path)
    if not generated_data_root.is_absolute():
        generated_data_root = worldlens_root / generated_data_root
    method_root = generated_data_root / method_name
    if not method_root.exists():
        return {}

    results: dict[str, object] = {}
    for path in sorted(method_root.rglob("*.json")):
        if path.name == "stage_manifest.json":
            continue
        try:
            rel = path.relative_to(method_root).as_posix()
        except ValueError:
            rel = str(path)
        try:
            results[rel] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results[rel] = {"path": str(path), "error": "invalid_json"}
    return results


def _read_existing_stage_manifest_cases(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    cases = payload.get("cases", {})
    if not isinstance(cases, list):
        return {}
    rows: dict[str, dict[str, object]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        scene_dir = case.get("scene_dir")
        if isinstance(scene_dir, str):
            rows[scene_dir] = dict(case)
    return rows


def _copy_or_link(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() or target.is_symlink():
        if not force:
            return
        target.unlink()
    try:
        target.symlink_to(
            os.path.relpath(source.resolve(), start=target.parent.resolve())
        )
    except OSError:
        shutil.copy2(source, target)


def _validate_slug(value: str, label: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            f"{label} must contain only letters, numbers, dots, dashes, and underscores"
        )
    return value


def _validate_camera_name(value: str) -> str:
    if not value or not re.fullmatch(r"[A-Z0-9_]+", value):
        raise ValueError(
            f"camera_name must contain only uppercase letters, numbers, and underscores"
        )
    return value


def _run(args: list[str], *, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _run_capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True)
