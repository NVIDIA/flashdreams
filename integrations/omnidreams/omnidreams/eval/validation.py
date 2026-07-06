# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for generated OmniDreams evaluation artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_AR_STEP_RE = re.compile(
    r"AR step (?P<index>\d+)/(?P<total>\d+), "
    r"num_frames=(?P<num_frames>\d+), frames=\[(?P<start>\d+), (?P<end>\d+)\)"
)
_LOADED_HDMAP_RE = re.compile(r"loaded hdmap_videos=\((?P<shape>[^)]*)\)")
_WROTE_VIDEO_RE = re.compile(r"wrote video \((?P<shape>[^)]*)\) -> (?P<path>[^\r\n]+)")


@dataclass(frozen=True)
class ARStep:
    index: int
    total: int
    num_frames: int
    start: int
    end: int


@dataclass(frozen=True)
class GenerationValidation:
    uuid: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    total_blocks: int | None = None
    ar_steps: int = 0
    expected_frames_from_steps: int | None = None
    runner_written_frames: int | None = None
    hdmap_frames: int | None = None
    stats_steps: int | None = None
    generated_video_path: str | None = None
    generated_video_bytes: int | None = None
    stacked_video_path: str | None = None
    stacked_video_bytes: int | None = None
    log_path: str | None = None
    stats_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_generated_run(
    run_root: Path, *, uuid: str | None = None
) -> list[GenerationValidation]:
    """Validate generated artifacts under ``run_root/generated``."""

    generated_root = run_root / "generated"
    if not generated_root.exists():
        raise FileNotFoundError(
            f"generated output directory does not exist: {generated_root}"
        )
    if uuid is not None:
        case_dirs = [generated_root / uuid]
    else:
        case_dirs = sorted(path for path in generated_root.iterdir() if path.is_dir())
    return [validate_generated_case(path) for path in case_dirs]


def validate_generated_case(case_dir: Path) -> GenerationValidation:
    uuid = case_dir.name
    issues: list[str] = []
    generation_json = _read_generation_json(case_dir / "generation.json", issues)
    command = generation_json.get("command", []) if generation_json else []
    total_blocks = _command_int_arg(command, "--total-blocks")

    generated_video = case_dir / "generated.mp4"
    if not generated_video.exists():
        issues.append(f"missing generated video: {generated_video}")
        generated_size = None
    else:
        generated_size = generated_video.stat().st_size
        if generated_size <= 0:
            issues.append(f"generated video is empty: {generated_video}")

    log_path = case_dir / "flashdreams-run.log"
    if not log_path.exists():
        issues.append(f"missing runner log: {log_path}")
        log_text = ""
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

    ar_steps = _parse_ar_steps(log_text)
    expected_frames = ar_steps[-1].end if ar_steps else None
    written_shape = _parse_last_written_shape(log_text)
    runner_written_frames = (
        written_shape[2] if written_shape and len(written_shape) >= 3 else None
    )
    hdmap_shape = _parse_loaded_hdmap_shape(log_text)
    hdmap_frames = hdmap_shape[2] if hdmap_shape and len(hdmap_shape) >= 3 else None

    if total_blocks is None:
        issues.append("could not determine --total-blocks from generation metadata")
    elif len(ar_steps) != total_blocks:
        issues.append(f"expected {total_blocks} AR steps, found {len(ar_steps)}")

    if ar_steps:
        for expected_index, step in enumerate(ar_steps):
            if step.index != expected_index:
                issues.append(
                    f"AR step index mismatch: expected {expected_index}, found {step.index}"
                )
                break
        if total_blocks is not None and any(
            step.total != total_blocks for step in ar_steps
        ):
            issues.append("AR step log total does not match --total-blocks")

    if expected_frames is not None and runner_written_frames is not None:
        if runner_written_frames != expected_frames:
            issues.append(
                f"runner wrote {runner_written_frames} frames, expected {expected_frames} from AR steps"
            )

    runner_mp4s = sorted((case_dir / "runner").glob("*.mp4"))
    stacked_video = runner_mp4s[0] if runner_mp4s else None
    if stacked_video is not None:
        stacked_size = stacked_video.stat().st_size
        if stacked_size <= 0:
            issues.append(f"stacked runner video is empty: {stacked_video}")
    else:
        stacked_size = None

    stats_path = _find_stats_path(case_dir)
    stats_steps = (
        _read_stats_steps(stats_path, issues) if stats_path is not None else None
    )
    if stats_path is None:
        issues.append(f"missing runner stats JSON under: {case_dir / 'runner'}")
    elif total_blocks is not None and stats_steps != total_blocks:
        issues.append(f"expected {total_blocks} stats rows, found {stats_steps}")

    return GenerationValidation(
        uuid=uuid,
        ok=not issues,
        issues=issues,
        total_blocks=total_blocks,
        ar_steps=len(ar_steps),
        expected_frames_from_steps=expected_frames,
        runner_written_frames=runner_written_frames,
        hdmap_frames=hdmap_frames,
        stats_steps=stats_steps,
        generated_video_path=str(generated_video),
        generated_video_bytes=generated_size,
        stacked_video_path=str(stacked_video) if stacked_video else None,
        stacked_video_bytes=stacked_size,
        log_path=str(log_path),
        stats_path=str(stats_path) if stats_path else None,
    )


def write_validation_json(results: list[GenerationValidation], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _read_generation_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing generation metadata: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"invalid generation metadata {path}: {exc}")
        return {}


def _command_int_arg(command: object, name: str) -> int | None:
    if not isinstance(command, list):
        return None
    args: list[str] = []
    for item in command:
        if not isinstance(item, str):
            return None
        args.append(item)
    try:
        index = args.index(name)
        return int(args[index + 1])
    except (ValueError, IndexError, TypeError):
        return None


def _parse_ar_steps(log_text: str) -> list[ARStep]:
    steps: list[ARStep] = []
    for match in _AR_STEP_RE.finditer(log_text):
        steps.append(
            ARStep(
                index=int(match.group("index")),
                total=int(match.group("total")),
                num_frames=int(match.group("num_frames")),
                start=int(match.group("start")),
                end=int(match.group("end")),
            )
        )
    return steps


def _parse_loaded_hdmap_shape(log_text: str) -> tuple[int, ...] | None:
    match = _LOADED_HDMAP_RE.search(log_text)
    if not match:
        return None
    return _parse_int_tuple(match.group("shape"))


def _parse_last_written_shape(log_text: str) -> tuple[int, ...] | None:
    matches = list(_WROTE_VIDEO_RE.finditer(log_text))
    if not matches:
        return None
    return _parse_int_tuple(matches[-1].group("shape"))


def _parse_int_tuple(value: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError:
        return None


def _find_stats_path(case_dir: Path) -> Path | None:
    paths = sorted((case_dir / "runner").glob("stats_*.json"))
    return paths[0] if paths else None


def _read_stats_steps(path: Path, issues: list[str]) -> int | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"invalid stats JSON {path}: {exc}")
        return None
    if not isinstance(value, list):
        issues.append(f"stats JSON is not a list: {path}")
        return None
    return len(value)
