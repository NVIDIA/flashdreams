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

"""Public PAI-Bench quality profile adapter for local benchmark reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

ProfileName = Literal["pai-bench-g", "pai-bench-long"]
PaiBenchRunner = Literal["local", "upstream"]

DEFAULT_PAI_BENCH_REPO = "https://github.com/SHI-Labs/physical-ai-bench.git"
DEFAULT_PAI_BENCH_REVISION = "2f3b687410029b98397fbc51fa4de36bfd45627d"
_SCHEMA_VERSION = 1
_PREFLIGHT_MARKER = "FLASHDREAMS_PAI_BENCH_PREFLIGHT="
_PREFLIGHT_SCRIPT = r"""
import importlib
import json
import sys
import traceback

failures = []
for dimension in sys.argv[1:]:
    try:
        importlib.import_module(f"pbench.{dimension}")
    except Exception as exc:
        failures.append(
            {
                "dimension": dimension,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "missing_import": getattr(exc, "name", None),
                "traceback": traceback.format_exc(),
            }
        )

payload = {
    "dimensions": list(sys.argv[1:]),
    "failures": failures,
    "status": "fail" if failures else "pass",
}
print("FLASHDREAMS_PAI_BENCH_PREFLIGHT=" + json.dumps(payload, sort_keys=True))
raise SystemExit(1 if failures else 0)
"""
_IMPORT_PACKAGE_HINTS = {
    "clip": "openai-clip",
    "cv2": "opencv-python-headless",
    "omegaconf": "omegaconf",
    "PIL": "Pillow",
    "pyiqa": "pyiqa",
}
_LOG_SCORE_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z0-9_]+)"
    r"(?:\s+\([^)]*\))?:\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class PaiBenchCheckout:
    path: Path
    repo_url: str
    revision: str
    resolved_commit: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": str(self.path),
            "repo_url": self.repo_url,
            "revision": self.revision,
            "resolved_commit": self.resolved_commit,
        }


@dataclass(frozen=True)
class StagedInputs:
    videos_dir: Path
    prompt_file: Path
    video_ids: tuple[str, ...]
    source_video: Path
    segment_duration_s: int | None

    def to_manifest(self) -> dict[str, object]:
        return {
            "videos_dir": self.videos_dir.as_posix(),
            "prompt_file": self.prompt_file.as_posix(),
            "video_ids": list(self.video_ids),
            "source_video": self.source_video.as_posix(),
            "segment_duration_s": self.segment_duration_s,
        }


@dataclass(frozen=True)
class PaiBenchInvocation:
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    runner: PaiBenchRunner

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "argv": list(self.command),
            "cwd": self.cwd.as_posix(),
            "runner": self.runner,
        }
        if self.env:
            payload["env"] = dict(self.env)
        return payload


@dataclass(frozen=True)
class PaiBenchPreflightResult:
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    log_path: Path
    payload: Mapping[str, Any] | None
    returncode: int

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and _preflight_status(self.payload) == "pass"

    def to_manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "argv": list(self.command),
            "cwd": self.cwd.as_posix(),
            "log_path": self.log_path.as_posix(),
            "returncode": self.returncode,
        }
        if self.env:
            manifest["env"] = dict(self.env)
        if self.payload is not None:
            manifest["payload"] = dict(self.payload)
        return manifest


def ensure_pai_bench_checkout(
    *,
    root: Path,
    repo_url: str = DEFAULT_PAI_BENCH_REPO,
    revision: str = DEFAULT_PAI_BENCH_REVISION,
    fetch: bool = True,
) -> PaiBenchCheckout:
    """Ensure a public PAI-Bench checkout exists and contains generation/evaluate.py."""
    root = root.resolve()
    if root.exists() and (root / "generation" / "evaluate.py").is_file():
        commit = _git_capture(("git", "rev-parse", "HEAD"), cwd=root)
        if (root / ".git").exists():
            if fetch:
                _run(("git", "fetch", "--tags", "origin"), cwd=root)
            _run(("git", "checkout", revision), cwd=root)
            commit = _git_capture(("git", "rev-parse", "HEAD"), cwd=root)
        return PaiBenchCheckout(
            path=root,
            repo_url=repo_url,
            revision=revision,
            resolved_commit=commit,
        )

    if root.exists():
        raise RuntimeError(
            f"PAI-Bench root exists but lacks generation/evaluate.py: {root}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    _run(("git", "clone", repo_url, str(root)), cwd=root.parent)
    _run(("git", "checkout", revision), cwd=root)
    return PaiBenchCheckout(
        path=root,
        repo_url=repo_url,
        revision=revision,
        resolved_commit=_git_capture(("git", "rev-parse", "HEAD"), cwd=root),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_warnings = _runner_setup_warnings(args)
    for warning in setup_warnings:
        print(f"PAI-Bench warning: {warning}", file=sys.stderr)
    quality_dir = args.quality_dir.resolve()
    quality_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output.resolve()

    video = args.video.resolve() if str(args.video) else args.video
    if not video.is_file():
        _write_payload(
            output_path,
            status="skipped",
            profile=args.profile,
            metrics={},
            warnings=[*setup_warnings, f"candidate MP4 not found: {video}"],
        )
        print(f"PAI-Bench profile skipped: candidate MP4 not found: {video}")
        return 0

    try:
        checkout = ensure_pai_bench_checkout(
            root=args.pai_bench_root,
            repo_url=args.pai_bench_repo,
            revision=args.pai_bench_revision,
            fetch=args.fetch,
        )
        preflight = _run_preflight(
            runner=args.runner,
            python_command=shlex.split(args.python),
            checkout=checkout,
            dimensions=args.dimensions,
            log_path=quality_dir / "pai_bench_preflight.log",
        )
        if not preflight.passed:
            warnings = [
                *setup_warnings,
                *_format_preflight_warnings(preflight),
            ]
            _write_payload(
                output_path,
                status="fail",
                profile=args.profile,
                metrics={},
                warnings=warnings,
                metadata={
                    "checkout": checkout.to_dict(),
                    "dimensions": list(args.dimensions),
                    "preflight": preflight.to_manifest(),
                    "runner": args.runner,
                    "score_range": "0-100",
                },
            )
            for warning in warnings:
                print(f"PAI-Bench preflight failed: {warning}", file=sys.stderr)
            print(f"PAI-Bench preflight log: {preflight.log_path}", file=sys.stderr)
            return 1
        staged = _stage_video_and_prompt(
            video=video,
            quality_dir=quality_dir,
            scenario_id=args.scenario_id,
            prompt=args.prompt,
            video_id=args.video_id,
            profile=args.profile,
            segment_duration_s=args.segment_duration,
        )
    except Exception as exc:  # noqa: BLE001 - report setup failures in artifacts.
        cleanup_warning = None
        if not args.keep_staged_videos:
            cleanup_warning = _remove_staged_videos(quality_dir / "staged" / "videos")
        warnings = [
            *setup_warnings,
            f"PAI-Bench setup failed: {type(exc).__name__}: {exc}",
        ]
        if cleanup_warning:
            warnings.append(cleanup_warning)
        _write_payload(
            output_path,
            status="fail",
            profile=args.profile,
            metrics={},
            warnings=warnings,
        )
        print(f"PAI-Bench setup failed: {exc}", file=sys.stderr)
        return 1

    eval_output_dir = quality_dir / "pai_bench_outputs"
    external_log_path = quality_dir / "pai_bench_command.log"
    invocation = _build_paibench_invocation(
        runner=args.runner,
        python_command=shlex.split(args.python),
        checkout=checkout,
        videos_path=staged.videos_dir,
        prompt_file=staged.prompt_file,
        output_path=eval_output_dir,
        dimensions=args.dimensions,
        nproc_per_node=args.nproc_per_node,
        custom_image_folder=args.custom_image_folder,
    )
    _write_json(quality_dir / "checkout.json", checkout.to_dict())
    _write_json(quality_dir / "staged_inputs.json", staged.to_manifest())
    _write_json(quality_dir / "pai_bench_command.json", invocation.to_manifest())

    print(f"PAI-Bench profile: {args.profile}")
    print(f"PAI-Bench runner: {invocation.runner}")
    print(f"PAI-Bench checkout: {checkout.path}")
    print(f"PAI-Bench staged videos: {staged.videos_dir}")
    print(f"PAI-Bench external log: {external_log_path}")
    returncode = _run_external(
        invocation.command,
        cwd=invocation.cwd,
        env_overrides=invocation.env,
        log_path=external_log_path,
    )

    metrics, result_files = _collect_metrics(
        eval_output_dir,
        profile=args.profile,
        dimensions=args.dimensions,
    )
    if not metrics:
        metrics = _collect_metrics_from_log(
            external_log_path,
            profile=args.profile,
            dimensions=args.dimensions,
        )

    cleanup_warning = None
    if not args.keep_staged_videos:
        cleanup_warning = _remove_staged_videos(staged.videos_dir)
        if cleanup_warning is None:
            print(f"PAI-Bench staged videos removed: {staged.videos_dir}")

    status = "pass" if returncode == 0 and metrics else "fail"
    warnings = list(setup_warnings)
    if returncode != 0:
        warnings.append(f"PAI-Bench evaluator exited with {returncode}")
    if returncode == 0 and not metrics:
        warnings.append("PAI-Bench evaluator completed without parseable metrics")
    if cleanup_warning:
        warnings.append(cleanup_warning)

    _write_payload(
        output_path,
        status=status,
        profile=args.profile,
        metrics=metrics,
        warnings=warnings,
        metadata={
            "command": shlex.join(invocation.command),
            "command_manifest": invocation.to_manifest(),
            "returncode": returncode,
            "checkout": checkout.to_dict(),
            "staged_inputs": staged.to_manifest(),
            "external_log_path": external_log_path.as_posix(),
            "preflight": preflight.to_manifest(),
            "result_files": [path.as_posix() for path in result_files],
            "dimensions": list(args.dimensions),
            "runner": invocation.runner,
            "score_range": "0-100",
            "staged_videos_retained": bool(args.keep_staged_videos),
        },
    )
    print(f"PAI-Bench metrics: {output_path}")
    return 0 if status == "pass" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("pai-bench-g", "pai-bench-long"), required=True
    )
    parser.add_argument(
        "--runner",
        choices=("local", "upstream"),
        default="local",
        help=(
            "PAI-Bench execution mode. 'local' uses the selected Python with "
            "FlashDreams compatibility shims; 'upstream' runs the public "
            "checkout exactly as before."
        ),
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--video-id")
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pai-bench-root", type=Path, required=True)
    parser.add_argument("--pai-bench-repo", default=DEFAULT_PAI_BENCH_REPO)
    parser.add_argument("--pai-bench-revision", default=DEFAULT_PAI_BENCH_REVISION)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--segment-duration", type=int, default=10)
    parser.add_argument("--custom-image-folder", type=Path)
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep-staged-videos",
        action="store_true",
        help=(
            "Keep copied or segmented MP4 inputs under quality/.../staged/videos "
            "for evaluator debugging."
        ),
    )
    args = parser.parse_args(argv)
    if args.segment_duration <= 0:
        parser.error("--segment-duration must be > 0")
    if args.nproc_per_node <= 0:
        parser.error("--nproc-per-node must be > 0")
    if not shlex.split(args.python):
        parser.error("--python must not be empty")
    return args


def _runner_setup_warnings(args: argparse.Namespace) -> list[str]:
    machine = platform.machine().lower()
    if args.runner == "upstream" and machine in ("aarch64", "arm64"):
        return [
            "upstream PAI-Bench dependencies include decord, which may not "
            "resolve on aarch64; use --pai-bench-runner local unless you have "
            "prepared a compatible upstream environment"
        ]
    return []


def _stage_video_and_prompt(
    *,
    video: Path,
    quality_dir: Path,
    scenario_id: str,
    prompt: str,
    video_id: str | None,
    profile: ProfileName,
    segment_duration_s: int,
) -> StagedInputs:
    staged_dir = quality_dir / "staged"
    videos_dir = staged_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    safe_video_id = _safe_video_id(video_id or scenario_id)
    prompt_text = (
        prompt.strip() or f"FlashDreams local benchmark scenario {scenario_id}"
    )

    if profile == "pai-bench-long":
        video_ids = _stage_video_segments(
            source=video,
            videos_dir=videos_dir,
            base_video_id=safe_video_id,
            segment_duration_s=segment_duration_s,
        )
        segment_duration: int | None = segment_duration_s
    else:
        target = videos_dir / f"{safe_video_id}__0.mp4"
        _link_or_copy(video, target)
        video_ids = (safe_video_id,)
        segment_duration = None

    prompt_file = staged_dir / "prompt_file.json"
    _write_json(
        prompt_file,
        [{"video_id": item, "prompt_en": prompt_text} for item in video_ids],
    )
    return StagedInputs(
        videos_dir=videos_dir,
        prompt_file=prompt_file,
        video_ids=video_ids,
        source_video=video,
        segment_duration_s=segment_duration,
    )


def _stage_video_segments(
    *,
    source: Path,
    videos_dir: Path,
    base_video_id: str,
    segment_duration_s: int,
) -> tuple[str, ...]:
    try:
        import cv2 as cv2_module  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PAI-Bench-Long staging requires opencv-python-headless"
        ) from exc

    cv2 = cast(Any, cv2_module)
    cap = cv2.VideoCapture(str(source))
    writer = None
    video_ids: list[str] = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open candidate MP4: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frames_per_segment = max(1, int(round(fps * segment_duration_s)))
        frame_index = 0
        segment_index = -1
        current_video_id = ""
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frames_per_segment == 0:
                if writer is not None:
                    writer.release()
                segment_index += 1
                current_video_id = f"{base_video_id}_seg{segment_index:03d}"
                video_ids.append(current_video_id)
                height, width = frame.shape[:2]
                target = videos_dir / f"{current_video_id}__0.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                new_writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
                if not new_writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {target}")
                writer = new_writer
            if writer is None:
                raise RuntimeError("video writer was not initialized")
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not video_ids:
        target = videos_dir / f"{base_video_id}__0.mp4"
        _link_or_copy(source, target)
        return (base_video_id,)
    return tuple(video_ids)


def _safe_video_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "flashdreams_benchmark"


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    try:
        os.symlink(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def _remove_staged_videos(videos_dir: Path) -> str | None:
    if not videos_dir.exists() and not videos_dir.is_symlink():
        return None
    try:
        if videos_dir.is_symlink():
            videos_dir.unlink()
        else:
            shutil.rmtree(videos_dir)
    except OSError as exc:
        return f"PAI-Bench staged video cleanup failed: {exc}"
    return None


def _build_paibench_invocation(
    *,
    runner: PaiBenchRunner,
    python_command: Sequence[str],
    checkout: PaiBenchCheckout,
    videos_path: Path,
    prompt_file: Path,
    output_path: Path,
    dimensions: Sequence[str],
    nproc_per_node: int,
    custom_image_folder: Path | None,
) -> PaiBenchInvocation:
    if runner == "local":
        training_script = str(checkout.path / "generation" / "evaluate.py")
        env = _local_paibench_env(checkout)
        cwd = Path.cwd()
    else:
        training_script = "evaluate.py"
        env = {}
        cwd = checkout.path / "generation"

    command = [
        *python_command,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        training_script,
        "--mode",
        "custom_input",
        "--prompt_file",
        str(prompt_file),
        "--dimension",
        *dimensions,
        "--videos_path",
        str(videos_path),
        "--output_path",
        str(output_path),
    ]
    if custom_image_folder is not None:
        command.extend(("--custom_image_folder", str(custom_image_folder)))
    return PaiBenchInvocation(
        command=tuple(command),
        cwd=cwd,
        env=env,
        runner=runner,
    )


def _run_preflight(
    *,
    runner: PaiBenchRunner,
    python_command: Sequence[str],
    checkout: PaiBenchCheckout,
    dimensions: Sequence[str],
    log_path: Path,
) -> PaiBenchPreflightResult:
    invocation = _build_paibench_preflight_invocation(
        runner=runner,
        python_command=python_command,
        checkout=checkout,
        dimensions=dimensions,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(invocation.env)
    try:
        completed = subprocess.run(
            list(invocation.command),
            cwd=str(invocation.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = completed.stdout
        returncode = int(completed.returncode)
    except OSError as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        returncode = 127

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command: {shlex.join(invocation.command)}\n")
        log.write(f"cwd: {invocation.cwd}\n\n")
        if invocation.env:
            log.write("env_overrides:\n")
            for key, value in sorted(invocation.env.items()):
                log.write(f"  {key}={value}\n")
            log.write("\n")
        log.write(output)

    return PaiBenchPreflightResult(
        command=invocation.command,
        cwd=invocation.cwd,
        env=invocation.env,
        log_path=log_path,
        payload=_parse_preflight_payload(output),
        returncode=returncode,
    )


def _build_paibench_preflight_invocation(
    *,
    runner: PaiBenchRunner,
    python_command: Sequence[str],
    checkout: PaiBenchCheckout,
    dimensions: Sequence[str],
) -> PaiBenchInvocation:
    if runner == "local":
        env = _local_paibench_env(checkout)
        cwd = Path.cwd()
    else:
        env = {}
        cwd = checkout.path / "generation"
    return PaiBenchInvocation(
        command=tuple([*python_command, "-c", _PREFLIGHT_SCRIPT, *dimensions]),
        cwd=cwd,
        env=env,
        runner=runner,
    )


def _parse_preflight_payload(output: str) -> Mapping[str, Any] | None:
    for line in reversed(output.splitlines()):
        if not line.startswith(_PREFLIGHT_MARKER):
            continue
        try:
            payload = json.loads(line.removeprefix(_PREFLIGHT_MARKER))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            return payload
        return None
    return None


def _preflight_status(payload: Mapping[str, Any] | None) -> str | None:
    status = payload.get("status") if payload else None
    return str(status) if isinstance(status, str) else None


def _format_preflight_warnings(result: PaiBenchPreflightResult) -> list[str]:
    warnings: list[str] = []
    payload = result.payload
    failures = payload.get("failures") if payload else None
    if isinstance(failures, list):
        for item in failures:
            if isinstance(item, Mapping):
                warnings.append(_format_preflight_failure(item))
    if warnings:
        return warnings
    return [
        "PAI-Bench dependency preflight failed before evaluator launch; "
        f"see {result.log_path} for details"
    ]


def _format_preflight_failure(failure: Mapping[str, Any]) -> str:
    dimension = str(failure.get("dimension") or "unknown dimension")
    error_type = str(failure.get("error_type") or "ImportError")
    message = str(failure.get("message") or "module import failed")
    missing_import = failure.get("missing_import")
    if isinstance(missing_import, str) and missing_import:
        package = _IMPORT_PACKAGE_HINTS.get(missing_import)
        package_hint = f" ({package})" if package else ""
        return (
            f"PAI-Bench dependency preflight failed for {dimension}: "
            f"missing Python import '{missing_import}'{package_hint}. "
            "Pass `--pai-bench-python` pointing at a prepared evaluator "
            "environment that contains the requested PAI-Bench dependencies."
        )
    return (
        f"PAI-Bench dependency preflight failed for {dimension}: "
        f"{error_type}: {message}. See the preflight log for traceback."
    )


def _local_paibench_env(checkout: PaiBenchCheckout) -> dict[str, str]:
    pythonpath_parts = [
        str(_paibench_compat_path()),
        str(checkout.path / "generation"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "FLASHDREAMS_PAI_BENCH_LOCAL_RUNNER": "1",
    }


def _paibench_compat_path() -> Path:
    return Path(__file__).resolve().parent / "_pai_bench_compat"


def _run_external(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command: {shlex.join(command)}\n")
        log.write(f"cwd: {cwd}\n\n")
        if env_overrides:
            log.write("env_overrides:\n")
            for key, value in sorted(env_overrides.items()):
                log.write(f"  {key}={value}\n")
            log.write("\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _collect_metrics(
    output_dir: Path,
    *,
    profile: ProfileName,
    dimensions: Sequence[str],
) -> tuple[dict[str, float | int], list[Path]]:
    files = sorted(
        output_dir.glob("**/*_eval_results.json"),
        key=lambda path: path.stat().st_mtime,
    )
    metrics: dict[str, float | int] = {}
    dimension_scores: dict[str, float] = {}
    videos_seen: set[str] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed_scores, parsed_videos = _parse_paibench_result_payload(
            payload,
            dimensions=dimensions,
        )
        dimension_scores.update(parsed_scores)
        videos_seen.update(parsed_videos)

    prefix = _metric_prefix(profile)
    for dimension, score in sorted(dimension_scores.items()):
        metrics[f"{prefix}_{dimension}_score"] = score
    if dimension_scores:
        metrics[f"{prefix}_score"] = sum(dimension_scores.values()) / len(
            dimension_scores
        )
        metrics[f"{prefix}_dimensions_evaluated"] = len(dimension_scores)
    if videos_seen:
        metrics[f"{prefix}_videos_evaluated"] = len(videos_seen)
    return metrics, files


def _parse_paibench_result_payload(
    payload: object,
    *,
    dimensions: Sequence[str],
) -> tuple[dict[str, float], set[str]]:
    if not isinstance(payload, Mapping):
        return {}, set()
    scores: dict[str, float] = {}
    videos_seen: set[str] = set()
    for dimension in dimensions:
        value = payload.get(dimension)
        score = _aggregate_score(value, dimension=dimension)
        if score is not None:
            scores[dimension] = score
        videos_seen.update(_video_paths(value))
    return scores, videos_seen


def _aggregate_score(value: object, *, dimension: str) -> float | None:
    raw_score: float | None = None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if (
            value
            and isinstance(value[0], (int, float))
            and not isinstance(value[0], bool)
        ):
            raw_score = float(value[0])
    elif isinstance(value, Mapping):
        for key in ("average", "mean", "score", "overall", "aggregate"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                raw_score = float(candidate)
                break
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_score = float(value)
    if raw_score is None or raw_score < 0.0 or not math.isfinite(raw_score):
        return None
    video_score = _average_video_results(value)
    if (
        video_score is not None
        and raw_score <= 1.0
        and video_score > 1.0
        and math.isfinite(video_score)
    ):
        return video_score
    if raw_score > 1.0:
        return raw_score
    return raw_score * 100.0


def _average_video_results(value: object) -> float | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2 or not isinstance(value[1], list):
        return None
    scores: list[float] = []
    for item in value[1]:
        if not isinstance(item, Mapping):
            continue
        score = item.get("video_results")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score_float = float(score)
            if score_float >= 0.0 and math.isfinite(score_float):
                scores.append(score_float)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _video_paths(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    if len(value) < 2 or not isinstance(value[1], list):
        return set()
    videos: set[str] = set()
    for item in value[1]:
        if not isinstance(item, Mapping):
            continue
        video_path = item.get("video_path")
        if isinstance(video_path, str):
            videos.add(video_path)
    return videos


def _collect_metrics_from_log(
    log_path: Path,
    *,
    profile: ProfileName,
    dimensions: Sequence[str],
) -> dict[str, float | int]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    prefix = _metric_prefix(profile)
    metrics: dict[str, float | int] = {}
    dimension_scores: dict[str, float] = {}
    expected = set(dimensions)
    aggregate_key = "paibench_long" if profile == "pai-bench-long" else "paibench_g"
    for line in lines:
        match = _LOG_SCORE_RE.match(line)
        if match is None:
            continue
        metric = match.group("metric")
        value = float(match.group("value"))
        if value < 0.0 or not math.isfinite(value):
            continue
        if metric == aggregate_key:
            metrics[f"{prefix}_score"] = value
        elif metric in expected:
            dimension_scores[metric] = value
    for dimension, score in dimension_scores.items():
        metrics[f"{prefix}_{dimension}_score"] = score
    if dimension_scores:
        metrics.setdefault(
            f"{prefix}_score",
            sum(dimension_scores.values()) / len(dimension_scores),
        )
        metrics[f"{prefix}_dimensions_evaluated"] = len(dimension_scores)
    return metrics


def _metric_prefix(profile: ProfileName) -> str:
    return "pai_bench_long" if profile == "pai-bench-long" else "pai_bench_g"


def _run(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(command), cwd=str(cwd), check=True)


def _git_capture(command: Sequence[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_payload(
    path: Path,
    *,
    status: str,
    profile: str,
    metrics: Mapping[str, float | int],
    warnings: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "profile": profile,
        "metrics": dict(metrics),
    }
    if warnings:
        payload["warnings"] = list(warnings)
    if metadata:
        payload["metadata"] = dict(metadata)
    _write_json(path, payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
