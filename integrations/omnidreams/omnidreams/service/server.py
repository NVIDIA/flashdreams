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

"""Queued HTTP service for ``flashdreams-run`` Omnidreams rollouts."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aiohttp import web
from loguru import logger

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig
from omnidreams.runner import OmnidreamsRunner, OmnidreamsRunnerConfig

DEFAULT_RUNNER_NAME = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
"""Default runner slug used by the upload service."""

DEFAULT_HOST = "127.0.0.1"
"""Default bind host for local testing."""

DEFAULT_PORT = 8090
"""Default HTTP port for local testing."""

DEFAULT_WORK_DIR = Path("/tmp/flashdreams-omnidreams-service")
"""Default directory for uploaded inputs, logs, and generated videos."""

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TEMPORAL_COMPRESSION_RATIO = 4
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MISSING_CHECKPOINT = "MISSING"


@dataclass(frozen=True)
class RunnerLayout:
    """Static temporal and view layout for one Omnidreams runner."""

    num_views: int
    """Number of per-camera paths expected by the runner."""

    len_t: int
    """Latent frames generated per autoregressive block."""


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime configuration for the queued video-generation service."""

    runner_name: str = DEFAULT_RUNNER_NAME
    """Default runner slug for requests that do not override it."""

    work_dir: Path = DEFAULT_WORK_DIR
    """Root directory for request inputs and outputs."""

    runner_command: tuple[str, ...] = ()
    """Command prefix used to launch ``flashdreams-run``."""

    runner_args: tuple[str, ...] = ()
    """Extra CLI args appended after service-managed runner overrides."""

    cwd: Path = _REPO_ROOT
    """Working directory for the runner subprocess."""

    checkpoint_path: str | None = None
    """Optional checkpoint override applied before preloading the runner."""

    device: str = "cuda"
    """Device string used when constructing the resident runner."""

    execution_mode: Literal["in_process", "subprocess"] = "in_process"
    """Run jobs on a resident runner by default, or spawn subprocesses."""


@dataclass
class GenerationJob:
    """One queued HTTP generation request."""

    job_id: str
    """Stable request identifier returned in response headers and queue state."""

    runner_name: str
    """Runner slug to execute."""

    prompt: str
    """Prompt text passed to ``flashdreams-run --prompt``."""

    first_frame_paths: tuple[Path, ...]
    """Per-view first-frame paths, expanded to the runner's view count."""

    hdmap_video_paths: tuple[Path, ...]
    """Per-view HDMap videos, expanded to the runner's view count."""

    total_blocks: int
    """Autoregressive block count derived from the shortest HDMap video."""

    output_dir: Path
    """Directory the runner writes outputs into."""

    future: asyncio.Future[Path]
    """Completes with the generated MP4 path or the subprocess error."""

    status: str = "queued"
    """Queue lifecycle state."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when the request entered the service."""

    started_at: float | None = None
    """Unix timestamp when the worker began executing this job."""

    finished_at: float | None = None
    """Unix timestamp when the worker finished this job."""

    error: str | None = None
    """Human-readable failure summary."""

    output_path: Path | None = None
    """Generated MP4 path once the job completes."""


def default_runner_command() -> tuple[str, ...]:
    """Return the best available command prefix for the local runner."""
    script = shutil.which("flashdreams-run")
    if script is not None:
        return (script,)
    return (sys.executable, "-m", "flashdreams.scripts.cli")


def resolve_runner_config(
    runner_name: str,
    *,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    fail_on_missing_checkpoint: bool = True,
) -> RunnerConfig:
    """Resolve and patch an Omnidreams runner config.

    Args:
        runner_name: Omnidreams runner slug.
        checkpoint_path: Optional checkpoint path override.
        device: Device string for the resident runner.
        fail_on_missing_checkpoint: Raise a clear error for internal-only
            configs whose checkpoint is not published in the repo.

    Returns:
        Runner config ready for setup.

    Raises:
        KeyError: ``runner_name`` is not known to the runner registry.
    """
    runners: dict[str, Any]
    try:
        from omnidreams.config import OMNIDREAMS_RUNNERS  # noqa: PLC0415

        runners = dict(OMNIDREAMS_RUNNERS)
    except Exception:  # pragma: no cover - fallback for plugin-only installs.
        runners = {}

    if runner_name not in runners:
        from flashdreams.configs.registry import supported_runners  # noqa: PLC0415

        runners.update(supported_runners())
    cfg = runners[runner_name]
    resolved_checkpoint = (
        checkpoint_path or cfg.pipeline.diffusion_model.transformer.checkpoint_path
    )
    if fail_on_missing_checkpoint and resolved_checkpoint == _MISSING_CHECKPOINT:
        raise RuntimeError(
            f"runner {runner_name!r} has checkpoint_path='MISSING'. "
            "This runner's checkpoint is not published by the repo and "
            "is intentionally skipped by `omnidreams-prepare`. Start the "
            "service with `--checkpoint-path /path/to/checkpoint` for this "
            "runner, or choose a public runner such as "
            "'omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae'."
        )
    cfg = derive_config(
        cfg,
        device=device,
        release_oneshot_encoders_after_run=False,
    )
    if checkpoint_path is not None:
        cfg = derive_config(
            cfg,
            pipeline=dict(
                diffusion_model=dict(
                    transformer=dict(checkpoint_path=checkpoint_path),
                ),
            ),
        )
    return cfg


def runner_layout(runner_name: str) -> RunnerLayout:
    """Resolve static view and temporal layout for ``runner_name``."""
    cfg = resolve_runner_config(runner_name, fail_on_missing_checkpoint=False)
    transformer = cfg.pipeline.diffusion_model.transformer
    return RunnerLayout(
        num_views=int(transformer.num_views),
        len_t=int(transformer.len_t),
    )


def total_blocks_for_frame_count(num_frames: int, len_t: int) -> int:
    """Return the largest full AR block count that fits ``num_frames``.

    Args:
        num_frames: HDMap conditioning frame count.
        len_t: Latent frames per AR block from the runner config.

    Returns:
        Number of full blocks to request from ``flashdreams-run``.

    Raises:
        ValueError: The video is too short for the runner's first block.
    """
    initial = 1 + (len_t - 1) * _TEMPORAL_COMPRESSION_RATIO
    steady = len_t * _TEMPORAL_COMPRESSION_RATIO
    if num_frames < initial:
        raise ValueError(
            f"HDMap video has {num_frames} frames, but runner len_t={len_t} "
            f"requires at least {initial} frames for the first block."
        )
    return 1 + (num_frames - initial) // steady


def expand_to_num_views(
    paths: tuple[Path, ...], num_views: int, *, name: str
) -> tuple[Path, ...]:
    """Expand one uploaded asset to every view or validate per-view assets."""
    if len(paths) == num_views:
        return paths
    if len(paths) == 1:
        return paths * num_views
    raise ValueError(
        f"{name} received {len(paths)} file(s), but runner expects {num_views}. "
        f"Upload exactly one file to reuse it for every view, or upload {num_views} files."
    )


def count_video_frames(path: Path) -> int:
    """Count frames in an MP4 without loading the full video when possible."""
    try:
        import cv2  # noqa: PLC0415

        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if count > 0:
            return count
    except ImportError:
        pass

    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency gate.
        raise ImportError(
            "Counting HDMap frames requires opencv-python-headless or mediapy."
        ) from exc
    return int(media.read_video(str(path)).shape[0])


def build_runner_command(
    *,
    config: ServiceConfig,
    job: GenerationJob,
) -> list[str]:
    """Build the ``flashdreams-run`` command for ``job``."""
    command = list(config.runner_command or default_runner_command())
    command.extend(
        [
            job.runner_name,
            "--prompt",
            job.prompt,
            "--first-frame-paths",
            *[str(path) for path in job.first_frame_paths],
            "--hdmap-video-paths",
            *[str(path) for path in job.hdmap_video_paths],
            "--total-blocks",
            str(job.total_blocks),
            "--output-dir",
            str(job.output_dir),
        ]
    )
    command.extend(config.runner_args)
    return command


def create_app(config: ServiceConfig) -> web.Application:
    """Create the aiohttp application and its single-worker queue."""
    app = web.Application(client_max_size=1024**3)
    app["config"] = config
    app["queue"] = asyncio.Queue()
    app["jobs"] = {}
    app["current_job_id"] = None
    app["runner"] = None
    app["runner_config"] = None
    app["startup_error"] = None
    app.router.add_get("/health", handle_health)
    app.router.add_get("/queue", handle_queue)
    app.router.add_post("/generate", handle_generate)
    app.on_startup.append(_start_worker)
    app.on_cleanup.append(_stop_worker)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """Return a minimal service-health response."""
    queue: asyncio.Queue[GenerationJob] = request.app["queue"]
    return web.json_response(
        {
            "ok": True,
            "execution_mode": request.app["config"].execution_mode,
            "loaded": request.app["runner"] is not None,
            "queued": queue.qsize(),
            "current_job_id": request.app["current_job_id"],
            "startup_error": request.app["startup_error"],
        }
    )


async def handle_queue(request: web.Request) -> web.Response:
    """Return queue and recent job state."""
    jobs: dict[str, GenerationJob] = request.app["jobs"]
    return web.json_response(
        {
            "current_job_id": request.app["current_job_id"],
            "jobs": [_job_summary(job) for job in jobs.values()],
        }
    )


async def handle_generate(request: web.Request) -> web.StreamResponse:
    """Accept multipart inputs, enqueue generation, and return the MP4."""
    config: ServiceConfig = request.app["config"]
    job_id = uuid.uuid4().hex
    input_dir = config.work_dir / job_id / "inputs"
    output_dir = config.work_dir / job_id / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = ""
    runner_name = config.runner_name
    first_frames: list[Path] = []
    hdmap_videos: list[Path] = []

    reader = await request.multipart()
    async for field in reader:
        if field.name == "prompt":
            prompt = (await field.text()).strip()
        elif field.name == "runner_name":
            value = (await field.text()).strip()
            if value:
                runner_name = value
        elif field.name == "first_frame":
            first_frames.append(await _save_field(field, input_dir, "first_frame"))
        elif field.name == "hdmap_video":
            hdmap_videos.append(await _save_field(field, input_dir, "hdmap_video"))

    if not prompt:
        raise web.HTTPBadRequest(text="multipart field 'prompt' is required")
    if not first_frames:
        raise web.HTTPBadRequest(text="multipart file field 'first_frame' is required")
    if not hdmap_videos:
        raise web.HTTPBadRequest(text="multipart file field 'hdmap_video' is required")

    try:
        layout = runner_layout(runner_name)
        first_frame_paths = expand_to_num_views(
            tuple(first_frames), layout.num_views, name="first_frame"
        )
        hdmap_video_paths = expand_to_num_views(
            tuple(hdmap_videos), layout.num_views, name="hdmap_video"
        )
        frame_count = min(count_video_frames(path) for path in set(hdmap_video_paths))
        total_blocks = total_blocks_for_frame_count(frame_count, layout.len_t)
    except Exception as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc

    loop = asyncio.get_running_loop()
    job = GenerationJob(
        job_id=job_id,
        runner_name=runner_name,
        prompt=prompt,
        first_frame_paths=first_frame_paths,
        hdmap_video_paths=hdmap_video_paths,
        total_blocks=total_blocks,
        output_dir=output_dir,
        future=loop.create_future(),
    )
    jobs: dict[str, GenerationJob] = request.app["jobs"]
    jobs[job_id] = job
    await request.app["queue"].put(job)

    try:
        output_path = await asyncio.shield(job.future)
    except Exception as exc:
        raise web.HTTPInternalServerError(
            text=f"job {job_id} failed: {exc}"
        ) from exc

    return web.FileResponse(
        output_path,
        headers={
            "X-Job-Id": job_id,
            "X-Total-Blocks": str(total_blocks),
            "X-Runner-Name": runner_name,
        },
    )


async def _start_worker(app: web.Application) -> None:
    config: ServiceConfig = app["config"]
    if config.execution_mode == "in_process":
        try:
            app["runner_config"] = resolve_runner_config(
                config.runner_name,
                checkpoint_path=config.checkpoint_path,
                device=config.device,
            )
            logger.info(f"preloading Omnidreams runner {config.runner_name!r}")
            app["runner"] = app["runner_config"].setup()
            logger.info(f"preloaded Omnidreams runner {config.runner_name!r}")
        except Exception as exc:
            app["startup_error"] = str(exc)
            raise
    app["worker_task"] = asyncio.create_task(_worker(app))


async def _stop_worker(app: web.Application) -> None:
    task: asyncio.Task[None] = app["worker_task"]
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _worker(app: web.Application) -> None:
    queue: asyncio.Queue[GenerationJob] = app["queue"]
    config: ServiceConfig = app["config"]
    while True:
        job = await queue.get()
        app["current_job_id"] = job.job_id
        job.status = "running"
        job.started_at = time.time()
        try:
            if config.execution_mode == "subprocess":
                output = await _run_job_subprocess(config=config, job=job)
            else:
                output = await _run_job_in_process(app=app, job=job)
            job.status = "completed"
            job.output_path = output
            if not job.future.done():
                job.future.set_result(output)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            job.finished_at = time.time()
            app["current_job_id"] = None
            queue.task_done()


async def _run_job_subprocess(*, config: ServiceConfig, job: GenerationJob) -> Path:
    command = build_runner_command(config=config, job=job)
    log_path = job.output_dir / "flashdreams-run.log"
    logger.info(f"[{job.job_id}] launching: {shlex.join(command)}")
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=config.cwd,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    with log_path.open("wb") as log_file:
        async for chunk in process.stdout:
            log_file.write(chunk)
    return_code = await process.wait()
    if return_code != 0:
        tail = _read_tail(log_path)
        raise RuntimeError(
            f"flashdreams-run exited with code {return_code}. "
            f"See {log_path}. Tail:\n{tail}"
        )

    expected = job.output_dir / f"{job.runner_name}.mp4"
    if expected.exists():
        return expected
    candidates = sorted(job.output_dir.glob("*.mp4"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"flashdreams-run completed but wrote no MP4 in {job.output_dir}"
    )


async def _run_job_in_process(*, app: web.Application, job: GenerationJob) -> Path:
    runner = app["runner"]
    if not isinstance(runner, OmnidreamsRunner):
        raise RuntimeError("resident Omnidreams runner is not loaded")

    cfg = runner.config
    assert isinstance(cfg, OmnidreamsRunnerConfig)
    cfg.prompt = job.prompt
    cfg.prompts = ()
    cfg.first_frame_paths = job.first_frame_paths
    cfg.hdmap_video_paths = job.hdmap_video_paths
    cfg.total_blocks = job.total_blocks
    cfg.output_dir = job.output_dir
    cfg.release_oneshot_encoders_after_run = False

    logger.info(
        f"[{job.job_id}] running resident Omnidreams runner "
        f"{job.runner_name!r} total_blocks={job.total_blocks}"
    )
    await asyncio.to_thread(runner.run)

    expected = job.output_dir / f"{job.runner_name}.mp4"
    if expected.exists():
        return expected
    candidates = sorted(job.output_dir.glob("*.mp4"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"resident runner wrote no MP4 in {job.output_dir}")


async def _save_field(field: Any, input_dir: Path, prefix: str) -> Path:
    filename = _safe_filename(field.filename or f"{prefix}.bin")
    path = input_dir / f"{prefix}_{uuid.uuid4().hex}_{filename}"
    with path.open("wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    return path


def _safe_filename(filename: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", Path(filename).name).strip("._")
    return cleaned or "upload.bin"


def _read_tail(path: Path, *, max_bytes: int = 8000) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def _job_summary(job: GenerationJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "runner_name": job.runner_name,
        "status": job.status,
        "total_blocks": job.total_blocks,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "output_path": str(job.output_path) if job.output_path else None,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runner-name", default=DEFAULT_RUNNER_NAME)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--runner-command",
        default="",
        help=(
            "Command prefix for launching the runner, for example "
            "'uv run --package flashdreams-omnidreams flashdreams-run'. "
            "Defaults to flashdreams-run on PATH."
        ),
    )
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Extra argument appended to subprocess flashdreams-run invocations.",
    )
    parser.add_argument("--cwd", type=Path, default=_REPO_ROOT)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--subprocess-runner",
        action="store_true",
        help="Spawn flashdreams-run per job instead of preloading one resident runner.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the queued Omnidreams HTTP service."""
    args = _parse_args(argv)
    runner_command = (
        tuple(shlex.split(args.runner_command))
        if args.runner_command
        else default_runner_command()
    )
    config = ServiceConfig(
        runner_name=args.runner_name,
        work_dir=args.work_dir,
        runner_command=runner_command,
        runner_args=tuple(args.runner_arg),
        cwd=args.cwd,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        execution_mode="subprocess" if args.subprocess_runner else "in_process",
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    web.run_app(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
