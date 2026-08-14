# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral video-to-video application and session implementations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import zipfile
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from flashdreams.demo import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.results import StepResult
from flashdreams.infra.runner_io import (
    read_video_fps,
    read_video_rgb,
    rgb_video_to_normalized_tensor,
    write_video_tensor,
)
from flashdreams.infra.video_output import VideoResultCollector, prepare_video_for_mp4
from flashdreams.runtime import StepRequirements


@dataclass(frozen=True, kw_only=True, slots=True)
class V2VApplicationDefaults:
    """Integration-provided defaults for a video-to-video application."""

    pipeline_config: Any
    """Pipeline configuration owned by the concrete model integration."""

    first_chunk_frames: int
    """Input frames consumed by the cold-start autoregressive step."""

    chunk_frames: int
    """Input frames consumed by each steady-state autoregressive step."""

    default_input_height: int
    """Input height used to preload the model before a browser upload."""

    default_input_width: int
    """Input width used to preload the model before a browser upload."""

    pipeline_config_for_video: Callable[[Any, int, int], Any] = (
        lambda config, _h, _w: config
    )
    """Optional integration hook that adapts a pipeline config to video dimensions."""

    device: str = "cuda"
    """Default device on which the pipeline is constructed."""

    output_layout: VideoTensorLayout = "bcthw"
    """Layout emitted by the integration pipeline."""

    model_name: str = "video-to-video"
    """Human-readable model name included in downloaded metadata."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _V2VSessionConfig:
    """Resolved settings for one video-to-video session."""

    defaults: V2VApplicationDefaults
    input_path: Path | None
    device: str
    fps: float | None
    pipeline_for_video: Callable[[int, int, str], Any]


class _V2VUploadStore:
    """Thread-safe browser-upload queue for a persistent V2V session."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paths: deque[Path] = deque()
        self._latest_path: Path | None = None

    def take(self) -> Path:
        """Block until the browser supplies the next source video."""
        with self._condition:
            while not self._paths:
                self._condition.wait()
            return self._paths.popleft()

    async def upload(self, request: Any) -> Any:
        """Persist the browser upload and make it available to the session."""
        from aiohttp import web

        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "video" or not field.filename:
            raise web.HTTPBadRequest(reason="Expected a multipart 'video' file.")
        suffix = Path(field.filename).suffix or ".mp4"
        descriptor, filename = tempfile.mkstemp(
            prefix="flashdreams-v2v-", suffix=suffix
        )
        path = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := await field.read_chunk():
                    output.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        with self._condition:
            self._paths.append(path)
            self._latest_path = path
            self._condition.notify_all()
        return web.json_response(
            {"source_url": "/api/v2v/source", "name": field.filename}
        )

    async def source(self, _: Any) -> Any:
        """Serve the accepted source video for browser-side preview."""
        from aiohttp import web

        with self._condition:
            path = self._latest_path
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    def configure_app(self, app: Any) -> None:
        """Register browser upload and source-preview routes."""
        app.router.add_post("/api/v2v/upload", self.upload)
        app.router.add_get("/api/v2v/source", self.source)

    def close(self) -> None:
        """Delete the temporary upload after the application terminates."""
        with self._condition:
            paths = tuple(self._paths)
            if self._latest_path is not None:
                paths += (self._latest_path,)
            self._paths.clear()
            self._latest_path = None
        for path in set(paths):
            path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _V2VWebConfiguration:
    """Assets and routes attached to the shared WebRTC application server."""

    model_web_resource: Any
    configure_app: Callable[[Any], None]
    cleanup: Callable[[], None]
    open: Callable[[SessionInfo], None]
    record: Callable[[StepResult], None]


class _V2VDownloadArchive:
    """Build one server-side MP4 and metadata ZIP per completed rollout."""

    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name
        self._directory = Path(tempfile.mkdtemp(prefix="flashdreams-v2v-output-"))
        self._collector: VideoResultCollector | None = None
        self._session_info: SessionInfo | None = None
        self._archive: Path | None = None

    def open(self, session_info: SessionInfo) -> None:
        self._session_info = session_info

    def record(self, result: StepResult) -> None:
        if result.layout is None or self._session_info is None:
            return
        if self._collector is None:
            self._collector = VideoResultCollector(output_layout=result.layout)
        self._collector.add(result)
        if result.metadata.get("rollout_complete"):
            self._finish(result)

    def _finish(self, result: StepResult) -> None:
        assert self._collector is not None
        assert result.layout is not None
        video = self._collector.finish()
        if video is None:
            return
        writable, layout = prepare_video_for_mp4(video, layout=result.layout)
        info = self._session_info
        assert info is not None and info.frames_per_second is not None
        mp4_path = self._directory / "generated.mp4"
        write_video_tensor(
            writable, mp4_path, fps=info.frames_per_second, layout=layout
        )
        metadata = {
            "model_name": self._model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video": {
                "frames": sum(
                    item["frames"]
                    for item in self._collector.stats_history
                    if isinstance(item.get("frames"), int)
                ),
                "fps": info.frames_per_second,
                "width": info.video_width,
                "height": info.video_height,
                "layout": result.layout,
                "tensor_shape": tuple(int(dim) for dim in video.shape),
            },
            "generation_metrics": self._collector.stats_history,
            "session": dict(info.metadata),
        }
        metadata_path = self._directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        archive = self._directory / "flashdreams-v2v-result.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            output.write(mp4_path, mp4_path.name)
            output.write(metadata_path, metadata_path.name)
        self._archive = archive
        self._collector = None

    async def download(self, _: Any) -> Any:
        from aiohttp import web

        if self._archive is None:
            raise web.HTTPNotFound(reason="No completed rollout is available yet.")
        return web.FileResponse(self._archive)

    def configure_app(self, app: Any) -> None:
        app.router.add_get("/api/v2v/download", self.download)

    def close(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)


class V2VApplication(IFlashDreamsApplication):
    """Reusable video-to-video application configured by one integration."""

    session_type: type["V2VApplicationSession"]

    def __init__(self, *, defaults: V2VApplicationDefaults) -> None:
        self.defaults = defaults
        self._session_config: _V2VSessionConfig | None = None
        self._upload_store: _V2VUploadStore | None = None
        self._cached_pipeline: Any | None = None
        self._cached_pipeline_key: tuple[int, int, str] | None = None

    @property
    def requires_pre_session_web(self) -> bool:
        """Start the WebRTC page before waiting for an optional browser upload."""
        return self._upload_store is not None

    def configure_webrtc(self, factory: Any) -> None:
        """Attach the V2V upload/preview UI to the shared WebRTC server."""
        self._upload_store = _V2VUploadStore()
        upload_store = self._upload_store
        archive = _V2VDownloadArchive(model_name=self.defaults.model_name)

        def configure_app(app: Any) -> None:
            upload_store.configure_app(app)
            archive.configure_app(app)

        def cleanup() -> None:
            upload_store.close()
            archive.close()

        factory.set_web_configuration(
            _V2VWebConfiguration(
                model_web_resource=files("v2v").joinpath("web"),
                configure_app=configure_app,
                cleanup=cleanup,
                open=archive.open,
                record=archive.record,
            )
        )

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare that input video is application-owned, not a live control."""
        return CanonicalInputSchema(description="file-backed video-to-video")

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse video source and session overrides."""
        parser = argparse.ArgumentParser(prog="flashdreams-run <v2v-slug>")
        parser.add_argument("--input-path")
        parser.add_argument("--device", default=self.defaults.device)
        parser.add_argument("--fps", type=float)
        args = parser.parse_args(list(commandline_args))
        if args.fps is not None and args.fps <= 0:
            raise ValueError("--fps must be greater than zero.")
        input_path = Path(args.input_path) if args.input_path else None
        if input_path is None and self._upload_store is None:
            raise ValueError("--input-path is required outside the WebRTC upload UI.")
        self._session_config = _V2VSessionConfig(
            defaults=self.defaults,
            input_path=input_path,
            device=args.device,
            fps=args.fps,
            pipeline_for_video=self._pipeline_for_video,
        )

    def _pipeline_for_video(self, height: int, width: int, device: str) -> Any:
        """Reuse the last matching pipeline or replace it for a new resolution."""
        key = (height, width, device)
        if self._cached_pipeline_key == key:
            return self._cached_pipeline
        previous = self._cached_pipeline
        pipeline_config = self.defaults.pipeline_config_for_video(
            self.defaults.pipeline_config, height, width
        )
        pipeline = pipeline_config.setup().to(device).eval()
        self._cached_pipeline = pipeline
        self._cached_pipeline_key = key
        if previous is not None:
            close = getattr(previous, "close", None)
            if callable(close):
                close()
        return pipeline

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create an isolated session for the configured input video."""
        if self._session_config is None:
            raise RuntimeError(
                "V2VApplication.init() must run before create_session()."
            )
        return self.session_type(
            config=self._session_config, upload_store=self._upload_store
        )


class V2VApplicationSession(IFlashDreamsApplicationSession):
    """Reusable cache-isolated video-to-video model session."""

    def __init__(
        self,
        *,
        config: _V2VSessionConfig,
        upload_store: _V2VUploadStore | None = None,
    ) -> None:
        self.config = config
        self._upload_store = upload_store
        self._pipeline: Any | None = None
        self._cache: Any | None = None
        self._video: torch.Tensor | None = None
        self._chunks: tuple[tuple[int, int], ...] = ()
        self._step_index = 0
        self._fps: float | None = None
        self._input_path: Path | None = None
        self._initial_input_path = config.input_path
        self._closed = False

    def init(self) -> None:
        """Load the input video, construct the pipeline, and initialize its cache."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed V2V session.")
        if self._pipeline is not None:
            return
        pipeline = self.config.pipeline_for_video(
            self.config.defaults.default_input_height,
            self.config.defaults.default_input_width,
            self.config.device,
        )
        input_path = self._initial_input_path
        if input_path is None:
            if self._upload_store is None:
                raise RuntimeError("V2V session has no source-video provider.")
            input_path = self._upload_store.take()
        self._load_source(input_path)

    def _load_source(self, input_path: Path) -> None:
        """Load a source video and reset the model cache for a fresh rollout."""
        if not input_path.is_file():
            raise ValueError(f"--input-path must be an existing video: {input_path}")
        self._input_path = input_path

        video_np = read_video_rgb(input_path)
        if video_np.ndim != 4 or video_np.shape[0] == 0:
            raise ValueError("--input-path must contain at least one RGB video frame.")
        frames, height, width, _channels = video_np.shape
        chunks = self._build_chunks(frames)
        if not chunks:
            raise ValueError(
                f"Input video needs at least {self.config.defaults.first_chunk_frames} frames."
            )

        pipeline = self.config.pipeline_for_video(height, width, self.config.device)
        self._video = (
            rgb_video_to_normalized_tensor(
                video_np, device=torch.device("cpu"), dtype=torch.float32
            )
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )
        self._chunks = chunks
        self._fps = self.config.fps or read_video_fps(input_path)
        self._pipeline = pipeline
        self._cache = pipeline.initialize_cache()
        self._step_index = 0
        self._initial_input_path = None

    def _build_chunks(self, total_frames: int) -> tuple[tuple[int, int], ...]:
        """Plan full cold-start and steady-state chunks for the source video."""
        chunks: list[tuple[int, int]] = []
        start = 0
        size = self.config.defaults.first_chunk_frames
        while start + size <= total_frames:
            chunks.append((start, size))
            start += size
            size = self.config.defaults.chunk_frames
        return tuple(chunks)

    def session_info(self) -> SessionInfo:
        """Return output geometry and timing after pipeline initialization."""
        if self._pipeline is None:
            raise RuntimeError(
                "V2VApplicationSession.init() must run before session_info()."
            )
        encoder = getattr(self._pipeline, "encoder", None)
        return SessionInfo(
            output_layout=self.config.defaults.output_layout,
            frames_per_second=self._fps,
            video_width=getattr(encoder, "target_W", None),
            video_height=getattr(encoder, "target_H", None),
            metadata={"input_path": str(self._input_path or "browser-upload")},
        )

    def next_step_requirements(self) -> StepRequirements | None:
        """Return requirements for the next available source-video chunk."""
        if self._step_index >= len(self._chunks):
            if self._upload_store is None:
                return None
            self._load_source(self._upload_store.take())
        return StepRequirements(step_index=self._step_index)

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Transform the next input-video chunk into one output-video chunk."""
        if self._closed:
            raise RuntimeError("V2V session is closed.")
        if self._pipeline is None or self._cache is None or self._video is None:
            raise RuntimeError("V2VApplicationSession.init() must run before step().")
        if self._step_index >= len(self._chunks):
            raise RuntimeError("V2V session has no remaining steps.")
        if inputs.values:
            raise ValueError("V2V does not declare live canonical inputs.")

        start, size = self._chunks[self._step_index]
        dtype = self._pipeline.diffusion_model.dtype
        source = self._video[:, :, start : start + size].to(
            device=self._pipeline.device, dtype=dtype
        )
        generated = self._pipeline.generate(self._step_index, self._cache, source)
        finalized = self._pipeline.finalize(self._step_index, self._cache)
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=generated.detach(),
            layout=self.config.defaults.output_layout,
            metadata={
                "input_path": str(self._input_path or "browser-upload"),
                "rollout_complete": self._step_index + 1 == len(self._chunks),
            },
            metrics=finalized if isinstance(finalized, dict) else {},
        )
        self._step_index += 1
        return result

    def close(self) -> None:
        """Release resources owned by this model session."""
        if self._closed:
            return
        self._closed = True
        self._cache = None
        self._pipeline = None
        self._video = None


V2VApplication.session_type = V2VApplicationSession


__all__ = ["V2VApplication", "V2VApplicationDefaults", "V2VApplicationSession"]
