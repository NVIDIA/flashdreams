# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed ``flashdreams-run t2v`` launch implementation."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aiohttp import web

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.runtime.demo.host import RuntimeHost
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.demo import serve_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCRuntimeConfig

from .backends import backend_metadata, resolve_backend
from .runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    T2VDemoAdapter,
    make_adapter,
)

if TYPE_CHECKING:
    from .runner import T2VDemoRunnerConfig


@dataclass(frozen=True, slots=True)
class T2VWebRTCConfig(WebRTCRuntimeConfig):
    """Shared WebRTC settings required by the prompt-only T2V demo."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class T2VWebRTCSessionManager(BaseWebRTCSessionManager[Any, T2VWebRTCConfig]):
    """Shared manager with a prompt update for the next browser session."""

    def update_prompt(self, prompt: str, duration_s: float) -> None:
        if not prompt.strip():
            raise ValueError("Prompt must be non-empty.")
        if not 0 < duration_s <= 60:
            raise ValueError("Duration must be greater than 0 and at most 60 seconds.")
        spec = self._shared_spec
        adapter = self._shared_adapter
        if spec is None or adapter is None:
            raise RuntimeError("T2V WebRTC shared session is not initialized.")
        scenario = dict(spec.scenario or {})
        scenario[FIELD_PROMPT] = prompt.strip()
        scenario[FIELD_TOTAL_BLOCKS] = self.runtime.blocks_for_duration(
            duration_s, fps=_int_value(scenario[FIELD_FPS], name=FIELD_FPS)
        )
        spec = replace(spec, scenario=scenario)
        self._shared_spec = spec
        self._shared_scenario = adapter.prepare_scenario(spec)


def launch_t2v(
    *,
    config: "T2VDemoRunnerConfig",
    mode: Literal["mp4", "null", "webrtc"],
    scenario_overrides: dict[str, object] | None = None,
    output_overrides: dict[str, object] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> object:
    """Launch T2V directly from its typed ``flashdreams-run`` configuration."""
    configure_logging()
    scenario_overrides = scenario_overrides or {}
    output_overrides = output_overrides or {}
    adapter = make_adapter(config.backend)
    scenario = _scenario(config, scenario_overrides)
    if mode == "mp4" or mode == "null":
        output = _replay_output(
            mode=mode,
            output_path=output_overrides.get(
                "path", output_overrides.get("output", config.output)
            ),
            fps=_int_value(
                output_overrides.get("fps", scenario[FIELD_FPS]), name="fps"
            ),
        )
        result = run_replay_demo(
            spec=_spec(
                config,
                adapter=adapter,
                scenario=scenario,
                input_mode="replay",
                output=output,
            ),
            adapter=adapter,
        )
        if result.status != "completed":
            reason = result.reason or str(result.error) or "T2V replay failed."
            raise RuntimeError(reason)
        return result

    context = initialize_cuda_distributed(default_device=config.device)
    output = WebRTCOutputSpec(
        host=str(host or output_overrides.get("host", "0.0.0.0")),
        port=_int_value(
            port if port is not None else output_overrides.get("port", 8080),
            name="port",
        ),
        fps=_int_value(output_overrides.get("fps", scenario[FIELD_FPS]), name="fps"),
        video_width=_int_value(
            output_overrides.get("video_width", scenario[FIELD_PIXEL_WIDTH]),
            name="video_width",
        ),
        video_height=_int_value(
            output_overrides.get("video_height", scenario[FIELD_PIXEL_HEIGHT]),
            name="video_height",
        ),
        warmup_chunks=_int_value(
            output_overrides.get("warmup_chunks", 0), name="warmup_chunks"
        ),
        warmup_timeout_s=_float_value(
            output_overrides.get("warmup_timeout_s", 600.0),
            name="warmup_timeout_s",
        ),
        client_liveness_timeout_s=_float_value(
            output_overrides.get("client_liveness_timeout_s", 30.0),
            name="client_liveness_timeout_s",
        ),
        preload_name="FlashDreams T2V",
    )
    spec = _spec(
        config,
        adapter=adapter,
        scenario=scenario,
        input_mode="webrtc",
        output=output,
        device=str(context.device),
    )
    prepared = adapter.prepare_scenario(spec)
    inference_config = spec.config
    if inference_config is None:
        raise RuntimeError("T2V DemoSpec.config was not initialized.")
    runtime = adapter.create_runtime(inference_config)
    manager = T2VWebRTCSessionManager(
        runtime=runtime,
        runtime_config=T2VWebRTCConfig(
            video_width=output.video_width,
            video_height=output.video_height,
            warmup_chunks=output.warmup_chunks,
            warmup_timeout_s=output.warmup_timeout_s,
        ),
        fps=output.fps,
        identity=adapter.model_id,
        supported_control_keys=frozenset({"g"}),
        shared_host=RuntimeHost(runtime),
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=prepared,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        keep_connection_after_completed=True,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=adapter.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("t2v_demo").joinpath("web"),
            configure_app=lambda app: _configure_app(
                app, manager=manager, backend=config.backend
            ),
            preload_name="FlashDreams T2V",
        ),
        world_rank=context.world_rank,
    )


def _scenario(
    config: "T2VDemoRunnerConfig", overrides: dict[str, object]
) -> dict[str, object]:
    runner = resolve_backend(config.backend).resolve_runner(config.preset_id)

    def value(name: str, default: object) -> object:
        overridden = overrides.get(name)
        configured = getattr(config, name)
        return (
            default
            if overridden is None and configured is None
            else (configured if overridden is None else overridden)
        )

    return {
        FIELD_PROMPT: value(FIELD_PROMPT, runner.prompt),
        FIELD_TOTAL_BLOCKS: value(FIELD_TOTAL_BLOCKS, runner.total_blocks),
        FIELD_PIXEL_HEIGHT: value(FIELD_PIXEL_HEIGHT, runner.pixel_height),
        FIELD_PIXEL_WIDTH: value(FIELD_PIXEL_WIDTH, runner.pixel_width),
        FIELD_FPS: value(FIELD_FPS, runner.fps),
    }


def _spec(
    config: "T2VDemoRunnerConfig",
    *,
    adapter: T2VDemoAdapter,
    scenario: dict[str, object],
    input_mode: Literal["replay", "webrtc"],
    output: Mp4OutputSpec | NullOutputSpec | WebRTCOutputSpec,
    device: str | None = None,
) -> DemoSpec:
    return DemoSpec(
        model_id=adapter.model_id,
        preset_id=config.preset_id or adapter.backend.default_preset_name,
        input_mode=input_mode,
        scenario=scenario,
        output=output,
        config=InferenceConfig(
            model_id=adapter.model_id,
            preset_id=config.preset_id or adapter.backend.default_preset_name,
            device=device or config.device,
            compile=config.compile,
            runtime_options={"backend": adapter.backend.key},
        ),
    )


def _replay_output(
    *, mode: Literal["mp4", "null"], output_path: object, fps: int
) -> Mp4OutputSpec | NullOutputSpec:
    if mode == "null":
        return NullOutputSpec()
    if output_path is None:
        raise ValueError("T2V MP4 mode requires an output path.")
    return Mp4OutputSpec(path=Path(str(output_path)), fps=fps, output_layout="tchw")


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{name} must be convertible to int, got {type(value).__name__}.")


def _float_value(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{name} must be convertible to float, got {type(value).__name__}.")


def _configure_app(
    app: web.Application,
    *,
    manager: T2VWebRTCSessionManager,
    backend: str,
) -> None:
    async def app_config(_: web.Request) -> web.StreamResponse:
        return web.json_response(
            {"backends": backend_metadata(), "selected_backend": backend}
        )

    async def update_prompt(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            raise web.HTTPBadRequest(reason="Expected a JSON prompt.")
        duration_s = payload.get("duration_s")
        if not isinstance(duration_s, int | float):
            raise web.HTTPBadRequest(reason="Expected numeric duration_s.")
        try:
            manager.update_prompt(payload["prompt"], float(duration_s))
        except (RuntimeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"status": "ok"})

    async def download(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None:
            raise web.HTTPNotFound(reason="No completed generation is available yet.")
        video_path, scenario = artifact
        if not video_path.is_file():
            raise web.HTTPNotFound(reason="Generated MP4 is no longer available.")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(video_path, "video.mp4")
            archive.writestr(
                "prompt.json",
                json.dumps(
                    {
                        "prompt": scenario.prompt,
                        "total_blocks": scenario.total_blocks,
                        "fps": scenario.fps,
                        "width": scenario.pixel_width,
                        "height": scenario.pixel_height,
                    },
                    indent=2,
                ),
            )
        return web.Response(
            body=buffer.getvalue(),
            headers={
                "Content-Disposition": "attachment; filename=flashdreams-generation.zip"
            },
            content_type="application/zip",
        )

    async def playback(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None or not artifact[0].is_file():
            raise web.HTTPNotFound(reason="No completed MP4 is available yet.")
        return web.FileResponse(artifact[0])

    app.router.add_get("/api/t2v/config", app_config)
    app.router.add_post("/api/t2v/prompt", update_prompt)
    app.router.add_get("/api/t2v/download", download)
    app.router.add_get("/api/t2v/playback", playback)


__all__ = ["T2VWebRTCSessionManager", "launch_t2v"]
