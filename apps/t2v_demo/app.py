# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed ``flashdreams-run t2v`` launch implementation."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aiohttp import web

from flashdreams.demo import (
    Application,
    ApplicationSession,
    FileOutputSink,
    InferenceSessionApplicationAdapter,
    Runner,
    create_replay_io_handler,
)
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
from flashdreams.runtime.demo.run_modes import RunResult
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
    T2VRuntime,
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


@dataclass(frozen=True, slots=True)
class T2VApplicationDefaults:
    """Author-facing defaults for one prompt-only text-to-video app."""

    backend: str = "causal-forcing"
    preset_id: str | None = None
    prompt: str | None = None
    total_blocks: int | None = None
    pixel_height: int | None = None
    pixel_width: int | None = None
    fps: int | None = None
    device: str = "cuda"
    compile: bool | None = None


@dataclass(slots=True)
class T2VApplication:
    """Small public app facade over the selected T2V runtime preset."""

    defaults: T2VApplicationDefaults = field(default_factory=T2VApplicationDefaults)
    _runtimes: list[T2VRuntime] = field(default_factory=list, init=False, repr=False)

    def init(self, launch_args: Sequence[str]) -> None:
        if launch_args:
            raise ValueError(
                "T2VApplication does not support launch arguments; pass defaults "
                "when constructing the app."
            )
        resolve_backend(self.defaults.backend).resolve_runner(self.defaults.preset_id)

    def create_session(self) -> ApplicationSession:
        adapter = make_adapter(self.defaults.backend)
        scenario = _scenario(self.defaults)
        spec = _spec(
            self.defaults,
            adapter=adapter,
            scenario=scenario,
            input_mode="replay",
            output=NullOutputSpec(),
        )
        if spec.config is None:
            raise RuntimeError("T2V DemoSpec.config was not initialized.")
        prepared = adapter.prepare_scenario(spec)
        runtime = adapter.create_runtime(spec.config)
        self._runtimes.append(runtime)
        return InferenceSessionApplicationAdapter(
            runtime.start_session(prepared.initial_inputs)
        )

    def close(self) -> None:
        errors: list[Exception] = []
        while self._runtimes:
            runtime = self._runtimes.pop()
            try:
                runtime.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"T2VApplication.close failed for {len(errors)} runtime(s)."
            ) from errors[0]


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
    defaults = _defaults_from_config(config, scenario_overrides)
    scenario = _scenario(defaults)
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
        return _run_replay_application(
            app=T2VApplication(defaults=defaults),
            output=output,
        )

    context = initialize_cuda_distributed(default_device=config.device)
    adapter = make_adapter(defaults.backend)
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
        replace(defaults, device=str(context.device)),
        adapter=adapter,
        scenario=scenario,
        input_mode="webrtc",
        output=output,
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


def create_app(config: "T2VDemoRunnerConfig | None" = None) -> Application:
    """Create the default public T2V application without CLI parsing."""
    from .runner import RUNNER_T2V

    config = RUNNER_T2V if config is None else config
    return T2VApplication(defaults=_defaults_from_config(config, {}))


createApp = create_app


def _defaults_from_config(
    config: "T2VDemoRunnerConfig",
    overrides: dict[str, object],
) -> T2VApplicationDefaults:
    def value(name: str) -> object:
        return getattr(config, name) if overrides.get(name) is None else overrides[name]

    return T2VApplicationDefaults(
        backend=config.backend,
        preset_id=config.preset_id,
        prompt=_optional_str(value(FIELD_PROMPT)),
        total_blocks=_optional_int(value(FIELD_TOTAL_BLOCKS)),
        pixel_height=_optional_int(value(FIELD_PIXEL_HEIGHT)),
        pixel_width=_optional_int(value(FIELD_PIXEL_WIDTH)),
        fps=_optional_int(value(FIELD_FPS)),
        device=config.device,
        compile=config.compile,
    )


def _scenario(defaults: T2VApplicationDefaults) -> dict[str, object]:
    runner = resolve_backend(defaults.backend).resolve_runner(defaults.preset_id)

    def value(name: str, default: object) -> object:
        configured = getattr(defaults, name)
        return default if configured is None else configured

    return {
        FIELD_PROMPT: value(FIELD_PROMPT, runner.prompt),
        FIELD_TOTAL_BLOCKS: value(FIELD_TOTAL_BLOCKS, runner.total_blocks),
        FIELD_PIXEL_HEIGHT: value(FIELD_PIXEL_HEIGHT, runner.pixel_height),
        FIELD_PIXEL_WIDTH: value(FIELD_PIXEL_WIDTH, runner.pixel_width),
        FIELD_FPS: value(FIELD_FPS, runner.fps),
    }


def _spec(
    defaults: T2VApplicationDefaults,
    *,
    adapter: T2VDemoAdapter,
    scenario: dict[str, object],
    input_mode: Literal["replay", "webrtc"],
    output: Mp4OutputSpec | NullOutputSpec | WebRTCOutputSpec,
) -> DemoSpec:
    return DemoSpec(
        model_id=adapter.model_id,
        preset_id=defaults.preset_id or adapter.backend.default_preset_name,
        input_mode=input_mode,
        scenario=scenario,
        output=output,
        config=InferenceConfig(
            model_id=adapter.model_id,
            preset_id=defaults.preset_id or adapter.backend.default_preset_name,
            device=defaults.device,
            compile=defaults.compile,
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


def _optional_int(value: object) -> int | None:
    return None if value is None else _int_value(value, name="value")


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _run_replay_application(
    *,
    app: Application,
    output: Mp4OutputSpec | NullOutputSpec,
) -> RunResult:
    output_sink = None
    if isinstance(output, Mp4OutputSpec):
        output_sink = FileOutputSink(
            output_path=output.path,
            fps=output.fps,
            output_layout=output.output_layout,
        )
    result = Runner(
        io_handler=create_replay_io_handler(output_sink=output_sink),
        app=app,
    ).run()
    if result.status != "completed":
        reason = result.reason or str(result.error) or "T2V replay failed."
        raise RuntimeError(reason)
    return result


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


__all__ = [
    "T2VApplication",
    "T2VApplicationDefaults",
    "T2VWebRTCSessionManager",
    "createApp",
    "create_app",
    "launch_t2v",
]
