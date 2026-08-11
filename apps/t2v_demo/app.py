# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the shared-runtime text-to-video replay or WebRTC demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from aiohttp import web

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.app import DemoApplication
from flashdreams.runtime.demo.host import RuntimeHost
from flashdreams.runtime.demo.webrtc import serve_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCRuntimeConfig

from .backends import backend_choices, backend_metadata, resolve_backend
from .runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    T2VDemoAdapter,
    make_adapter,
)


@dataclass(frozen=True, slots=True)
class T2VWebRTCConfig(WebRTCRuntimeConfig):
    """Subset of the shared WebRTC configuration used by prompt-only T2V."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse replay and WebRTC launch options."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("replay", "webrtc"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--backend", choices=backend_choices(), default="causal-forcing")
        subparser.add_argument("--preset-id", default=None)
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
        subparser.add_argument("--prompt", default=None)
        subparser.add_argument("--total-blocks", type=int, default=None)
        subparser.add_argument("--pixel-height", type=int, default=None)
        subparser.add_argument("--pixel-width", type=int, default=None)
        subparser.add_argument("--fps", type=int, default=None)
    replay = subparsers.choices["replay"]
    replay.add_argument("--output-mode", choices=("mp4", "null"), default="mp4")
    replay.add_argument("--output", type=Path, default=None)
    webrtc = subparsers.choices["webrtc"]
    webrtc.add_argument("--host", default="0.0.0.0")
    webrtc.add_argument("--port", type=int, default=8080)
    webrtc.add_argument("--warmup-chunks", type=int, default=0)
    webrtc.add_argument("--warmup-timeout-s", type=float, default=600.0)
    webrtc.add_argument("--client-liveness-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.command == "replay" and args.output_mode == "mp4" and args.output is None:
        parser.error("replay --output is required when --output-mode=mp4.")
    return args


class T2VDemoApplication(DemoApplication):
    """Text-to-video demo using the common replay and WebRTC runtimes."""

    def __init__(self) -> None:
        self._backend = "causal-forcing"

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        args = parse_args(argv)
        self._backend = args.backend
        return args

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        return _spec(args, input_mode="replay", output=_replay_output(args))

    def replay_adapter(self) -> T2VDemoAdapter:
        return make_adapter(self._backend)

    def serve_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        adapter = make_adapter(args.backend)
        output = WebRTCOutputSpec(
            host=args.host,
            port=args.port,
            fps=_scenario(args)[FIELD_FPS],
            video_width=_scenario(args)[FIELD_PIXEL_WIDTH],
            video_height=_scenario(args)[FIELD_PIXEL_HEIGHT],
            warmup_chunks=args.warmup_chunks,
            warmup_timeout_s=args.warmup_timeout_s,
            client_liveness_timeout_s=args.client_liveness_timeout_s,
            preload_name="FlashDreams T2V",
        )
        spec = _spec(args, input_mode="webrtc", output=output, device=str(context.device))
        prepared = adapter.prepare_scenario(spec)
        runtime = adapter.create_runtime(spec.config)
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
        )
        serve_webrtc_demo(
            output=output,
            model_id=adapter.model_id,
            session_manager=manager,
            app_resources=WebRTCAppResources(
                model_web_resource=files("apps.t2v_demo").joinpath("web"),
                configure_app=lambda app: _configure_app(app, manager=manager, args=args),
                preload_name="FlashDreams T2V",
            ),
            world_rank=context.world_rank,
        )


class T2VWebRTCSessionManager(BaseWebRTCSessionManager[Any, T2VWebRTCConfig]):
    """Shared manager with a prompt update for the next browser session."""

    def update_prompt(self, prompt: str) -> None:
        if self.has_active_session():
            raise RuntimeError("Wait for the current generation to finish before changing the prompt.")
        if not prompt.strip():
            raise ValueError("Prompt must be non-empty.")
        scenario = dict(self._shared_spec.scenario or {})
        scenario[FIELD_PROMPT] = prompt.strip()
        self._shared_spec = replace(self._shared_spec, scenario=scenario)
        self._shared_scenario = self._shared_adapter.prepare_scenario(self._shared_spec)


def _configure_app(app: web.Application, *, manager: T2VWebRTCSessionManager, args: argparse.Namespace) -> None:
    async def config(_: web.Request) -> web.StreamResponse:
        return web.json_response({"backends": backend_metadata(), "selected_backend": args.backend})

    async def update_prompt(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            raise web.HTTPBadRequest(reason="Expected a JSON prompt.")
        try:
            manager.update_prompt(payload["prompt"])
        except (RuntimeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"status": "ok"})

    app.router.add_get("/api/t2v/config", config)
    app.router.add_post("/api/t2v/prompt", update_prompt)


def _scenario(args: argparse.Namespace) -> dict[str, object]:
    runner = resolve_backend(args.backend).resolve_runner(args.preset_id)
    return {
        FIELD_PROMPT: args.prompt or str(getattr(runner, FIELD_PROMPT)),
        FIELD_TOTAL_BLOCKS: args.total_blocks or int(getattr(runner, FIELD_TOTAL_BLOCKS, 1)),
        FIELD_PIXEL_HEIGHT: args.pixel_height or int(getattr(runner, FIELD_PIXEL_HEIGHT, 480)),
        FIELD_PIXEL_WIDTH: args.pixel_width or int(getattr(runner, FIELD_PIXEL_WIDTH, 832)),
        FIELD_FPS: args.fps or int(getattr(runner, FIELD_FPS, 16)),
    }


def _spec(args: argparse.Namespace, *, input_mode: str, output: object, device: str | None = None) -> DemoSpec:
    backend = resolve_backend(args.backend)
    return DemoSpec(
        model_id="flashdreams-t2v",
        preset_id=args.preset_id or backend.default_preset_name,
        input_mode=input_mode,
        scenario=_scenario(args),
        output=output,
        config=InferenceConfig(
            model_id="flashdreams-t2v",
            preset_id=args.preset_id or backend.default_preset_name,
            device=device or args.device,
            compile=args.compile,
            runtime_options={"backend": backend.key},
        ),
    )


def _replay_output(args: argparse.Namespace) -> Mp4OutputSpec | NullOutputSpec:
    if args.output_mode == "null":
        return NullOutputSpec()
    return Mp4OutputSpec(path=args.output, fps=_scenario(args)[FIELD_FPS], output_layout="tchw")


def main(argv: list[str] | None = None) -> None:
    """Run the T2V demo."""
    T2VDemoApplication().main(argv)


if __name__ == "__main__":
    main()
