# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2V-specific WebRTC controls, recording, playback, and downloads."""

from __future__ import annotations

import io
import json
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol, cast

from aiohttp import web

from flashdreams.runtime import InferenceInput, InferenceRuntime
from flashdreams.runtime.demo import (
    DemoSpec,
    PreparedScenario,
    RuntimeHost,
    WebRTCAppResources,
    WebRTCOutputSpec,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCRuntimeConfig
from flashdreams_runner import AppConfig, Runtime
from flashdreams_runner.webrtc import ModelInputProviderFactory

DEFAULT_DURATION_S = 5.0
"""Initial browser duration when a preset has no finite block count."""

MAX_DURATION_S = 60.0
"""Prototype UI limit that bounds one browser generation request."""


class _Scenario(Protocol):
    @property
    def prompt(self) -> str: ...

    @property
    def total_blocks(self) -> int | None: ...

    @property
    def pixel_height(self) -> int: ...

    @property
    def pixel_width(self) -> int: ...

    @property
    def fps(self) -> int: ...


class _Artifact(Protocol):
    @property
    def path(self) -> Path: ...

    @property
    def scenario(self) -> _Scenario: ...


class _T2VRuntime(Protocol):
    @property
    def config(self) -> AppConfig: ...

    @property
    def latest_artifact(self) -> _Artifact | None: ...

    def prepare_session_input(
        self,
        *,
        prompt: str | None = None,
        total_blocks: int | None = None,
    ) -> InferenceInput: ...

    def blocks_for_duration(self, duration_s: float) -> int: ...


class T2VWebRTCSessionManager(
    BaseWebRTCSessionManager[_T2VRuntime, WebRTCRuntimeConfig]
):
    """Keep one browser connection alive across finite T2V generations."""

    def update_generation(self, *, prompt: str, duration_s: float) -> None:
        """Prepare the prompt and duration used by the next generation."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        if not 0 < duration_s <= MAX_DURATION_S:
            raise ValueError(
                f"Duration must be greater than 0 and at most {MAX_DURATION_S:g} "
                "seconds."
            )
        self._shared_scenario = PreparedScenario(
            initial_inputs=self.runtime.prepare_session_input(
                prompt=prompt,
                total_blocks=self.runtime.blocks_for_duration(duration_s),
            )
        )


class T2VWebRTCCustomization:
    """Install T2V browser assets and HTTP routes into the runner mode."""

    def __init__(self, *, runtime: _T2VRuntime) -> None:
        self._runtime = runtime

    def prepare_initial_input(self) -> InferenceInput:
        """Return a complete finite input for the first browser session."""
        total_blocks = self._runtime.config.default_steps
        if total_blocks is None:
            total_blocks = self._runtime.blocks_for_duration(DEFAULT_DURATION_S)
        return self._runtime.prepare_session_input(total_blocks=total_blocks)

    def create_session_manager(
        self,
        *,
        runtime: Runtime,
        output: WebRTCOutputSpec,
        spec: DemoSpec,
        scenario: PreparedScenario,
        input_provider_factory: ModelInputProviderFactory,
    ) -> BaseWebRTCSessionManager[Any, Any]:
        """Create a finite-generation manager over the initialized runtime."""
        if runtime is not cast(object, self._runtime):
            raise ValueError("T2V WebRTC customization received a different runtime.")
        inference_runtime = cast(InferenceRuntime, runtime)
        return T2VWebRTCSessionManager(
            runtime=self._runtime,
            runtime_config=cast(WebRTCRuntimeConfig, cast(object, output)),
            fps=int(self._runtime.config.fps),
            identity=self._runtime.config.model_id,
            supported_control_keys=frozenset(),
            shared_host=RuntimeHost(inference_runtime),
            shared_spec=spec,
            shared_scenario=scenario,
            shared_model_input_provider_factory=input_provider_factory,
            client_liveness_timeout_s=output.client_liveness_timeout_s,
            keep_connection_after_completed=True,
            runtime_ready=True,
        )

    def create_app_resources(
        self,
        *,
        session_manager: BaseWebRTCSessionManager[Any, Any],
    ) -> WebRTCAppResources:
        """Return the T2V adapter assets and application-owned routes."""
        if not isinstance(session_manager, T2VWebRTCSessionManager):
            raise TypeError("T2V WebRTC requires T2VWebRTCSessionManager.")
        return WebRTCAppResources(
            model_web_resource=files("t2v_app").joinpath("web"),
            configure_app=lambda app: _configure_app(
                app,
                manager=session_manager,
            ),
            preload_name="FlashDreams T2V",
        )


def _configure_app(
    app: web.Application,
    *,
    manager: T2VWebRTCSessionManager,
) -> None:
    """Register T2V metadata, prompt, playback, and download endpoints."""

    async def app_config(_: web.Request) -> web.StreamResponse:
        config = manager.runtime.config
        initial_input = manager.runtime.prepare_session_input()
        return web.json_response(
            {
                "model_id": config.model_id,
                "default_prompt": initial_input.global_conditioning["prompt"],
                "default_duration_s": DEFAULT_DURATION_S,
            }
        )

    async def update_generation(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            raise web.HTTPBadRequest(reason="Expected a JSON prompt.")
        duration_s = payload.get("duration_s")
        if isinstance(duration_s, bool) or not isinstance(duration_s, int | float):
            raise web.HTTPBadRequest(reason="Expected numeric duration_s.")
        try:
            manager.update_generation(
                prompt=payload["prompt"],
                duration_s=float(duration_s),
            )
        except (RuntimeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"status": "ok"})

    async def download(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None:
            raise web.HTTPNotFound(reason="No completed generation is available yet.")
        if not artifact.path.is_file():
            raise web.HTTPNotFound(reason="Generated MP4 is no longer available.")
        scenario = artifact.scenario
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(artifact.path, "video.mp4")
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
                "Content-Disposition": (
                    "attachment; filename=flashdreams-generation.zip"
                )
            },
            content_type="application/zip",
        )

    async def playback(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None or not artifact.path.is_file():
            raise web.HTTPNotFound(reason="No completed MP4 is available yet.")
        return web.FileResponse(artifact.path)

    app.router.add_get("/api/t2v/config", app_config)
    app.router.add_post("/api/t2v/prompt", update_generation)
    app.router.add_get("/api/t2v/download", download)
    app.router.add_get("/api/t2v/playback", playback)


__all__ = ["T2VWebRTCCustomization", "T2VWebRTCSessionManager"]
