# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

import dataclasses
import mimetypes
import shutil
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    Cam2VSlangPyUILoop,
    generate_camera_step,
)
from loguru import logger

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from lingbot.apps.cam2v.scene import SceneState
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.impl.conditioning import resolve_lingbot_conditioning

LINGBOT_CAM2V_DEFAULTS = Cam2VApplicationDefaults(
    pipeline_config=derive_config(
        PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
        enable_sync_and_profile=False,
    ),
    input_resolver=resolve_lingbot_conditioning,
    total_blocks=20,
    pixel_width=832,
    pixel_height=464,
    first_frame_dtype=torch.bfloat16,
    first_frame_interpolation="cubic",
    fps=16,
    log_model_timing=True,
    install_hint="Install the Lingbot integration: pip install flashdreams-lingbot.",
    input_defaults={"example_data": False, "example_idx": 0},
)
"""Lingbot defaults for the reusable Cam2V application."""


class _SilentCam2VUILoop(Cam2VSlangPyUILoop):
    """Cam2V UI loop that presents frames without drawing the overlay.

    The camera-controls panel is drawn into the generated frames before they
    are encoded, so it is part of the video and cannot be dismissed from the
    browser. ``--no-ui`` removes it by registering no UI loop at all, but a UI
    loop is the only thing able to ask the runtime for a replacement session,
    which is how the page switches scenes. This keeps the loop and drops only
    the drawing: no widgets are built, so nothing is composited over the
    frame, and the state the overlay would have displayed goes unmaintained.
    """

    def step_ui(self, ui: Any, step_index: int, events: Any) -> Any:
        """Return the current model frame, building no widgets."""
        del ui, step_index, events
        return self.presented_model_frame()


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World specialization of the shared Cam2V application.

    Also serves the browser UI under ``web/``: a scene picker whose presets
    carry a prompt, a first frame, and a catalog of text events. Implementing
    :class:`IWebUiProvider` is what makes the v2 WebRTC server route
    ``/request_session`` and the ``/api/session`` endpoints here.
    """

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """Select the pipeline config used by this Cam2V application.

        Args:
            pipeline_config: Model variant to run; ``None`` uses the application
                default.
        """
        defaults = LINGBOT_CAM2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(
                defaults,
                pipeline_config=derive_config(
                    pipeline_config, enable_sync_and_profile=False
                ),
            )
        # Wrap the generation step so a requested prompt swap is applied on
        # the model thread between chunks, where the rollout cache is idle.
        defaults = dataclasses.replace(defaults, generate_step=self._generate_step)
        super().__init__(defaults=defaults)
        self._scene: SceneState | None = None
        self._pending_prompt: str | None = None
        self._active_prompt: str | None = None
        self._prompt_lock = threading.Lock()
        self._prompt_embeddings: dict[str, Any] = {}
        self._active_cache: Any | None = None
        self._ui_loop: Any | None = None
        self._draw_overlay: bool | None = None
        self._scene_dir: Path | None = None

    def _configure_argument_parser(self, parser: Any) -> None:
        """Default the overlay off, since it is drawn into the video itself.

        The shared application defaults ``--ui`` on, which composites the
        camera-controls panel into the generated frames -- part of the video,
        not something the browser can dismiss. ``--ui`` still turns it on for
        the timing readout.
        """
        super()._configure_argument_parser(parser)
        parser.set_defaults(ui=False)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create the rollout and describe it for the browser UI."""
        # --ui selects which UI loop runs, not whether one runs at all. A UI
        # loop is the only thing that can ask the runtime for a replacement
        # session, so registering none -- what --no-ui does -- would silently
        # disable switching scenes from the page. The flag keeps the meaning
        # its name implies: draw the overlay, or do not.
        #
        # Recorded once: forcing _use_ui on below would otherwise be read back
        # as "the overlay was asked for" by the next session, so a scene
        # switch brought it back.
        if self._draw_overlay is None:
            self._draw_overlay = self._use_ui
        draw_overlay = self._draw_overlay
        self._use_ui = True
        session = super().create_session(session_desc)
        # The shared session registers its loops without keeping one, so the
        # bound method is wrapped to capture what it registers.
        register_ui_loop = session.register_ui_loop

        def capture_ui_loop(loop_type: Any, **kwargs: Any) -> Any:
            if loop_type is Cam2VSlangPyUILoop and not draw_overlay:
                loop_type = _SilentCam2VUILoop
            loop = register_ui_loop(loop_type, **kwargs)
            self._ui_loop = loop
            return loop

        session.register_ui_loop = capture_ui_loop  # type: ignore[method-assign]
        self._ui_loop = None
        scene = self._scene
        if scene is None:
            self._scene = self._build_scene(session_desc)
        else:
            # A page may have built the scene already, before the runtime got
            # this far; take the resolution the session actually resolved to.
            scene.video_width = session_desc.video_width
            scene.video_height = session_desc.video_height
        # A new rollout re-encodes from its own prompt, so nothing carries
        # over from the previous one.
        with self._prompt_lock:
            self._pending_prompt = None
            self._active_prompt = None
        return session

    def _generate_step(
        self,
        pipeline: Any,
        autoregressive_index: int,
        cache: Any,
        camera_input: Any,
    ) -> torch.Tensor:
        """Apply any pending text swap, then generate the chunk as usual."""
        if cache is not self._active_cache:
            # A reset discards the cache, and the next step re-seeds it from
            # the session's own conditioning -- which restores the prompt the
            # rollout started with. Re-apply whatever the page has chosen
            # since, so a reset clears the drifted image without also
            # reverting the scene.
            self._active_cache = cache
            self._active_prompt = None
            scene = self._scene
            if scene is not None and scene.active_prompt() != "":
                with self._prompt_lock:
                    self._pending_prompt = scene.active_prompt()
        with self._prompt_lock:
            prompt = self._pending_prompt
            self._pending_prompt = None
        if prompt is not None and prompt != self._active_prompt:
            self._swap_text_context(pipeline, cache, prompt)
            self._active_prompt = prompt
        return generate_camera_step(
            pipeline, autoregressive_index, cache, camera_input
        )

    def _swap_text_context(self, pipeline: Any, cache: Any, prompt: str) -> None:
        """Replace the rollout's cross-attention text context with ``prompt``.

        Only the static text context changes: the self-attention KV cache
        stays intact, so the rollout keeps its horizon and the swap shows up
        from this chunk on.

        Raises:
            RuntimeError: This pipeline's transformer cannot swap text.
        """
        transformer = pipeline.diffusion_model.transformer
        replace_text_embeddings = getattr(transformer, "replace_text_embeddings", None)
        if not callable(replace_text_embeddings):
            raise RuntimeError(
                "Lingbot text events need a pipeline whose transformer supports "
                "replace_text_embeddings; this pipeline does not."
            )
        embeddings = self._prompt_embeddings.get(prompt)
        if embeddings is None:
            # initialize_cache releases the one-shot encoders once the rollout
            # starts, so the text encoder is reloaded before encoding here.
            pipeline._ensure_oneshot_encoders_loaded()
            embeddings = pipeline.text_encoder([prompt]).to(
                device=torch.device(self._device)
            )
            self._prompt_embeddings[prompt] = embeddings
        replace_text_embeddings(cache.transformer_cache, embeddings)
        logger.info("Lingbot text context updated: {}", prompt)

    def _build_scene(self, session_desc: SessionDesc | None = None) -> SceneState:
        """Describe the session this application runs, or is about to.

        Resolves the conditioning the same way :meth:`create_session` does, so
        the page shows the prompt and first frame the rollout actually starts
        from rather than only what was passed on the command line. Example
        data, in particular, supplies both and neither appears in the raw
        arguments.
        """
        input_values = dict(self._input_values or {})
        desc = session_desc or self.session_desc()
        prompt = str(input_values.get("prompt", ""))
        image_path = str(input_values.get("image_path") or "")
        try:
            conditioning = self.defaults.input_resolver(
                {
                    **input_values,
                    "pixel_height": desc.video_height,
                    "pixel_width": desc.video_width,
                    "fps": desc.frames_per_second_for_step,
                }
            )
        except Exception as error:  # noqa: BLE001 - the page still needs a scene
            # Resolving can download example data, so it is allowed to fail
            # without taking the browser UI with it.
            logger.debug("Lingbot scene falling back to raw inputs: {}", error)
        else:
            prompt = str(getattr(conditioning, "prompt", "") or prompt)
            image_path = str(getattr(conditioning, "first_frame_path", "") or image_path)
        return SceneState(
            prompt=prompt,
            model=getattr(self._pipeline_config, "name", type(self).__name__),
            video_width=desc.video_width,
            video_height=desc.video_height,
            first_frame_path=image_path,
        )

    def _require_scene(self) -> SceneState:
        """Return the scene, building it if no session has started yet.

        The HTTP server is serving before the runtime creates a session, so a
        browser that connects while the model is still loading would otherwise
        be told the session has not started -- which is exactly when its page
        needs the scene.
        """
        scene = self._scene
        if scene is None:
            scene = self._build_scene()
            self._scene = scene
        return scene

    def close(self) -> None:
        """Release the pipeline, and the scratch directory for uploads."""
        scene_dir = self._scene_dir
        self._scene_dir = None
        if scene_dir is not None:
            shutil.rmtree(scene_dir, ignore_errors=True)
        super().close()

    # --- IWebUiProvider ---------------------------------------------------
    def web_root(self) -> Path:
        """Return the directory holding this application's page and assets."""
        return Path(__file__).parent / "web"

    def initial_scene(self) -> Mapping[str, Any]:
        """Return the current scene for the page to render."""
        return self._require_scene().as_dict()

    def first_frame(self) -> tuple[bytes, str] | None:
        """Return the frame the session starts from, uploaded or resolved.

        A resolved first frame is a path on the server, so it is read here
        rather than handed to the page as a URL it could not load.
        """
        scene = self._require_scene()
        if scene.image_bytes is not None:
            return scene.image_bytes, scene.image_content_type
        if not scene.first_frame_path:
            return None
        path = Path(scene.first_frame_path)
        try:
            data = path.read_bytes()
        except OSError as error:
            logger.debug("Lingbot first frame unreadable: {}", error)
            return None
        content_type, _ = mimetypes.guess_type(path.name)
        return data, content_type or "application/octet-stream"

    def apply_session_input(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply one page submission, steering the rollout when it can.

        A triggered text event swaps the rollout's text conditioning in place
        so it takes effect without restarting. Prompt, first-frame, and
        catalog changes are recorded for the next session, since the frame a
        rollout was initialized from cannot be replaced mid-flight.

        Raises:
            ValueError: The submission is malformed or names an unknown event.
        """
        scene = self._require_scene()
        before = (scene.image_bytes, scene.image_url)
        prompt = scene.apply(payload)
        if (scene.image_bytes, scene.image_url) != before:
            # A different first frame is a different scene, and a rollout
            # cannot swap the frame it was initialized from -- so ask for a
            # new session rather than steering the current one, which would
            # only blend the new prompt into the old world.
            self._restart_on(scene)
        elif prompt is not None:
            # Handed to the model thread rather than applied here: this runs
            # on the HTTP thread, where the rollout cache is in use.
            with self._prompt_lock:
                self._pending_prompt = prompt
        return scene.as_dict()

    def _restart_on(self, scene: SceneState) -> None:
        """Ask the runtime for a session that starts from ``scene``.

        Does nothing when the scene names no image, or when no UI loop has
        registered yet -- the only thing able to make the request. The scene
        is recorded either way, so the next session starts from it.

        Raises:
            ValueError: The scene names an image that cannot seed a rollout.
        """
        if not scene.image_url and scene.image_bytes is None:
            return
        first_frame = self._materialize_first_frame(scene)
        if first_frame is None:
            # The scene asked for an image and none could be produced. Failing
            # here surfaces it as a 400 on the page rather than leaving the
            # rollout quietly generating from the frame it already had.
            raise ValueError(
                f"No usable start image for this scene: {scene.image_url!r} "
                "could not be resolved. Preset images are paths under the "
                "application's web/ directory; a remote URL is loaded by the "
                "browser and cannot seed a rollout."
            )
        if str(first_frame) == scene.first_frame_path:
            # The page posts its scene on every connect, so without this a
            # reconnect rebuilt the session -- and paid the warmup again --
            # for a frame the rollout had already started from.
            return
        scene.first_frame_path = str(first_frame)
        # Only the frame and prompt are overridden. The resolver takes an
        # explicit image_path over the example's own ("image_path or
        # example_dir / image.jpg"), while example data still supplies the
        # intrinsics and poses it requires and nothing else provides --
        # turning it off here made every replacement session fail with
        # "Lingbot Cam2V requires intrinsic_path".
        self._input_values = {
            **(self._input_values or {}),
            "prompt": scene.prompt,
            "image_path": str(first_frame),
        }
        ui_loop = self._ui_loop
        if ui_loop is None:
            logger.info("Lingbot scene recorded; it starts on the next session.")
            return
        ui_loop.request_new_session(self.session_desc())
        logger.info("Lingbot restarting on a new scene: {}", first_frame.name)

    def _materialize_first_frame(self, scene: SceneState) -> Path | None:
        """Return a readable path for the scene's first frame.

        Uploads are written out, and a page-supplied URL is resolved inside
        the web root, which is where the built-in presets keep their images.
        A URL pointing anywhere else is not fetched -- the server does not
        make outbound requests on a page's behalf.
        """
        if scene.image_bytes is not None:
            if self._scene_dir is None:
                self._scene_dir = Path(tempfile.mkdtemp(prefix="lingbot-scene-"))
            suffix = mimetypes.guess_extension(scene.image_content_type) or ".jpg"
            path = self._scene_dir / f"first_frame{suffix}"
            path.write_bytes(scene.image_bytes)
            return path
        if scene.image_url.startswith(("http://", "https://")):
            return None
        root = self.web_root().resolve()
        candidate = (root / scene.image_url).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(root):
            return None
        return candidate


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


def create_app_fast() -> IApplication:
    """Return the Lingbot World Fast application."""
    return LingbotCam2VApplication(pipeline_config=PIPELINE_LINGBOT_WORLD_FAST)


def create_app_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World Fast bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3
    )


def create_app_v2_14b_causal_fast() -> IApplication:
    """Return the Lingbot World v2 14B Causal Fast application."""
    return LingbotCam2VApplication(
        pipeline_config=PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST
    )


def create_app_v2_14b_causal_fast_taehv_window15_sink3() -> IApplication:
    """Return the Lingbot World v2 bounded-window TAEHV application."""
    return LingbotCam2VApplication(
        pipeline_config=(PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3)
    )


__all__ = [
    "LingbotCam2VApplication",
    "create_app",
    "create_app_fast",
    "create_app_fast_taehv_window15_sink3",
    "create_app_v2_14b_causal_fast",
    "create_app_v2_14b_causal_fast_taehv_window15_sink3",
]
