# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive Drive application and immediate Dear ImGui HUD."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch import Tensor

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .core import (
    BackendFactory,
    DriveInputState,
    DriveTelemetry,
    InteractiveDriveApplicationDefaults,
    InteractiveDriveConfig,
    InteractiveDriveModelLoop,
    InteractiveDriveModelState,
    SceneLoader,
    ViewMode,
    _InteractiveDriveApplicationBase,
)
from .scene_loader import load_scene_bundle

_SCENE_STEM = re.compile(r"^clipgt-(?P<uuid>[0-9a-fA-F-]{36})(?:-(?P<variant>.+))?$")


@dataclass(frozen=True, slots=True)
class InteractiveDriveSceneOption:
    """One scene exposed by the Interactive Drive HUD."""

    path: Path
    label: str
    variants: tuple[str, ...] = ("default",)


@dataclass(slots=True)
class InteractiveDriveUIState:
    """UI-thread state for the Interactive Drive HUD."""

    model_loop: IModelLoop[InteractiveDriveModelState]
    title: str
    prompt: str
    scene_options: tuple[InteractiveDriveSceneOption, ...]
    current_scene_index: int = 0
    current_variant_index: int = 0
    view_mode: ViewMode = "rgb"
    viewport_width: int = 1280
    viewport_height: int = 704
    postprocess_enabled: bool = False
    show_postprocess_toggle: bool = False
    status: str = (
        "W/S drive, Space brakes, A/D steer; gamepads and wheels are supported."
    )
    telemetry: DriveTelemetry | None = None
    drive_input: DriveInputState = field(default_factory=DriveInputState)
    """Driving input updated on the UI thread."""

    sprites: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = field(
        default_factory=dict
    )
    wheel_cache_angle: int | None = None
    wheel_cache_pixels: np.ndarray[Any, np.dtype[np.uint8]] | None = None

    def set_status(self, status: str) -> None:
        """Replace the status line displayed by the HUD."""
        self.status = status

    def set_view_mode(self, view_mode: ViewMode) -> None:
        """Select the RGB, HD-map, or PhysX presentation stream."""
        self.view_mode = view_mode

    def set_drive_telemetry(self, telemetry: DriveTelemetry) -> None:
        """Accept the latest model-thread telemetry snapshot."""
        self.telemetry = telemetry


class InteractiveDriveUILoop(ImGuiUILoop[InteractiveDriveUIState]):
    """Composite the driving frame with immediate Dear ImGui controls."""

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Draw scene controls, telemetry, wheel, pedals, and BEV minimap."""
        del step_index
        state = self.state
        state.drive_input.apply(events)
        command = state.drive_input.command()
        telemetry = state.telemetry
        self._ensure_sprites()
        imgui.set_next_window_pos(
            imgui.ImVec2(float(max(12, state.viewport_width - 366)), 12.0),
            imgui.Cond_.once,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(
                354.0,
                float(max(480, min(540, state.viewport_height - 24))),
            ),
            imgui.Cond_.once,
        )
        imgui.begin(state.title)
        try:
            imgui.text("INTERACTIVE DRIVE")
            imgui.separator()
            self._draw_scene_controls(imgui)
            imgui.separator()
            if telemetry is None:
                imgui.text("Waiting for the first driving block...")
                speed_mph = 0.0
                reverse = False
            else:
                speed_mph = abs(telemetry.speed_mps) * 2.236936
                reverse = telemetry.reverse
                imgui.text(f"Block {telemetry.blocks_generated}")
            imgui.text(f"Input  {state.drive_input.source()}")
            imgui.text(f"Speed  {speed_mph:5.1f} mph")
            imgui.text(f"Gear   {'R' if reverse else 'D'}")
            imgui.text(f"Steer  {command.steer:+.2f}")
            self._progress(imgui, "Throttle", command.throttle)
            self._progress(imgui, "Brake", command.brake)
            imgui.image(
                "steering-wheel",
                self._wheel_pixels(command.steer),
                size=(138.0, 138.0),
            )
            imgui.same_line()
            imgui.image(
                "brake-pedal",
                state.sprites[
                    "brake_pressed" if command.brake > 0.05 else "brake_unpressed"
                ],
                size=(64.0, 138.0),
            )
            imgui.same_line()
            imgui.image(
                "throttle-pedal",
                state.sprites[
                    "throttle_pressed"
                    if command.throttle > 0.05
                    else "throttle_unpressed"
                ],
                size=(64.0, 138.0),
            )
            imgui.separator()
            if state.show_postprocess_toggle:
                postprocess = (
                    telemetry.postprocess_enabled
                    if telemetry is not None
                    else state.postprocess_enabled
                )
                changed, postprocess = imgui.checkbox("Post-processing", postprocess)
                if changed:
                    state.postprocess_enabled = bool(postprocess)
                    invoke_async(
                        state.model_loop,
                        lambda model_state, enabled=bool(postprocess): (
                            model_state.set_postprocess_enabled(enabled)
                        ),
                    )
            if imgui.button("Restart rollout"):
                self._restart()
            imgui.same_line()
            if imgui.button(f"View: {state.view_mode.upper()}"):
                self._toggle_view()
            imgui.text(state.status)
            imgui.separator()
            imgui.text(
                "frames_in_chunk: --"
                if telemetry is None
                else f"frames_in_chunk: {telemetry.frames_in_chunk}"
            )
            imgui.text(
                "model_loop_ms: --"
                if telemetry is None
                else f"model_loop_ms: {telemetry.model_loop_ms:.1f}"
            )
        finally:
            imgui.end()

        imgui.set_next_window_pos(
            imgui.ImVec2(12.0, float(max(12, state.viewport_height - 282))),
            imgui.Cond_.once,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(350.0, 258.0),
            imgui.Cond_.once,
        )
        imgui.begin("Map")
        try:
            try:
                bev_frame = self.presented_model_frame(1)
            except IndexError:
                bev_frame = None
            if bev_frame is not None:
                imgui.image(
                    "bev-minimap",
                    bev_frame.permute(1, 2, 0),
                    size=(318.0, 210.0),
                )
            else:
                imgui.text("Waiting for BEV output.")
        finally:
            imgui.end()
        return self.presented_model_frame()

    def _draw_scene_controls(self, imgui: Any) -> None:
        state = self.state
        scene_labels = [option.label for option in state.scene_options]
        changed, index = imgui.combo("Scene", state.current_scene_index, scene_labels)
        if changed:
            state.current_scene_index = int(index)
            state.current_variant_index = 0
            option = state.scene_options[state.current_scene_index]
            variant = option.variants[0]
            state.set_status(f"Switching to {option.label} ({variant})...")
            invoke_async(
                state.model_loop,
                lambda model_state, option=option, variant=variant: (
                    model_state.select_scene(option.path, variant)
                ),
            )
        option = state.scene_options[state.current_scene_index]
        changed, index = imgui.combo(
            "Variant", state.current_variant_index, list(option.variants)
        )
        if changed:
            state.current_variant_index = int(index)
            variant = option.variants[state.current_variant_index]
            state.set_status(f"Switching to {option.label} ({variant})...")
            invoke_async(
                state.model_loop,
                lambda model_state, variant=variant: (
                    model_state.select_variant(variant)
                ),
            )

    @staticmethod
    def _progress(imgui: Any, label: str, value: float) -> None:
        imgui.text(label)
        imgui.progress_bar(
            max(0.0, min(1.0, float(value))),
            imgui.ImVec2(-1.0, 0.0),
        )

    def _restart(self) -> None:
        prompt = self.state.prompt.strip()
        if not prompt:
            self.state.set_status("The current scene does not provide a prompt.")
            return
        self.state.set_status("Restart queued.")
        invoke_async(
            self.state.model_loop,
            lambda model_state, prompt=prompt: model_state.restart(prompt),
        )

    def _toggle_view(self) -> None:
        state = self.state
        views: tuple[ViewMode, ...] = ("rgb", "hdmap", "physx")
        state.set_view_mode(views[(views.index(state.view_mode) + 1) % len(views)])
        invoke_async(
            state.model_loop,
            lambda model_state, view=state.view_mode: model_state.set_view_mode(view),
        )

    def _ensure_sprites(self) -> None:
        if self.state.sprites:
            return
        for name in (
            "steering_wheel",
            "throttle_pressed",
            "throttle_unpressed",
            "brake_pressed",
            "brake_unpressed",
        ):
            size = (138, 138) if name == "steering_wheel" else (64, 138)
            self.state.sprites[name] = _load_control_sprite(name, size)

    def _wheel_pixels(self, steering: float) -> np.ndarray[Any, np.dtype[np.uint8]]:
        state = self.state
        angle = int(round(max(-90.0, min(90.0, steering * 180.0))))
        if state.wheel_cache_pixels is None or state.wheel_cache_angle != angle:
            wheel = Image.fromarray(state.sprites["steering_wheel"], mode="RGBA")
            rotated = wheel.rotate(angle, resample=Image.Resampling.BICUBIC)
            state.wheel_cache_pixels = np.asarray(rotated, dtype=np.uint8)
            state.wheel_cache_angle = angle
        return state.wheel_cache_pixels


class InteractiveDriveSession(ISession):
    """Register the separate Interactive Drive model and HUD loops."""

    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        config: InteractiveDriveConfig,
        desc: SessionDesc,
        scene_loader: SceneLoader,
        title: str,
        scene_options: tuple[InteractiveDriveSceneOption, ...],
    ) -> None:
        self._backend_factory = backend_factory
        self._config = config
        self._desc = desc
        self._scene_loader = scene_loader
        self._title = title
        self._scene_options = scene_options

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        model_state = InteractiveDriveModelState(
            backend_factory=self._backend_factory,
            config=self._config,
            desc=self._desc,
            scene_loader=self._scene_loader,
            view_mode=self._config.view_mode,
            postprocess_enabled=self._config.app.postprocess.is_enabled(),
        )
        model_loop = self.register_model_loop(
            InteractiveDriveModelLoop, state=model_state
        )
        if self._config.no_ui:
            return
        current_scene_index = next(
            (
                index
                for index, option in enumerate(self._scene_options)
                if _scene_key(option.path) == _scene_key(self._config.app.scene_path)
            ),
            0,
        )
        current_option = self._scene_options[current_scene_index]
        current_variant_index = next(
            (
                index
                for index, variant in enumerate(current_option.variants)
                if variant == self._config.app.variant
            ),
            0,
        )
        ui_loop = self.register_ui_loop(
            InteractiveDriveUILoop,
            state=InteractiveDriveUIState(
                model_loop=model_loop,
                title=self._title,
                prompt=self._config.app.prompt_override or "Drive through the scene.",
                scene_options=self._scene_options,
                current_scene_index=current_scene_index,
                current_variant_index=current_variant_index,
                view_mode=self._config.view_mode,
                viewport_width=self._desc.video_width,
                viewport_height=self._desc.video_height,
                postprocess_enabled=self._config.app.postprocess.is_enabled(),
                show_postprocess_toggle=bool(self._config.app.postprocess.preset),
            ),
            width=self._desc.video_width,
            height=self._desc.video_height,
        )
        model_state.ui_loop = ui_loop


class InteractiveDriveApplication(_InteractiveDriveApplicationBase):
    """Create the distinct long-running Interactive Drive demo."""

    def __init__(
        self,
        *,
        scene_loader: SceneLoader = load_scene_bundle,
        scene_options: Sequence[InteractiveDriveSceneOption] = (),
        defaults: InteractiveDriveApplicationDefaults | None = None,
    ) -> None:
        super().__init__(
            defaults=defaults
            or InteractiveDriveApplicationDefaults(
                title="Interactive Drive",
                slug="interactive-drive",
                total_blocks=0,
            ),
            scene_loader=scene_loader,
        )
        self._interactive_scene_options = tuple(scene_options)

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse common driving options and enable BEV output for the HUD."""
        super().init(commandline_args)
        assert self._config is not None
        self._config = InteractiveDriveConfig(
            app=self._config.app,
            total_blocks=self._config.total_blocks,
            view_mode=self._config.view_mode,
            no_ui=self._config.no_ui,
        )
        if not self._config.no_ui:
            self._config = replace(
                self._config,
                app=replace(
                    self._config.app,
                    bev=replace(
                        self._config.app.bev,
                        enabled=True,
                        show_ego_car=True,
                    ),
                ),
            )
        if not self._interactive_scene_options:
            self._interactive_scene_options = _discover_scene_options(
                self._config.app.scene_path
            )
        selected_variant = _variant_from_scene_path(self._config.app.scene_path)
        if self._config.app.variant == "default" and selected_variant != "default":
            selected_key = _scene_key(self._config.app.scene_path)
            option = next(
                (
                    candidate
                    for candidate in self._interactive_scene_options
                    if _scene_key(candidate.path) == selected_key
                ),
                None,
            )
            if option is not None and selected_variant in option.variants:
                self._config = replace(
                    self._config,
                    app=replace(
                        self._config.app,
                        scene_path=option.path,
                        variant=selected_variant,
                    ),
                )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create a session that owns the Interactive Drive HUD loop."""
        if self._config is None:
            raise RuntimeError("init() must run before create_session().")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Interactive Drive requires tchw output.")
        options = self._interactive_scene_options
        if not options:
            options = (
                InteractiveDriveSceneOption(
                    path=self._config.app.scene_path,
                    label=self._config.app.scene_path.stem,
                    variants=(self._config.app.variant,),
                ),
            )
        elif all(
            _scene_key(option.path) != _scene_key(self._config.app.scene_path)
            for option in options
        ):
            options = (
                *options,
                InteractiveDriveSceneOption(
                    path=self._config.app.scene_path,
                    label=self._config.app.scene_path.stem,
                    variants=(self._config.app.variant,),
                ),
            )
        return InteractiveDriveSession(
            backend_factory=self._backend_factory,
            config=self._config,
            desc=session_desc,
            scene_loader=self._scene_loader,
            title=self._title,
            scene_options=options,
        )


def _load_control_sprite(
    name: str, size: tuple[int, int]
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Load one bundled wheel/pedal sprite as RGBA pixels."""
    asset = (
        resources.files("interactive_drive")
        .joinpath("assets")
        .joinpath("wheel_and_pedals")
        .joinpath(f"{name}.png")
    )
    with asset.open("rb") as stream, Image.open(stream) as image:
        return np.asarray(
            image.convert("RGBA").resize(size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )


def _discover_scene_options(
    scene_path: Path,
) -> tuple[InteractiveDriveSceneOption, ...]:
    """Discover scene and weather-variant archives next to the selected scene."""
    paths = set(scene_path.parent.glob("*.usdz"))
    paths.add(scene_path)
    grouped: dict[str, dict[str, Path]] = {}
    for path in paths:
        grouped.setdefault(_scene_key(path), {})[_variant_from_scene_path(path)] = path
    options: list[InteractiveDriveSceneOption] = []
    for key, variants in sorted(grouped.items()):
        ordered_variants = tuple(
            sorted(
                variants,
                key=lambda value: (
                    {"default": 0, "snow": 1, "rain": 2}.get(value, 3),
                    value,
                ),
            )
        )
        base_path = variants.get("default", variants[ordered_variants[0]])
        label = key[:8] if _SCENE_STEM.match(base_path.stem) else base_path.stem
        options.append(
            InteractiveDriveSceneOption(
                path=base_path,
                label=label,
                variants=ordered_variants,
            )
        )
    return tuple(options)


def _scene_key(path: Path) -> str:
    """Return a grouping key shared by a scene's weather archives."""
    match = _SCENE_STEM.match(Path(path).stem)
    return (
        match.group("uuid").lower() if match is not None else str(Path(path).resolve())
    )


def _variant_from_scene_path(path: Path) -> str:
    """Return the weather suffix encoded in a canonical scene filename."""
    match = _SCENE_STEM.match(Path(path).stem)
    if match is None:
        return "default"
    return match.group("variant") or "default"


__all__ = [
    "InteractiveDriveApplication",
    "InteractiveDriveSceneOption",
    "InteractiveDriveSession",
    "InteractiveDriveUILoop",
    "InteractiveDriveUIState",
]
