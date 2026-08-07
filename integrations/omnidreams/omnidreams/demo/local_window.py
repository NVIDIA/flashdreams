# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OmniDreams local-window demo: shared ``DemoSpec`` onto interactive-drive."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flashdreams.runtime.demo import DemoSpec, LocalWindowOutputSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsLocalWindowScenario:
    """Scene and control settings for a windowed OmniDreams drive.

    Mirrors the flags ``interactive-drive`` already accepts. Anything left at
    its default is resolved by the demo's own discovery, so a bare
    ``omnidreams-demo local-window`` behaves like a bare ``interactive-drive``.
    """

    scene_dir: Path | None = None
    scene: Path | None = None
    synthetic_scene: bool = False
    synthetic_initial_rgb: Path | None = None
    manifest: str | Path | None = None
    camera_name: str | None = None
    variant: str | None = None
    postprocess_preset: str = ""
    presenter_backend: str = "local-window"
    """``local-window`` swaps the driving HUD for the shared presenter."""

    auto_start: bool = False
    preload_scenes: bool = False
    no_wheel: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    """Escape hatch for interactive-drive flags this scenario does not model."""


@dataclass(slots=True)
class InteractiveDriveApp:
    """Adapter between the shared launcher's ``run`` and the demo's ``main``."""

    argv: tuple[str, ...]

    def run(self) -> None:
        """Run interactive-drive to completion with the resolved arguments."""
        from omnidreams.interactive_drive.demo import build_parser, run_parsed_args

        run_parsed_args(build_parser().parse_args(list(self.argv)))


@dataclass(slots=True)
class PlugCompatibleInteractiveDriveApp:
    """OmniDreams session runner using the app-owned standard runtime path."""

    spec: DemoSpec
    adapter: Any

    def run(self) -> None:
        from interactive_drive_app import InteractiveDriveApplication

        sessions = self.adapter.list_sessions(self.spec)
        if not sessions:
            raise RuntimeError("OmniDreams adapter returned no driving sessions.")
        index = 0
        app = InteractiveDriveApplication(
            adapter=self.adapter,
            initial_spec=sessions[index],
        )
        try:
            while True:
                outcome = app.run_session(
                    spec=sessions[index],
                    session_id=f"session-{index}",
                )
                if outcome.action == "reset":
                    continue
                if outcome.action == "next":
                    index = (index + 1) % len(sessions)
                    continue
                if outcome.action == "previous":
                    index = (index - 1) % len(sessions)
                    continue
                break
        finally:
            app.close()


def build_omnidreams_local_window_app(
    *,
    spec: DemoSpec,
    adapter: Any,
) -> InteractiveDriveApp | PlugCompatibleInteractiveDriveApp:
    """Select the standard app route or the legacy compatibility app."""
    scenario = spec.scenario or OmnidreamsLocalWindowScenario()
    if not isinstance(scenario, OmnidreamsLocalWindowScenario):
        raise TypeError(
            "OmniDreams local-window scenario must be an "
            f"OmnidreamsLocalWindowScenario, got {type(scenario).__name__}."
        )
    if scenario.presenter_backend == "local-window":
        return PlugCompatibleInteractiveDriveApp(spec=spec, adapter=adapter)
    return build_interactive_drive_app(spec)


def build_interactive_drive_app(spec: DemoSpec) -> InteractiveDriveApp:
    """Translate ``spec`` into interactive-drive arguments.

    Raises:
        ValueError: ``spec`` does not carry local-window output.
    """
    output = spec.output
    if not isinstance(output, LocalWindowOutputSpec):
        raise TypeError(
            "OmniDreams local-window output requires LocalWindowOutputSpec."
        )
    scenario = spec.scenario or OmnidreamsLocalWindowScenario()
    if not isinstance(scenario, OmnidreamsLocalWindowScenario):
        raise TypeError(
            "OmniDreams local-window scenario must be an "
            f"OmnidreamsLocalWindowScenario, got {type(scenario).__name__}."
        )

    argv: list[str] = []
    if not output.show_hud:
        argv.append("--no-hud")
    argv.extend(("--window-width", str(output.width)))
    argv.extend(("--window-height", str(output.height)))
    argv.extend(("--window-title", output.title))
    _append_option(argv, "--scene-dir", scenario.scene_dir)
    _append_option(argv, "--scene", scenario.scene)
    _append_option(argv, "--manifest", scenario.manifest)
    _append_option(argv, "--camera", scenario.camera_name)
    _append_option(argv, "--variant", scenario.variant)
    _append_option(argv, "--synthetic-initial-rgb", scenario.synthetic_initial_rgb)
    if scenario.postprocess_preset:
        argv.extend(("--postprocess-preset", scenario.postprocess_preset))
    if scenario.presenter_backend != "legacy":
        argv.extend(("--presenter-backend", scenario.presenter_backend))
    if scenario.synthetic_scene:
        argv.append("--synthetic-scene")
    if scenario.auto_start:
        argv.append("--auto-start")
    if scenario.preload_scenes:
        argv.append("--preload-scenes")
    if scenario.no_wheel:
        argv.append("--no-wheel")
    argv.extend(scenario.extra_args)
    return InteractiveDriveApp(argv=tuple(argv))


def _append_option(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend((flag, str(value)))


__all__ = [
    "InteractiveDriveApp",
    "OmnidreamsLocalWindowScenario",
    "PlugCompatibleInteractiveDriveApp",
    "build_interactive_drive_app",
    "build_omnidreams_local_window_app",
]
