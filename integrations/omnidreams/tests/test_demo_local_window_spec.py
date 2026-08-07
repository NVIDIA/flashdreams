# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``DemoSpec`` translation for the OmniDreams local-window demo.

The spec reaches the window by way of ``interactive-drive``'s argument parser,
so a field the translator forgets to emit is silently ignored rather than
rejected. These tests pin the round trip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from flashdreams.runtime.demo import (
    DemoSpec,
    LocalWindowOutputSpec,
    Mp4OutputSpec,
)
from omnidreams.demo.local_window import (
    InteractiveDriveApp,
    OmnidreamsLocalWindowScenario,
    PlugCompatibleInteractiveDriveApp,
    build_interactive_drive_app,
    build_omnidreams_local_window_app,
)

pytestmark = pytest.mark.ci_cpu


def _spec(
    *,
    output: LocalWindowOutputSpec | Mp4OutputSpec | None = None,
    scenario: OmnidreamsLocalWindowScenario | None = None,
) -> DemoSpec:
    return DemoSpec(
        model_id="omnidreams",
        input_mode="keyboard-driving",
        output=output if output is not None else LocalWindowOutputSpec(),
        scenario=scenario,
    )


def _parsed(spec: DemoSpec) -> argparse.Namespace:
    """Run the emitted arguments back through the parser that consumes them."""
    from omnidreams.interactive_drive.demo import build_parser

    argv = build_interactive_drive_app(spec).argv
    return build_parser().parse_args(list(argv))


## Output spec


def test_window_geometry_reaches_the_parser() -> None:
    """Emitting nothing for these left ``--window-width`` silently inert."""
    args = _parsed(
        _spec(output=LocalWindowOutputSpec(width=1280, height=720, title="probe"))
    )

    assert (args.window_width, args.window_height) == (1280, 720)
    assert args.window_title == "probe"


def test_every_output_spec_field_is_accounted_for() -> None:
    """Fails when a field is added to the spec but not handled here.

    A forgotten field is invisible at runtime -- the spec still validates and
    the value is simply ignored -- so the dataclass is checked directly.
    """
    import dataclasses

    handled = {
        "mode",  # union discriminator, not a setting
        "show_hud",  # emitted as the --no-hud switch
        "width",
        "height",
        "title",
    }
    fields = {f.name for f in dataclasses.fields(LocalWindowOutputSpec)}

    assert fields == handled, (
        f"unhandled LocalWindowOutputSpec fields: {sorted(fields - handled)}"
    )


def test_geometry_flags_are_always_emitted() -> None:
    emitted = build_interactive_drive_app(_spec()).argv

    assert "--window-width" in emitted
    assert "--window-height" in emitted
    assert "--window-title" in emitted


def test_hud_can_be_turned_off() -> None:
    args = _parsed(_spec(output=LocalWindowOutputSpec(show_hud=False)))

    assert args.no_hud


def test_hud_is_on_by_default() -> None:
    args = _parsed(_spec())

    assert not args.no_hud


def test_non_local_window_output_is_rejected() -> None:
    spec = _spec(output=Mp4OutputSpec(path="out.mp4", fps=30))

    with pytest.raises(TypeError, match="LocalWindowOutputSpec"):
        build_interactive_drive_app(spec)


## Scenario


def test_the_shared_presenter_is_selected_by_the_scenario() -> None:
    args = _parsed(
        _spec(scenario=OmnidreamsLocalWindowScenario(presenter_backend="local-window"))
    )

    assert args.presenter_backend == "local-window"


def test_shared_backend_selects_plug_compatible_app() -> None:
    adapter = object()
    spec = _spec(
        scenario=OmnidreamsLocalWindowScenario(presenter_backend="local-window")
    )

    app = build_omnidreams_local_window_app(spec=spec, adapter=adapter)

    assert isinstance(app, PlugCompatibleInteractiveDriveApp)
    assert app.adapter is adapter


def test_legacy_backend_remains_an_explicit_compatibility_fallback() -> None:
    app = build_omnidreams_local_window_app(
        spec=_spec(scenario=OmnidreamsLocalWindowScenario(presenter_backend="legacy")),
        adapter=object(),
    )

    assert isinstance(app, InteractiveDriveApp)


def test_the_plug_compatible_presenter_is_the_default() -> None:
    args = _parsed(_spec())

    assert args.presenter_backend == "local-window"


def test_scene_selection_reaches_the_parser() -> None:
    args = _parsed(
        _spec(
            scenario=OmnidreamsLocalWindowScenario(
                scene=Path("/scenes/town.usd"),
                variant="night",
            )
        )
    )

    assert str(args.scene) == "/scenes/town.usd"
    assert args.variant == "night"


def test_manifest_selects_the_world_model_backend() -> None:
    args = _parsed(
        _spec(
            scenario=OmnidreamsLocalWindowScenario(
                manifest="configs/omnidreams.yaml",
            )
        )
    )

    assert args.backend == "omnidreams"
    assert str(args.manifest) == "configs/omnidreams.yaml"


def test_a_wrongly_typed_scenario_is_rejected() -> None:
    spec = DemoSpec(
        model_id="omnidreams",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
        scenario={"scene": "town"},
    )

    with pytest.raises(TypeError, match="OmnidreamsLocalWindowScenario"):
        build_interactive_drive_app(spec)


def test_extra_args_pass_through_for_unmodelled_flags() -> None:
    args = _parsed(
        _spec(
            scenario=OmnidreamsLocalWindowScenario(
                extra_args=("--stop-after-chunks", "3")
            )
        )
    )

    assert args.stop_after_chunks == 3
