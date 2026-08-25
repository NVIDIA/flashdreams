# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest
from crazy_robotaxi import runtime_cli as cli
from crazy_robotaxi.runtime_cli import build_parser
from omnidreams_game_engine.cli import build_parser as build_engine_parser
from omnidreams_game_engine.cli_args import arg_was_explicit

pytestmark = pytest.mark.ci_cpu

_MAP_ARGS = ["--map", "city.robotaxi.yaml"]


@pytest.mark.parametrize("parser_factory", [build_parser, build_engine_parser])
def test_map_flag_sets_internal_scene_path(
    parser_factory: Callable[[], argparse.ArgumentParser],
) -> None:
    args = parser_factory().parse_args(_MAP_ARGS)

    assert args.scene == Path("city.robotaxi.yaml")


@pytest.mark.parametrize("parser_factory", [build_parser, build_engine_parser])
def test_force_map_recompile_flag(
    parser_factory: Callable[[], argparse.ArgumentParser],
) -> None:
    assert parser_factory().parse_args([]).force_map_recompile is False
    assert (
        parser_factory().parse_args(["--force-map-recompile"]).force_map_recompile
        is True
    )


def test_force_map_recompile_flag_is_forwarded_to_app_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RasterRenderBackend", lambda **_kwargs: object())

    config, _backend = cli.prepare_config_and_backend(
        build_parser().parse_args([*_MAP_ARGS, "--force-map-recompile"])
    )

    assert config.force_map_recompile is True


@pytest.mark.parametrize(
    "removed_flag",
    ["--scene", "--synthetic-scene", "--synthetic-initial-rgb", "--synthetic-prompt"],
)
@pytest.mark.parametrize("parser_factory", [build_parser, build_engine_parser])
def test_removed_map_input_flags_are_not_accepted(
    removed_flag: str, parser_factory: Callable[[], argparse.ArgumentParser]
) -> None:
    with pytest.raises(SystemExit):
        parser_factory().parse_args([removed_flag])


def test_offload_text_encoder_flag_defaults_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.offload_text_encoder is False


def test_offload_text_encoder_flag_enables() -> None:
    args = build_parser().parse_args(["--offload-text-encoder"])

    assert args.offload_text_encoder is True


def test_game_mode_defaults_to_taxi_and_accepts_named_modes() -> None:
    assert build_parser().parse_args([]).game_mode == "taxi"
    assert build_parser().parse_args(["--game-mode", "race"]).game_mode == "race"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--game-mode"])


def test_visual_flare_override_defaults_disabled() -> None:
    assert build_parser().parse_args([]).disable_visual_flare is False
    assert (
        build_parser().parse_args(["--disable-visual-flare"]).disable_visual_flare
        is True
    )


@pytest.mark.parametrize(
    ("argv", "visual_flare_enabled"),
    [
        ([], False),
        (["--game-mode", "race"], False),
        (["--game-mode", "race", "--disable-visual-flare"], False),
    ],
)
def test_named_game_modes_enable_game_physics(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    visual_flare_enabled: bool,
) -> None:
    monkeypatch.setattr(cli, "RasterRenderBackend", lambda **_k: object())

    config, _backend = cli.prepare_config_and_backend(
        build_parser().parse_args([*_MAP_ARGS, *argv])
    )

    assert config.vehicle.speed_limit_enabled is True
    assert config.vehicle.actor_collision_enabled is True
    assert config.vehicle.static_collision_enabled is True
    assert config.visual_flare_enabled is visual_flare_enabled


@pytest.mark.parametrize(
    ("argv", "expected_synchronization"),
    [([], True), (["--game-mode", "race"], True)],
)
def test_taxi_game_selects_frame_synchronous_bev(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_synchronization: bool,
) -> None:
    backend_kwargs: dict[str, object] = {}

    def build_backend(**kwargs: object) -> object:
        backend_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "RasterRenderBackend", build_backend)

    cli.prepare_config_and_backend(build_parser().parse_args([*_MAP_ARGS, *argv]))

    assert backend_kwargs["synchronize_bev_with_rgb"] is expected_synchronization


def test_taxi_alignment_diagnostics_accepts_output_directory() -> None:
    args = build_parser().parse_args(["--taxi-alignment-diagnostics", "diagnostics"])

    assert args.taxi_alignment_diagnostics == Path("diagnostics")


@pytest.mark.parametrize(
    ("diagnostic_args", "expected_enabled"),
    [([], False), (["--taxi-alignment-diagnostics", "diagnostics"], True)],
)
def test_taxi_alignment_diagnostics_gate_motion_conformance(
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_args: list[str],
    expected_enabled: bool,
) -> None:
    backend_kwargs: dict[str, object] = {}

    def build_backend(**kwargs: object) -> object:
        backend_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "WorldModelRenderBackend", build_backend)
    args = build_parser().parse_args(
        [
            *_MAP_ARGS,
            "--backend",
            "omnidreams",
            "--manifest",
            "example_world_model_synthetic.yaml",
            *diagnostic_args,
        ]
    )

    cli.prepare_config_and_backend(args)

    assert backend_kwargs["motion_conformance_diagnostics_enabled"] is expected_enabled


def test_postprocess_preset_defaults_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.postprocess_preset == ""


def test_postprocess_preset_accepts_rtx_super_resolution() -> None:
    args = build_parser().parse_args(["--postprocess-preset", "rtx-super-resolution"])

    assert args.postprocess_preset == "rtx-super-resolution"


def test_parser_records_explicit_arg_destinations() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "example_world_model_perf.yaml",
            "--offload-text-encoder",
            "--no-bev",
        ]
    )

    assert arg_was_explicit(args, "manifest")
    assert arg_was_explicit(args, "offload_text_encoder")
    assert arg_was_explicit(args, "bev")
    assert not arg_was_explicit(args, "camera")
