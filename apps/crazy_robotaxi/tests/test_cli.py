# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import pytest
from crazy_robotaxi import runtime_cli as cli
from crazy_robotaxi.runtime_cli import build_parser
from omnidreams_game_engine.cli_args import arg_was_explicit

pytestmark = pytest.mark.ci_cpu


def test_offload_text_encoder_flag_defaults_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.offload_text_encoder is False


def test_offload_text_encoder_flag_enables() -> None:
    args = build_parser().parse_args(["--offload-text-encoder"])

    assert args.offload_text_encoder is True


def test_game_mode_defaults_disabled_and_can_be_enabled() -> None:
    assert build_parser().parse_args([]).game_mode is False
    assert build_parser().parse_args(["--game-mode"]).game_mode is True


def test_visual_flare_override_defaults_disabled() -> None:
    assert build_parser().parse_args([]).disable_visual_flare is False
    assert (
        build_parser().parse_args(["--disable-visual-flare"]).disable_visual_flare
        is True
    )


@pytest.mark.parametrize(
    ("argv", "game_mode_enabled", "visual_flare_enabled"),
    [
        ([], False, False),
        (["--game-mode"], True, True),
        (["--game-mode", "--disable-visual-flare"], True, False),
    ],
)
def test_game_mode_controls_speed_limit_collisions_and_visual_flare(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    game_mode_enabled: bool,
    visual_flare_enabled: bool,
) -> None:
    monkeypatch.setattr(cli, "RasterRenderBackend", lambda **_k: object())

    config, _backend = cli.prepare_config_and_backend(build_parser().parse_args(argv))

    assert config.vehicle.speed_limit_enabled is game_mode_enabled
    assert config.vehicle.actor_collision_enabled is game_mode_enabled
    assert config.vehicle.static_collision_enabled is game_mode_enabled
    assert config.visual_flare_enabled is visual_flare_enabled


@pytest.mark.parametrize(
    ("argv", "expected_synchronization"),
    [([], True), (["--taxi-game"], True)],
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

    cli.prepare_config_and_backend(build_parser().parse_args(argv))

    assert backend_kwargs["synchronize_bev_with_rgb"] is expected_synchronization


def test_taxi_alignment_diagnostics_accepts_output_directory() -> None:
    args = build_parser().parse_args(
        ["--taxi-game", "--taxi-alignment-diagnostics", "diagnostics"]
    )

    assert args.taxi_alignment_diagnostics == Path("diagnostics")


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
