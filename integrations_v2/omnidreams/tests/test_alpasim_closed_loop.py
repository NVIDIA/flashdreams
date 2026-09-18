# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the source-mounted AlpaSim closed-loop launcher."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from omnidreams.impl.tools.alpasim_closed_loop import (
    _parser,
    _resolve_and_validate,
    build_docker_commands,
    build_wizard_command,
)

pytestmark = pytest.mark.ci_cpu


def _parse_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    flashdreams_repo = tmp_path / "flashdreams"
    (flashdreams_repo / "docker").mkdir(parents=True)
    (flashdreams_repo / "docker" / "Dockerfile.alpasim").write_text("")
    (flashdreams_repo / "flashdreams").mkdir()
    (flashdreams_repo / "apps" / "interactive_drive").mkdir(parents=True)
    (flashdreams_repo / "integrations_v2" / "omnidreams").mkdir(parents=True)

    alpasim_repo = tmp_path / "alpasim"
    (alpasim_repo / "src" / "wizard").mkdir(parents=True)

    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    (scene_dir / "scene.usdz").write_bytes(b"fixture")

    args = _parser().parse_args(
        [
            "--scene-dir",
            str(scene_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--flashdreams-repo",
            str(flashdreams_repo),
            "--alpasim-repo",
            str(alpasim_repo),
            "--hf-cache",
            str(tmp_path / "hf"),
            "--torch-cache",
            str(tmp_path / "torch"),
            "--flashdreams-cache",
            str(tmp_path / "fd-cache"),
            *extra,
        ]
    )
    _resolve_and_validate(args)
    return args


def test_builds_development_image_and_source_mounted_wizard_command(
    tmp_path: Path,
) -> None:
    args = _parse_args(tmp_path)

    docker_commands = build_docker_commands(args)
    assert len(docker_commands) == 1
    assert docker_commands[0][:5] == [
        "docker",
        "build",
        "--target",
        "development",
        "--build-arg",
    ]
    assert "FLASHDREAMS_BASE_IMAGE=flashdreams:local" in docker_commands[0]

    wizard = build_wizard_command(args)
    assert "deploy=managed_flashdreams" in wizard
    assert "scenes.scene_ids=null" in wizard
    assert "runtime.simulation_config.n_sim_steps=80" in wizard
    assert "services.renderer.gpus=[6]" in wizard
    assert "services.driver.gpus=[7]" in wizard
    assert "services.physics.gpus=[7]" in wizard
    assert "services.trafficsim.gpus=[7]" in wizard
    assert "eval.video.render_every_nth_frame=1" in wizard

    volume_override = next(
        item for item in wizard if item.startswith("services.renderer.volumes=")
    )
    assert f"{args.flashdreams_repo / 'flashdreams'}:" in volume_override
    assert f"{args.flashdreams_repo / 'apps' / 'interactive_drive'}:" in (
        volume_override
    )
    assert f"{args.flashdreams_repo / 'integrations_v2' / 'omnidreams'}:" in (
        volume_override
    )
    assert f"{args.flashdreams_repo}:/workspace/flashdreams:ro" not in volume_override

    command_override = next(
        item for item in wizard if item.startswith("services.renderer.command=")
    )
    assert (
        "/workspace/.venv/bin/python -m omnidreams.impl.grpc.server" in command_override
    )


def test_scene_selection_dry_run_and_base_build_overrides(tmp_path: Path) -> None:
    args = _parse_args(
        tmp_path,
        "--scene-id",
        "clipgt-a",
        "--scene-id",
        "clipgt-b",
        "--dry-run",
        "--build-base",
        "--limit",
        "2",
    )

    docker_commands = build_docker_commands(args)
    assert len(docker_commands) == 2
    assert docker_commands[0] == [
        "docker",
        "build",
        "-t",
        "flashdreams:local",
        "-f",
        "docker/Dockerfile",
        ".",
    ]

    wizard = build_wizard_command(args)
    assert 'scenes.scene_ids=["clipgt-a","clipgt-b"]' in wizard
    assert "scenes.test_suite_id=null" in wizard
    assert "scenes.test_suite_id=local" not in wizard
    assert "scenes.limit_to_first_n=2" in wizard
    assert "wizard.dry_run=true" in wizard


def test_rejects_build_only_when_all_builds_are_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enabled image build"):
        _parse_args(tmp_path, "--build-only", "--skip-build")


def test_skip_build_does_not_require_alpasim_dockerfile(tmp_path: Path) -> None:
    args = _parse_args(tmp_path, "--skip-build")
    (args.flashdreams_repo / "docker" / "Dockerfile.alpasim").unlink()

    _resolve_and_validate(args)


def test_image_build_requires_alpasim_dockerfile(tmp_path: Path) -> None:
    args = _parse_args(tmp_path)
    dockerfile = args.flashdreams_repo / "docker" / "Dockerfile.alpasim"
    dockerfile.unlink()

    with pytest.raises(ValueError, match=f"Missing AlpaSim Dockerfile: {dockerfile}"):
        _resolve_and_validate(args)


def test_rejects_output_path_that_is_not_a_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "output"
    output_path.write_text("not a directory")

    with pytest.raises(ValueError, match="Output path is not a directory"):
        _parse_args(tmp_path)
