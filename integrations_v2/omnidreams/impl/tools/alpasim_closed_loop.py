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

"""Source-mounted FlashDreams and AlpaSim closed-loop launcher."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

_DEFAULT_PIPELINE_CONFIG = "omnidreams-perf"
"""OmniDreams runner used by AlpaSim's managed FlashDreams deployment."""

_CONTAINER_WORKSPACE = Path("/workspace/flashdreams")
"""Read-only workspace root assembled from the two source bind mounts."""


def _flashdreams_repo_from_module() -> Path:
    return Path(__file__).resolve().parents[4]


def _hydra_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _source_mounts(flashdreams_repo: Path) -> list[str]:
    return [
        f"{flashdreams_repo / 'flashdreams'}:{_CONTAINER_WORKSPACE / 'flashdreams'}:ro",
        (
            f"{flashdreams_repo / 'apps' / 'interactive_drive'}:"
            f"{_CONTAINER_WORKSPACE / 'apps' / 'interactive_drive'}:ro"
        ),
        (
            f"{flashdreams_repo / 'integrations_v2' / 'omnidreams'}:"
            f"{_CONTAINER_WORKSPACE / 'integrations_v2' / 'omnidreams'}:ro"
        ),
    ]


def _renderer_command(pipeline_config: str) -> list[str]:
    return [
        "/workspace/.venv/bin/python -m omnidreams.impl.grpc.server",
        f"--pipeline_config_name {pipeline_config}",
        "--host 0.0.0.0",
        "--port {port}",
    ]


def build_docker_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build Docker commands requested by the launcher arguments.

    Args:
        args: Parsed launcher arguments.

    Returns:
        Commands for the base image and source-mounted dependency image.
    """
    commands: list[list[str]] = []
    if args.build_base:
        commands.append(
            [
                "docker",
                "build",
                "-t",
                args.base_image,
                "-f",
                "docker/Dockerfile",
                ".",
            ]
        )
    if not args.skip_build:
        commands.append(
            [
                "docker",
                "build",
                "--target",
                "development",
                "--build-arg",
                f"FLASHDREAMS_BASE_IMAGE={args.base_image}",
                "-t",
                args.image,
                "-f",
                "docker/Dockerfile.alpasim",
                ".",
            ]
        )
    return commands


def build_wizard_command(args: argparse.Namespace) -> list[str]:
    """Build the AlpaSim wizard command for a managed closed loop.

    Args:
        args: Parsed launcher arguments with absolute repository and data paths.

    Returns:
        Command that starts the AlpaSim services and managed renderer.
    """
    container_pythonpath = ":".join(
        [
            str(_CONTAINER_WORKSPACE / "flashdreams"),
            str(_CONTAINER_WORKSPACE / "apps" / "interactive_drive"),
            str(_CONTAINER_WORKSPACE / "integrations_v2" / "omnidreams"),
            str(
                _CONTAINER_WORKSPACE
                / "integrations_v2"
                / "omnidreams"
                / "impl"
                / "ludus-renderer"
            ),
        ]
    )
    renderer_volumes = [
        *_source_mounts(args.flashdreams_repo),
        f"{args.hf_cache}:/home/flashdreams/.cache/huggingface",
        f"{args.torch_cache}:/home/flashdreams/.cache/torch",
        f"{args.flashdreams_cache}:/home/flashdreams/.cache/flashdreams",
    ]
    renderer_environment = [
        "HF_TOKEN",
        "HF_HOME=/home/flashdreams/.cache/huggingface",
        "TORCH_HOME=/home/flashdreams/.cache/torch",
        "TORCH_EXTENSIONS_DIR=/home/flashdreams/.cache/torch/extensions",
        "FLASHDREAMS_CACHE_DIR=/home/flashdreams/.cache/flashdreams",
        f"PYTHONPATH={container_pythonpath}",
    ]

    scene_ids = _hydra_list(args.scene_id) if args.scene_id else "null"
    test_suite_id = "null" if args.scene_id else "local"
    command = [
        "uv",
        "run",
        "--project",
        "src/wizard",
        "alpasim_wizard",
        "deploy=managed_flashdreams",
        "topology=1gpu",
        f"driver={args.driver}",
        f"+chunking={args.chunking}",
        f"wizard.log_dir={args.output_dir}",
        f"wizard.dry_run={'true' if args.dry_run else 'false'}",
        f"scenes.local_usdz_dir={args.scene_dir}",
        f"scenes.scene_ids={scene_ids}",
        f"scenes.test_suite_id={test_suite_id}",
        f"scenes.limit_to_first_n={args.limit}",
        f"runtime.simulation_config.n_sim_steps={args.n_sim_steps}",
        "runtime.nr_workers=1",
        "runtime.endpoints.renderer.n_concurrent_rollouts=1",
        "runtime.endpoints.driver.n_concurrent_rollouts=1",
        "runtime.endpoints.physics.n_concurrent_rollouts=1",
        "runtime.endpoints.controller.n_concurrent_rollouts=1",
        "eval.video.render_video=true",
        f"eval.video.render_every_nth_frame={args.video_frame_stride}",
        f"services.renderer.gpus=[{args.flashdreams_gpu}]",
        f"services.driver.gpus=[{args.alpasim_gpu}]",
        f"services.physics.gpus=[{args.alpasim_gpu}]",
        f"services.trafficsim.gpus=[{args.alpasim_gpu}]",
        f"services.renderer.image={args.image}",
        f"services.renderer.workdir={_CONTAINER_WORKSPACE}",
        f"services.renderer.volumes={_hydra_list(renderer_volumes)}",
        f"services.renderer.environments={_hydra_list(renderer_environment)}",
        (
            "services.renderer.command="
            f"{_hydra_list(_renderer_command(args.pipeline_config))}"
        ),
    ]
    return command


def _parser() -> argparse.ArgumentParser:
    repo = _flashdreams_repo_from_module()
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-mounted FlashDreams image and run an AlpaSim "
            "closed loop over local USDZ scenes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flashdreams-repo", type=Path, default=repo)
    parser.add_argument("--alpasim-repo", type=Path, default=repo.parent / "alpasim")
    parser.add_argument("--n-sim-steps", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--flashdreams-gpu", type=int, default=6)
    parser.add_argument("--alpasim-gpu", type=int, default=7)
    parser.add_argument("--video-frame-stride", type=int, default=1)
    parser.add_argument("--driver", default="vavam_video_model")
    parser.add_argument("--chunking", default="8frame")
    parser.add_argument("--pipeline-config", default=_DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--base-image", default="flashdreams:local")
    parser.add_argument("--image", default="flashdreams-alpasim-dev:local")
    parser.add_argument(
        "--hf-cache", type=Path, default=Path.home() / ".cache" / "huggingface"
    )
    parser.add_argument(
        "--torch-cache", type=Path, default=Path.home() / ".cache" / "torch"
    )
    parser.add_argument(
        "--flashdreams-cache",
        type=Path,
        default=Path.home() / ".cache" / "flashdreams",
    )
    parser.add_argument("--build-base", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate the AlpaSim deployment without starting services.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print all commands without executing them.",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser


def _resolve_and_validate(args: argparse.Namespace) -> None:
    args.flashdreams_repo = args.flashdreams_repo.expanduser().resolve()
    args.alpasim_repo = args.alpasim_repo.expanduser().resolve()
    args.scene_dir = args.scene_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.hf_cache = args.hf_cache.expanduser().resolve()
    args.torch_cache = args.torch_cache.expanduser().resolve()
    args.flashdreams_cache = args.flashdreams_cache.expanduser().resolve()

    if not (args.flashdreams_repo / "docker" / "Dockerfile.alpasim").is_file():
        raise ValueError(f"Not a FlashDreams checkout: {args.flashdreams_repo}")
    for source_path in (
        args.flashdreams_repo / "flashdreams",
        args.flashdreams_repo / "apps" / "interactive_drive",
        args.flashdreams_repo / "integrations_v2" / "omnidreams",
    ):
        if not source_path.is_dir():
            raise ValueError(f"Missing FlashDreams source directory: {source_path}")
    if not (args.alpasim_repo / "src" / "wizard").is_dir():
        raise ValueError(f"Not an AlpaSim checkout: {args.alpasim_repo}")
    if not args.scene_dir.is_dir():
        raise ValueError(f"USDZ scene directory does not exist: {args.scene_dir}")
    if not any(args.scene_dir.rglob("*.usdz")):
        raise ValueError(f"No USDZ scenes found under: {args.scene_dir}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()) and not args.allow_existing_output:
            raise ValueError(
                f"Output directory is not empty: {args.output_dir}; "
                "pass --allow-existing-output to reuse it"
            )
    if args.n_sim_steps < 1:
        raise ValueError("--n-sim-steps must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.flashdreams_gpu < 0 or args.alpasim_gpu < 0:
        raise ValueError("GPU indices must be non-negative")
    if args.video_frame_stride < 1:
        raise ValueError("--video-frame-stride must be at least 1")
    if args.build_only and args.skip_build and not args.build_base:
        raise ValueError("--build-only needs an enabled image build")


def _run(command: Sequence[str], *, cwd: Path, print_only: bool) -> None:
    print(f"[{cwd}] {shlex.join(command)}", flush=True)
    if print_only:
        return
    executable = command[0]
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"Required executable is not available: {executable}")
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the renderer environment and launch the AlpaSim closed loop.

    Args:
        argv: Optional command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _parser().parse_args(argv)
    _resolve_and_validate(args)

    if not args.print_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for cache_dir in (args.hf_cache, args.torch_cache, args.flashdreams_cache):
            cache_dir.mkdir(parents=True, exist_ok=True)

    for command in build_docker_commands(args):
        _run(command, cwd=args.flashdreams_repo, print_only=args.print_only)

    if args.build_only:
        return
    _run(
        build_wizard_command(args),
        cwd=args.alpasim_repo,
        print_only=args.print_only,
    )


if __name__ == "__main__":
    main()
