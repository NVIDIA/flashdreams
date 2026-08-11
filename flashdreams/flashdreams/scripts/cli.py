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

"""``flashdreams-run`` CLI: pick a runner, override any field, generate.

One hyphenated console script fronts a tyro subcommand union built
from the runner registry; each subcommand uses its
:class:`RunnerConfig` literal as ``defaults=`` and exposes every
nested field as a CLI flag.

Usage::

    flashdreams-run --help                            # list every runner
    flashdreams-run wan21-t2v-1.3b-480p --help        # show overridable fields
    flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."
    flashdreams-run wan21-i2v-14b-480p --prompt "..." --image-path frame.png
    flashdreams-run --no-instantiate template-offline # resolve config only
    flashdreams-run wan21-t2v-1.3b-480p --postprocess.preset flashvsr-v1.1-sparse-2.0
    flashdreams-run lingbot-world-fast webrtc --host 0.0.0.0 --port 8080
    flashdreams-run omnidreams local-window

    # Multi-GPU via context-parallelism (integration transformers auto-detect
    # CP size from the launcher's WORLD group). ``--no-python`` tells
    # torchrun to execvp the console script directly instead of wrapping
    # it in ``python <script>``:
    torchrun --nproc_per_node=N --no-python flashdreams-run <slug> ...
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import tyro

from flashdreams.configs.runner_configs import _annotated_base_runner_union, all_runners
from flashdreams.core.distributed import shutdown as shutdown_distributed
from flashdreams.core.io.disk import disk_space_error_from_exception
from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import (
    LaunchMode,
    LaunchOptions,
    available_launch_modes,
    resolve_launch,
)
from flashdreams.serving.launch_manifest import (
    FlashDreamsLaunchManifest,
    load_launch_manifest,
)

_POSITIONAL_MODES = frozenset({"run", "mp4", "null", "webrtc", "local-window"})


def main(
    config: RunnerConfig,
    no_instantiate: bool = False,
    *,
    mode: LaunchMode = "run",
    host: str | None = None,
    port: int | None = None,
    legacy_world_manifest: Path | None = None,
    prefer_sw_encoder: bool = False,
    launch_manifest: FlashDreamsLaunchManifest | None = None,
) -> None:
    """Print the resolved config and (by default) run the runner.

    Under ``torchrun`` only local-rank 0 prints; every rank holds the
    same resolved config.
    """
    resolved_launch = None
    launch_options = LaunchOptions(
        host=host,
        port=port,
        prefer_sw_encoder=prefer_sw_encoder,
        legacy_world_manifest=legacy_world_manifest,
        launch_manifest=None if launch_manifest is None else launch_manifest.path,
        scenario={} if launch_manifest is None else launch_manifest.scenario,
        output={} if launch_manifest is None else launch_manifest.output,
    )
    if mode != "run":
        resolved_launch = resolve_launch(
            config,
            mode=mode,
            options=launch_options,
        )

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Resolved config for {config.runner_name!r}:")
        print(config)
        print(
            f"Available modes: {', '.join(available_launch_modes(config, launch_options))}"
        )
        if launch_manifest is not None:
            print(f"Launch manifest: {launch_manifest.path}")
            print(f"Launch mode: {launch_manifest.mode}")
            print(f"Scenario: {dict(launch_manifest.scenario)}")
            print(f"Output settings: {dict(launch_manifest.output)}")
        if resolved_launch is not None:
            print(f"Selected launch: {resolved_launch.label}")
            print(f"Launch settings: {dict(resolved_launch.summary)}")
            for note in resolved_launch.notes:
                print(f"Note: {note}")
    if no_instantiate:
        return
    if resolved_launch is not None:
        resolved_launch.launch()
        return
    runner = config.setup()
    completed = False
    try:
        runner.run()
        completed = True
    finally:
        # Successful ranks rendezvous before bounded NCCL process exit.
        # A failed rank skips the barrier to avoid creating a cleanup deadlock.
        shutdown_distributed(
            synchronize=completed,
            terminate_process=completed,
        )


def _is_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _run_with_disk_error_handling(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:
        disk_error = disk_space_error_from_exception(exc)
        if disk_error is not None:
            if _is_rank_zero():
                print(str(disk_error), file=sys.stderr)
            raise SystemExit(1) from None
        raise


def entrypoint(argv: list[str] | None = None) -> None:
    """``flashdreams-run`` console-script entry point.

    Plugin/entry-point discovery is deferred until call time so
    importing :mod:`flashdreams.scripts.cli` is cheap.
    """
    tyro.extras.set_accent_color("bright_yellow")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    (
        normalized_args,
        runners,
        launch_manifest,
        mode,
        legacy_world_manifest,
    ) = _prepare_cli_args(raw_args)
    selected_runner_name = next(
        (value for value in normalized_args if value in runners),
        None,
    )
    help_suffix = ""
    if selected_runner_name is not None:
        help_options = LaunchOptions(
            legacy_world_manifest=legacy_world_manifest,
            scenario={} if launch_manifest is None else launch_manifest.scenario,
            output={} if launch_manifest is None else launch_manifest.output,
        )
        supported = available_launch_modes(
            runners[selected_runner_name],
            help_options,
        )
        help_suffix = (
            f" Selected mode: {mode}. Available modes: {', '.join(supported)}."
            " Use --manifest PATH for scenario and output settings."
        )
        if mode == "webrtc":
            help_suffix += (
                " WebRTC CLI overrides: --host HOST, --port PORT, and"
                " --prefer-sw-encoder."
            )
        runners[selected_runner_name] = dataclasses.replace(
            runners[selected_runner_name],
            description=runners[selected_runner_name].description + help_suffix,
        )
    union = _annotated_base_runner_union(runners)

    # ``name=""`` on the synthetic ``runner`` field suppresses its own
    # name from child prefixes, so ``--runner.prompt`` collapses to
    # ``--prompt`` and ``runner.pipeline.<encoder>:<concrete>``
    # selectors collapse to ``pipeline.<encoder>:<concrete>``. Nested
    # struct fields keep their own names for disambiguation.
    cli_fields: list[tuple] = [
        ("runner", Annotated[union, tyro.conf.arg(name="")]),
        (
            "no_instantiate",
            bool,
            dataclasses.field(default=False),
        ),
    ]
    if mode == "webrtc":
        cli_fields.extend(
            [
                (
                    "host",
                    str | None,
                    dataclasses.field(default=None),
                ),
                (
                    "port",
                    int | None,
                    dataclasses.field(default=None),
                ),
                (
                    "prefer_sw_encoder",
                    bool,
                    dataclasses.field(default=False),
                ),
            ]
        )
    args_cls = dataclasses.make_dataclass(
        "FlashdreamsRunArgs",
        cli_fields,
    )
    args_cls.__doc__ = (__doc__ or "") + help_suffix

    # Silence ``--help`` / parse-error banners on non-rank-0 ranks so
    # they print exactly once even though every rank parses argv. Every
    # rank still exits via ``sys.exit`` inside ``tyro.cli``; only the
    # printed output is gated.
    args = tyro.cli(
        args_cls,
        prog="flashdreams-run",
        description=args_cls.__doc__,
        console_outputs=_is_rank_zero(),
        args=normalized_args,
    )
    # ``args_cls`` is built dynamically; keep the untyped boundary explicit.
    parsed_args = cast(Any, args)
    runner_cfg: RunnerConfig = parsed_args.runner
    no_instantiate: bool = parsed_args.no_instantiate
    host: str | None = getattr(parsed_args, "host", None)
    port: int | None = getattr(parsed_args, "port", None)
    prefer_sw_encoder: bool = getattr(parsed_args, "prefer_sw_encoder", False)
    _run_with_disk_error_handling(
        lambda: main(
            runner_cfg,
            no_instantiate,
            mode=mode,
            host=host,
            port=port,
            legacy_world_manifest=legacy_world_manifest,
            prefer_sw_encoder=prefer_sw_encoder,
            launch_manifest=launch_manifest,
        )
    )


def _prepare_cli_args(
    args: list[str],
) -> tuple[
    list[str],
    dict[str, RunnerConfig],
    FlashDreamsLaunchManifest | None,
    LaunchMode,
    Path | None,
]:
    """Normalize positional launch modes and load an optional manifest."""
    normalized, manifest_path = _pop_option(args, "--manifest")
    runners = dict(all_runners())
    runner_index = next(
        (index for index, value in enumerate(normalized) if value in runners),
        None,
    )
    if runner_index is None:
        if manifest_path is not None:
            raise ValueError("--manifest requires an explicit runner slug.")
        return normalized, runners, None, "run", None

    runner_name = normalized[runner_index]
    positional_mode: LaunchMode | None = None
    if runner_index + 1 < len(normalized):
        candidate = normalized[runner_index + 1]
        if candidate in _POSITIONAL_MODES:
            positional_mode = cast(LaunchMode, candidate)
            del normalized[runner_index + 1]

    launch_manifest: FlashDreamsLaunchManifest | None = None
    legacy_world_manifest: Path | None = None
    if manifest_path is not None:
        try:
            launch_manifest = load_launch_manifest(manifest_path)
        except ValueError:
            if positional_mode != "local-window":
                raise
            legacy_world_manifest = Path(manifest_path).expanduser().resolve()
        else:
            if launch_manifest.runner != runner_name:
                raise ValueError(
                    f"Manifest runner {launch_manifest.runner!r} does not match "
                    f"selected runner {runner_name!r}."
                )
            if positional_mode is not None and launch_manifest.mode != positional_mode:
                raise ValueError(
                    f"Manifest mode {launch_manifest.mode!r} does not match "
                    f"selected mode {positional_mode!r}."
                )
            runners[runner_name] = launch_manifest.apply_runner_overrides(
                runners[runner_name]
            )

    raw_mode = positional_mode or (
        "run" if launch_manifest is None else launch_manifest.mode
    )
    if raw_mode not in _POSITIONAL_MODES:
        raise ValueError(
            f"Unsupported launch mode {raw_mode!r}. Expected one of: "
            f"{', '.join(sorted(_POSITIONAL_MODES))}."
        )
    mode = cast(LaunchMode, raw_mode)
    normalized = _hoist_global_options(normalized)
    return normalized, runners, launch_manifest, mode, legacy_world_manifest


def _pop_option(args: list[str], name: str) -> tuple[list[str], str | None]:
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(args):
        item = args[index]
        if item == name:
            if value is not None:
                raise ValueError(f"{name} may be specified only once.")
            if index + 1 >= len(args):
                raise ValueError(f"{name} requires a path.")
            value = args[index + 1]
            index += 2
            continue
        prefix = name + "="
        if item.startswith(prefix):
            if value is not None:
                raise ValueError(f"{name} may be specified only once.")
            value = item[len(prefix) :]
            index += 1
            continue
        remaining.append(item)
        index += 1
    return remaining, value


def _hoist_global_options(args: list[str]) -> list[str]:
    """Allow central launch flags before or after the runner subcommand."""
    value_options = {"--host", "--port"}
    flag_options = {"--no-instantiate", "--prefer-sw-encoder"}
    prefix: list[str] = []
    remaining: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in flag_options:
            prefix.append(item)
            index += 1
            continue
        if item in value_options:
            if index + 1 >= len(args):
                raise ValueError(f"{item} requires a value.")
            prefix.extend((item, args[index + 1]))
            index += 2
            continue
        if any(item.startswith(option + "=") for option in value_options):
            prefix.append(item)
            index += 1
            continue
        remaining.append(item)
        index += 1
    return [*prefix, *remaining]


if __name__ == "__main__":
    entrypoint()
