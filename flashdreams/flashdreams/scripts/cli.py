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
    flashdreams-run t2v-causal-forcing --prompt "A forest waterfall."

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import tyro
import yaml

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
_LAUNCH_OVERRIDE_SECTIONS = frozenset({"scenario", "output"})


@dataclasses.dataclass(frozen=True, slots=True)
class _LaunchCliOverrides:
    scenario: Mapping[str, object] = dataclasses.field(default_factory=dict)
    output: Mapping[str, object] = dataclasses.field(default_factory=dict)


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
    scenario_overrides: Mapping[str, object] | None = None,
    output_overrides: Mapping[str, object] | None = None,
) -> None:
    """Print the resolved config and (by default) run the runner.

    Under ``torchrun`` only local-rank 0 prints; every rank holds the
    same resolved config.
    """
    resolved_launch = None
    scenario = _merge_launch_settings(
        {} if launch_manifest is None else launch_manifest.scenario,
        scenario_overrides,
    )
    output = _merge_launch_settings(
        {} if launch_manifest is None else launch_manifest.output,
        output_overrides,
    )
    launch_options = LaunchOptions(
        host=host,
        port=port,
        prefer_sw_encoder=prefer_sw_encoder,
        legacy_world_manifest=legacy_world_manifest,
        launch_manifest=None if launch_manifest is None else launch_manifest.path,
        scenario=scenario,
        output=output,
    )
    if mode != "run":
        resolved_launch = resolve_launch(
            config,
            mode=mode,
            options=launch_options,
        )

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print_full_config = mode == "run" or no_instantiate
        if print_full_config:
            print(f"Resolved config for {config.runner_name!r}:")
            print(config)
            print(
                "Available modes: "
                f"{', '.join(available_launch_modes(config, launch_options))}"
            )
        else:
            print(f"Resolved runner: {config.runner_name!r}")
        if launch_manifest is not None:
            print(f"Launch manifest: {launch_manifest.path}")
            print(f"Launch mode: {launch_manifest.mode}")
        if launch_manifest is not None or scenario:
            print(f"Scenario: {dict(scenario)}")
        if launch_manifest is not None or output:
            print(f"Output settings: {dict(output)}")
        if resolved_launch is not None:
            print(f"Selected launch: {resolved_launch.label}")
            print(f"Launch settings: {dict(resolved_launch.summary)}")
            for note in resolved_launch.notes:
                print(f"Note: {note}")
    if no_instantiate:
        return
    if resolved_launch is not None:
        _handle_launch_result(resolved_launch.launch())
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


def _handle_launch_result(result: object) -> None:
    from flashdreams.runtime.demo import RunResult

    if not isinstance(result, RunResult):
        return
    if result.status in {"completed", "skipped"}:
        return
    reason = result.reason or (str(result.error) if result.error is not None else None)
    if reason is None:
        reason = f"Launch ended with status {result.status!r}."
    if _is_rank_zero():
        print(reason, file=sys.stderr)
    raise SystemExit(1)


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
    from flashdreams.demo.application import (
        entrypoint as application_entrypoint,
    )
    from flashdreams.demo.application import (
        registered_application_slugs,
    )

    application_slugs = registered_application_slugs()
    if raw_args and raw_args[0] in application_slugs:
        application_entrypoint(raw_args)
        return

    (
        normalized_args,
        runners,
        launch_manifest,
        mode,
        legacy_world_manifest,
        launch_overrides,
    ) = _prepare_cli_args(raw_args)
    selected_runner_name = next(
        (value for value in normalized_args if value in runners),
        None,
    )
    help_suffix = ""
    if selected_runner_name is not None:
        help_options = LaunchOptions(
            legacy_world_manifest=legacy_world_manifest,
            scenario=_merge_launch_settings(
                {} if launch_manifest is None else launch_manifest.scenario,
                launch_overrides.scenario,
            ),
            output=_merge_launch_settings(
                {} if launch_manifest is None else launch_manifest.output,
                launch_overrides.output,
            ),
        )
        supported = available_launch_modes(
            runners[selected_runner_name],
            help_options,
        )
        help_suffix = (
            f" Selected mode: {mode}. Available modes: {', '.join(supported)}."
            " Use --manifest PATH for scenario/output settings, or"
            " --scenario.KEY VALUE and --output.KEY VALUE for simple overrides."
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
    parser_runners = (
        {selected_runner_name: runners[selected_runner_name]}
        if selected_runner_name is not None
        else runners
    )
    union = _annotated_base_runner_union(parser_runners)

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
    application_help = ""
    if application_slugs:
        application_help = "\n\nInstalled application demo slugs:\n  " + "\n  ".join(
            application_slugs
        )
    args_cls.__doc__ = (__doc__ or "") + help_suffix + application_help

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
            scenario_overrides=launch_overrides.scenario,
            output_overrides=launch_overrides.output,
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
    _LaunchCliOverrides,
]:
    """Normalize positional launch modes and load an optional manifest."""
    normalized, launch_overrides = _pop_launch_overrides(args)
    normalized, manifest_path = _pop_option(normalized, "--manifest")
    runners = dict(all_runners())
    runner_index = next(
        (index for index, value in enumerate(normalized) if value in runners),
        None,
    )
    if runner_index is None:
        if manifest_path is not None:
            raise ValueError("--manifest requires an explicit runner slug.")
        return normalized, runners, None, "run", None, launch_overrides

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
    return (
        normalized,
        runners,
        launch_manifest,
        mode,
        legacy_world_manifest,
        launch_overrides,
    )


def _merge_launch_settings(
    base: Mapping[str, object],
    overrides: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def _pop_launch_overrides(args: list[str]) -> tuple[list[str], _LaunchCliOverrides]:
    remaining: list[str] = []
    overrides: dict[str, dict[str, object]] = {"scenario": {}, "output": {}}
    index = 0
    while index < len(args):
        parsed = _parse_launch_override_token(args[index])
        if parsed is None:
            remaining.append(args[index])
            index += 1
            continue
        section, key, inline_value = parsed
        if inline_value is None:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(
                    f"--{section}.{key.replace('_', '-')} requires a value."
                )
            raw_value = args[index + 1]
            index += 2
        else:
            raw_value = inline_value
            index += 1
        overrides[section][key] = _parse_launch_override_value(raw_value)
    return remaining, _LaunchCliOverrides(
        scenario=overrides["scenario"],
        output=overrides["output"],
    )


def _parse_launch_override_token(
    token: str,
) -> tuple[str, str, str | None] | None:
    for section in _LAUNCH_OVERRIDE_SECTIONS:
        prefix = f"--{section}."
        if not token.startswith(prefix):
            continue
        raw_key, separator, inline_value = token[len(prefix) :].partition("=")
        if not raw_key:
            raise ValueError(f"{prefix}<key> requires a non-empty key.")
        key = raw_key.replace("-", "_")
        return section, key, inline_value if separator else None
    return None


def _parse_launch_override_value(raw_value: str) -> object:
    text = raw_value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if text.lstrip("+-").isdigit():
        return int(text)
    if any(marker in text for marker in (".", "e", "E")):
        try:
            return float(text)
        except ValueError:
            pass
    if text.startswith("[") and text.endswith("]"):
        return _parse_launch_override_list(text)
    return raw_value


def _parse_launch_override_list(raw_value: str) -> list[object]:
    parsed = yaml.safe_load(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a list override value, got {type(parsed).__name__}."
        )
    return [_validate_launch_override_list_item(item) for item in parsed]


def _validate_launch_override_list_item(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Launch override list values must be strings, numbers, booleans, or null; "
        f"got {type(value).__name__}."
    )


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
