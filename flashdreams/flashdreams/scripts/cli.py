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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, cast

import tyro
import yaml

from flashdreams.configs.runner_configs import _annotated_base_runner_union, all_runners
from flashdreams.core.distributed import shutdown as shutdown_distributed
from flashdreams.core.io.disk import disk_space_error_from_exception
from flashdreams.demo import (
    Application,
    DemoAdapterApplication,
    run_application_replay,
    run_application_webrtc,
)
from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins import discover_applications
from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
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
    runners = dict(all_runners())
    target_name = _first_positional_arg(raw_args)
    if target_name is not None and target_name not in runners:
        applications = discover_applications()
        application_name = _selected_application_name(target_name, applications)
        if application_name is not None:
            _run_with_disk_error_handling(
                lambda: _entrypoint_application(
                    raw_args,
                    application_name=application_name,
                    application=applications[application_name],
                )
            )
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
            scenario_overrides=launch_overrides.scenario,
            output_overrides=launch_overrides.output,
        )
    )


def _first_positional_arg(args: Sequence[str]) -> str | None:
    index = 0
    while index < len(args):
        item = args[index]
        if item in {"--no-instantiate", "--prefer-sw-encoder", "--help", "-h"}:
            index += 1
            continue
        if item in {"--host", "--port", "--manifest"}:
            index += 2
            continue
        if any(item.startswith(option + "=") for option in ("--host", "--port")):
            index += 1
            continue
        parsed_override = _parse_launch_override_token(item)
        if parsed_override is not None:
            _section, _key, inline_value = parsed_override
            index += 1 if inline_value is not None else 2
            continue
        if item.startswith("-"):
            index += 1
            continue
        return item
    return None


def _selected_application_name(
    target_name: str,
    applications: Mapping[str, Application],
) -> str | None:
    if target_name in applications:
        return target_name
    return None


def _entrypoint_application(
    raw_args: list[str],
    *,
    application_name: str,
    application: Application,
) -> None:
    if "--help" in raw_args or "-h" in raw_args:
        _print_application_help(application_name, application)
        raise SystemExit(0)
    mode, launch_args, no_instantiate, launch_overrides = _prepare_application_cli_args(
        raw_args, application_name=application_name
    )
    if mode not in {"mp4", "null", "webrtc"}:
        raise ValueError(
            f"Application {application_name!r} currently supports direct launch "
            "modes 'mp4', 'null', and 'webrtc'. Use a compatibility runner for "
            f"{mode!r}."
        )
    scenario = dict(launch_overrides.scenario)
    output = dict(launch_overrides.output)
    configured = _configure_application_launch(
        application=application,
        application_name=application_name,
        mode=mode,
        scenario_overrides=scenario,
        output_overrides=output,
    )
    if _is_rank_zero():
        print(f"Resolved application: {application_name!r}")
        print(f"Launch mode: {mode}")
        if scenario:
            print(f"Scenario: {scenario}")
        if output:
            print(f"Output settings: {output}")
    if no_instantiate:
        return
    if mode == "webrtc":
        _handle_launch_result(
            run_application_webrtc(app=configured, launch_args=launch_args)
        )
    else:
        _handle_launch_result(
            run_application_replay(app=configured, launch_args=launch_args)
        )


def _prepare_application_cli_args(
    args: list[str],
    *,
    application_name: str,
) -> tuple[LaunchMode, tuple[str, ...], bool, _LaunchCliOverrides]:
    normalized, launch_overrides = _pop_launch_overrides(args)
    normalized, manifest_path = _pop_option(normalized, "--manifest")
    if manifest_path is not None:
        raise ValueError(
            "--manifest is not supported for direct application launches yet; "
            "use --scenario.KEY and --output.KEY overrides."
        )
    normalized, host = _pop_option(normalized, "--host")
    normalized, port = _pop_option(normalized, "--port")
    if host is not None or port is not None:
        output_overrides = dict(launch_overrides.output)
        if host is not None:
            output_overrides["host"] = host
        if port is not None:
            output_overrides["port"] = port
        launch_overrides = dataclasses.replace(
            launch_overrides,
            output=output_overrides,
        )
    normalized = _hoist_global_options(normalized)
    no_instantiate = False
    remaining: list[str] = []
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item == "--no-instantiate":
            no_instantiate = True
            index += 1
            continue
        if item == "--prefer-sw-encoder":
            raise ValueError(
                "--prefer-sw-encoder is only supported by WebRTC runner launches."
            )
        remaining.append(item)
        index += 1

    try:
        app_index = remaining.index(application_name)
    except ValueError as exc:
        raise ValueError(
            f"Application slug {application_name!r} was not present in argv."
        ) from exc
    del remaining[app_index]
    raw_mode: LaunchMode = "run"
    if app_index < len(remaining) and remaining[app_index] in _POSITIONAL_MODES:
        raw_mode = cast(LaunchMode, remaining.pop(app_index))
    return raw_mode, tuple(remaining), no_instantiate, launch_overrides


def _configure_application_launch(
    *,
    application: Application,
    application_name: str,
    mode: LaunchMode,
    scenario_overrides: Mapping[str, object],
    output_overrides: Mapping[str, object],
) -> Application:
    if not isinstance(application, DemoAdapterApplication):
        if scenario_overrides or output_overrides or mode != "null":
            raise ValueError(
                "Direct application launch with scenario/output overrides requires "
                "a DemoAdapterApplication."
            )
        return application

    scenario = _application_scenario(application.spec, scenario_overrides)
    output = _application_output_spec(
        application_name=application_name,
        mode=mode,
        spec=application.spec,
        scenario=scenario,
        output_overrides=output_overrides,
    )
    adapter = _application_adapter_for_output(application.adapter, output)
    return DemoAdapterApplication(
        adapter=adapter,
        spec=dataclasses.replace(
            application.spec,
            input_mode="webrtc" if mode == "webrtc" else "replay",
            scenario=scenario,
            output=output,
        ),
    )


def _application_adapter_for_output(
    adapter: DemoAdapter,
    output: object,
) -> DemoAdapter:
    configure_for_output = getattr(adapter, "configure_for_output", None)
    if not callable(configure_for_output):
        return adapter
    return cast(DemoAdapter, configure_for_output(output))


def _application_scenario(
    spec: DemoSpec,
    overrides: Mapping[str, object],
) -> object:
    if not overrides:
        return spec.scenario
    if spec.scenario is None:
        return dict(overrides)
    if not isinstance(spec.scenario, Mapping):
        raise ValueError(
            "Scenario overrides require the application DemoSpec.scenario to be "
            f"a mapping, got {type(spec.scenario).__name__}."
        )
    return {**dict(spec.scenario), **dict(overrides)}


def _application_output_spec(
    *,
    application_name: str,
    mode: LaunchMode,
    spec: DemoSpec,
    scenario: object,
    output_overrides: Mapping[str, object],
) -> Mp4OutputSpec | NullOutputSpec | WebRTCOutputSpec:
    if mode == "null":
        _reject_unknown_output_keys(output_overrides, allowed={"store_results"})
        return NullOutputSpec(
            store_results=bool(output_overrides.get("store_results", False))
        )
    if mode == "webrtc":
        return _application_webrtc_output_spec(
            spec=spec,
            scenario=scenario,
            output_overrides=output_overrides,
        )
    if mode != "mp4":
        raise ValueError(f"Direct application launch mode {mode!r} is not implemented.")
    _reject_unknown_output_keys(
        output_overrides,
        allowed={"fps", "layout", "move_to_cpu", "output", "output_layout", "path"},
    )
    current_output = spec.output
    path = output_overrides.get("path", output_overrides.get("output"))
    if path is None and isinstance(current_output, Mp4OutputSpec):
        path = current_output.path
    if path is None:
        path = Path("outputs") / f"{application_name}.mp4"
    fps = output_overrides.get("fps")
    if fps is None and isinstance(current_output, Mp4OutputSpec):
        fps = current_output.fps
    if fps is None:
        fps = _scenario_field(scenario, "fps")
    if fps is None:
        fps = spec.metadata.get("fps")
    if fps is None:
        raise ValueError(
            "Direct application MP4 launch requires --output.fps or an fps "
            "value in the application scenario or metadata."
        )
    output_layout = output_overrides.get(
        "output_layout",
        output_overrides.get("layout"),
    )
    if output_layout is None and isinstance(current_output, Mp4OutputSpec):
        output_layout = current_output.output_layout
    if output_layout is None:
        output_layout = spec.metadata.get("output_layout", "bvtchw")
    move_to_cpu = output_overrides.get("move_to_cpu")
    if move_to_cpu is None and isinstance(current_output, Mp4OutputSpec):
        move_to_cpu = current_output.move_to_cpu
    return Mp4OutputSpec(
        path=Path(str(path)),
        fps=_positive_number(fps, name="fps"),
        output_layout=cast(Any, str(output_layout)),
        move_to_cpu=bool(True if move_to_cpu is None else move_to_cpu),
    )


def _application_webrtc_output_spec(
    *,
    spec: DemoSpec,
    scenario: object,
    output_overrides: Mapping[str, object],
) -> WebRTCOutputSpec:
    _reject_unknown_output_keys(
        output_overrides,
        allowed={
            "client_liveness_timeout_s",
            "fps",
            "host",
            "port",
            "preload_name",
            "request_session_path",
            "video_height",
            "video_width",
            "warmup_chunks",
            "warmup_timeout_s",
            "web_dir",
        },
    )
    current_output = spec.output
    return WebRTCOutputSpec(
        host=str(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "host",
                default="127.0.0.1",
            )
        ),
        port=_positive_int(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "port",
                default=8080,
            ),
            name="port",
        ),
        fps=_positive_int(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "fps",
                default=(
                    _scenario_field(scenario, "fps") or spec.metadata.get("fps") or 30
                ),
            ),
            name="fps",
        ),
        video_width=_positive_int(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "video_width",
                default=(
                    _scenario_field(scenario, "pixel_width")
                    or spec.metadata.get("video_width")
                    or 1280
                ),
            ),
            name="video_width",
        ),
        video_height=_positive_int(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "video_height",
                default=(
                    _scenario_field(scenario, "pixel_height")
                    or spec.metadata.get("video_height")
                    or 720
                ),
            ),
            name="video_height",
        ),
        warmup_chunks=_non_negative_int(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "warmup_chunks",
                default=0,
            ),
            name="warmup_chunks",
        ),
        warmup_timeout_s=float(
            _positive_number(
                _webrtc_output_value(
                    output_overrides,
                    current_output,
                    "warmup_timeout_s",
                    default=30.0,
                ),
                name="warmup_timeout_s",
            )
        ),
        client_liveness_timeout_s=float(
            _positive_number(
                _webrtc_output_value(
                    output_overrides,
                    current_output,
                    "client_liveness_timeout_s",
                    default=30.0,
                ),
                name="client_liveness_timeout_s",
            )
        ),
        web_dir=_optional_path(
            _webrtc_output_value(output_overrides, current_output, "web_dir")
        ),
        request_session_path=str(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "request_session_path",
                default="/request_session",
            )
        ),
        preload_name=_optional_str(
            _webrtc_output_value(
                output_overrides,
                current_output,
                "preload_name",
            )
        ),
    )


def _webrtc_output_value(
    output_overrides: Mapping[str, object],
    current_output: object,
    name: str,
    *,
    default: object | None = None,
) -> object | None:
    if name in output_overrides:
        return output_overrides[name]
    if isinstance(current_output, WebRTCOutputSpec):
        return getattr(current_output, name)
    return default


def _scenario_field(scenario: object, name: str) -> object | None:
    if isinstance(scenario, Mapping):
        return scenario.get(name)
    return None


def _positive_number(value: object, *, name: str) -> int | float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")
    if isinstance(value, int | float):
        number = value
    elif isinstance(value, str):
        number = float(value) if "." in value else int(value)
    else:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    if float(number) <= 0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _positive_int(value: object, *, name: str) -> int:
    number = _positive_number(value, name=name)
    return int(number)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        number = int(value)
    elif isinstance(value, str):
        number = int(value)
    else:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if number < 0:
        raise ValueError(f"{name} must be >= 0.")
    return number


def _optional_path(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _reject_unknown_output_keys(
    output_overrides: Mapping[str, object],
    *,
    allowed: set[str],
) -> None:
    unknown = sorted(set(output_overrides) - allowed)
    if unknown:
        raise ValueError(f"Unsupported application output fields: {', '.join(unknown)}")


def _print_application_help(application_name: str, application: Application) -> None:
    modes = ("null",)
    if isinstance(application, DemoAdapterApplication):
        supported = set(application.adapter.supported_output_modes())
        modes = tuple(mode for mode in ("mp4", "null", "webrtc") if mode in supported)
    print(f"Usage: flashdreams-run {application_name} <mode> [options]")
    print(f"Available direct application modes: {', '.join(modes)}")
    print("Use --scenario.KEY VALUE and --output.KEY VALUE for mode settings.")


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
