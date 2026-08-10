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
    flashdreams-run --output webrtc lingbot-world-fast
    flashdreams-run --output local-window omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae

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

from flashdreams.configs.runner_configs import _annotated_base_runner_union
from flashdreams.core.distributed import shutdown as shutdown_distributed
from flashdreams.core.io.disk import disk_space_error_from_exception
from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.output_targets import (
    OutputLaunchOptions,
    OutputMode,
    available_output_modes,
    launch_output_target,
    resolve_output_target,
)


def main(
    config: RunnerConfig,
    no_instantiate: bool = False,
    *,
    output: OutputMode = "cli",
    output_host: str | None = None,
    output_port: int | None = None,
    output_manifest: Path | None = None,
    prefer_sw_encoder: bool = False,
) -> None:
    """Print the resolved config and (by default) run the runner.

    Under ``torchrun`` only local-rank 0 prints; every rank holds the
    same resolved config.
    """
    output_spec = None
    output_options = OutputLaunchOptions(
        host=output_host,
        port=output_port,
        prefer_sw_encoder=prefer_sw_encoder,
        local_window_manifest=output_manifest,
    )
    if output != "cli":
        output_spec = resolve_output_target(
            config,
            mode=output,
            options=output_options,
        )

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Resolved config for {config.runner_name!r}:")
        print(config)
        print(
            f"Available outputs: {', '.join(available_output_modes(config, output_options))}"
        )
        if output_spec is not None:
            print(f"Selected output: {output_spec.label}")
            print(f"Launch command: {output_spec.command}")
            for note in output_spec.notes:
                print(f"Note: {note}")
    if no_instantiate:
        return
    if output_spec is not None:
        launch_output_target(output_spec)
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


def entrypoint() -> None:
    """``flashdreams-run`` console-script entry point.

    Plugin/entry-point discovery is deferred until call time so
    importing :mod:`flashdreams.scripts.cli` is cheap.
    """
    tyro.extras.set_accent_color("bright_yellow")
    union = _annotated_base_runner_union()

    # ``name=""`` on the synthetic ``runner`` field suppresses its own
    # name from child prefixes, so ``--runner.prompt`` collapses to
    # ``--prompt`` and ``runner.pipeline.<encoder>:<concrete>``
    # selectors collapse to ``pipeline.<encoder>:<concrete>``. Nested
    # struct fields keep their own names for disambiguation.
    args_cls = dataclasses.make_dataclass(
        "FlashdreamsRunArgs",
        [
            ("runner", Annotated[union, tyro.conf.arg(name="")]),
            (
                "no_instantiate",
                bool,
                dataclasses.field(default=False),
            ),
            (
                "output",
                OutputMode,
                dataclasses.field(default="cli"),
            ),
            (
                "output_host",
                str | None,
                dataclasses.field(default=None),
            ),
            (
                "output_port",
                int | None,
                dataclasses.field(default=None),
            ),
            (
                "output_manifest",
                Path | None,
                dataclasses.field(default=None),
            ),
            (
                "prefer_sw_encoder",
                bool,
                dataclasses.field(default=False),
            ),
        ],
    )
    args_cls.__doc__ = __doc__

    # Silence ``--help`` / parse-error banners on non-rank-0 ranks so
    # they print exactly once even though every rank parses argv. Every
    # rank still exits via ``sys.exit`` inside ``tyro.cli``; only the
    # printed output is gated.
    args = tyro.cli(
        args_cls,
        prog="flashdreams-run",
        description=__doc__,
        console_outputs=_is_rank_zero(),
    )
    # ``args_cls`` is built dynamically; keep the untyped boundary explicit.
    parsed_args = cast(Any, args)
    runner_cfg: RunnerConfig = parsed_args.runner
    no_instantiate: bool = parsed_args.no_instantiate
    output: OutputMode = parsed_args.output
    output_host: str | None = parsed_args.output_host
    output_port: int | None = parsed_args.output_port
    output_manifest: Path | None = parsed_args.output_manifest
    prefer_sw_encoder: bool = parsed_args.prefer_sw_encoder
    _run_with_disk_error_handling(
        lambda: main(
            runner_cfg,
            no_instantiate,
            output=output,
            output_host=output_host,
            output_port=output_port,
            output_manifest=output_manifest,
            prefer_sw_encoder=prefer_sw_encoder,
        )
    )


if __name__ == "__main__":
    entrypoint()
