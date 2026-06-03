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

    # Multi-GPU via context-parallelism (integration transformers auto-detect
    # CP size from the launcher's WORLD group). ``--no-python`` tells
    # torchrun to execvp the console script directly instead of wrapping
    # it in ``python <script>``:
    torchrun --nproc_per_node=N --no-python flashdreams-run <slug> ...
"""

from __future__ import annotations

import dataclasses
import os
from typing import Annotated

import tyro

from flashdreams.configs.runner_configs import _annotated_base_runner_union
from flashdreams.infra.runner import RunnerConfig


def main(config: RunnerConfig, no_instantiate: bool = False) -> None:
    """Print the resolved config and (by default) run the runner.

    Under ``torchrun`` only local-rank 0 prints; every rank holds the
    same resolved config.
    """
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Resolved config for {config.runner_name!r}:")
        print(config)
    if no_instantiate:
        return
    runner = config.setup()
    runner.run()


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
        ],
    )
    args_cls.__doc__ = __doc__

    # Silence ``--help`` / parse-error banners on non-rank-0 ranks so
    # they print exactly once even though every rank parses argv. Every
    # rank still exits via ``sys.exit`` inside ``tyro.cli``; only the
    # printed output is gated.
    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    args = tyro.cli(
        args_cls,
        prog="flashdreams-run",
        description=__doc__,
        console_outputs=is_rank_zero,
    )
    # ``args_cls`` is built dynamically so the static checker only
    # sees ``object``; ``getattr`` keeps the type narrowing local.
    runner_cfg: RunnerConfig = getattr(args, "runner")
    no_instantiate: bool = getattr(args, "no_instantiate")
    main(runner_cfg, no_instantiate)


if __name__ == "__main__":
    entrypoint()
