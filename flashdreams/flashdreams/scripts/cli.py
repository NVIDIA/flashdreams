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

Mirrors nerfstudio's ``ns-train`` shape: one hyphenated console script
fronts a tyro subcommand union built from the runner registry. Each
subcommand uses its registered :class:`RunnerConfig` literal as
``defaults=`` and exposes every nested field as a CLI flag.

Usage::

    flashdreams-run --help                            # list every runner
    flashdreams-run wan21-t2v-1.3b-480p --help        # show overridable fields
    flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."
    flashdreams-run wan21-i2v-14b-480p --prompt "..." --image_path frame.png
    flashdreams-run template-offline --no-instantiate # resolve config only

    # Multi-GPU via context-parallelism (recipe transformers auto-detect
    # CP size from the launcher's WORLD group). ``--no-python`` tells
    # torchrun to execvp the console script directly instead of wrapping
    # it in ``python <script>`` (which would treat ``flashdreams-run`` as
    # a relative path in cwd, not a PATH lookup):
    torchrun --nproc_per_node=N --no-python flashdreams-run <slug> ...

The runner owns the entire end-to-end recipe: load inputs (prompt,
image, ...), drive the pipeline AR loop, persist outputs.
``StreamInferencePipeline`` stays narrow so serving deployments
(e.g. ``integrations/lingbot`` WebRTC) can keep using the bare
pipeline without inheriting any of the CLI's I/O.
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

    The print step is the easiest way to confirm overrides landed in
    the right place; the instantiate step then builds the pipeline on
    the configured device and dispatches into ``runner.run()``. Under
    ``torchrun`` only the local-rank-0 process prints (every rank holds
    the same resolved config, so N copies would just spam stdout).
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

    Built lazily so the heavy entry-point discovery only fires when the
    CLI actually runs -- importing :mod:`flashdreams.scripts.cli` is
    cheap. Under ``torchrun`` non-zero local ranks silence tyro's
    stdout/stderr so ``--help`` and parse-error blocks print exactly
    once even though every rank parses the same argv independently.
    """
    tyro.extras.set_accent_color("bright_yellow")
    union = _annotated_base_runner_union()

    # Built at runtime because ``union`` is a tyro-annotated subcommand
    # type assembled from the (in-tree + plugin) runner registry.
    # ``name=""`` on the synthetic ``runner`` field suppresses its own
    # name from child prefixes, so ``--runner.prompt`` collapses to
    # ``--prompt`` and the ``runner.pipeline.<encoder>:<concrete>``
    # subcommand selectors collapse to ``pipeline.<encoder>:<concrete>``.
    # Nested struct fields (``pipeline``, ``encoder``, ...) keep their
    # own names for disambiguation. (``prefix_name=False`` is the wrong
    # knob -- it strips the *parent's* prefix from this field, not this
    # field's name from its children.)
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

    # Silence ``--help`` / parse-error banners on non-rank-0 ranks via
    # tyro's built-in distributed-mode hook. Every rank still exits
    # consistently because ``--help`` / parse-error paths call
    # ``sys.exit(...)`` inside ``tyro.cli`` regardless; only the printed
    # output is gated. The success path falls through cleanly; ``main``
    # gates its own prints separately.
    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    args = tyro.cli(
        args_cls,
        prog="flashdreams-run",
        description=__doc__,
        console_outputs=is_rank_zero,
    )
    main(args.runner, args.no_instantiate)


if __name__ == "__main__":
    entrypoint()
