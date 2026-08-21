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

"""Strict deterministic launcher for FlashDreams benchmark scenarios.

This mirrors the subprocess setup used by the Omnidreams same-seed CI test: set
CUDA/PyTorch determinism knobs before the first CUDA context, then invoke a
FlashDreams runner entrypoint with the remaining arguments. Either API's
entrypoint can be launched, a quality scenario wanting the same determinism
whichever one generates the clip.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence

_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
_ENTRYPOINT_MODULES = {
    "flashdreams-run": "flashdreams.scripts.cli",
    "flashdreams-run-v2": "flashdreams.runtime_v2.cli",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run a FlashDreams entrypoint with deterministic CUDA/PyTorch settings."""

    args = _parse_args(argv)
    configure_strict_determinism()
    _enable_torch_deterministic_algorithms()

    original_argv = sys.argv
    try:
        # In-process rather than spawned, so the determinism above is already
        # set when the runner opens its first CUDA context.
        sys.argv = [args.entrypoint, *args.runner_args]
        module = importlib.import_module(_ENTRYPOINT_MODULES[args.entrypoint])
        module.entrypoint()
    finally:
        sys.argv = original_argv
    return 0


def configure_strict_determinism() -> None:
    """Set deterministic CUDA allocator/cuBLAS defaults before Torch import."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        _DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    )


def _enable_torch_deterministic_algorithms() -> None:
    import torch  # noqa: PLC0415

    torch.use_deterministic_algorithms(True, warn_only=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrypoint",
        choices=tuple(_ENTRYPOINT_MODULES),
        default="flashdreams-run",
        help=(
            "Runner to launch. Give it before the runner arguments, which are "
            "everything after '--'. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed to the runner. Prefix them with '--' when the "
            "first runner argument is an option."
        ),
    )
    args = parser.parse_args(argv)
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    if not args.runner_args:
        parser.error(f"pass {args.entrypoint} arguments after '--'")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
