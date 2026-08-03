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

"""Strict deterministic launcher for ``flashdreams-run`` benchmark scenarios.

This mirrors the subprocess setup used by the Omnidreams same-seed CI test: set
CUDA/PyTorch determinism knobs before the first CUDA context, then invoke the
existing ``flashdreams-run`` entrypoint with the remaining arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``flashdreams-run`` with deterministic CUDA/PyTorch settings."""

    args = _parse_args(argv)
    configure_strict_determinism()
    _enable_torch_deterministic_algorithms()

    original_argv = sys.argv
    try:
        sys.argv = ["flashdreams-run", *args.runner_args]
        from flashdreams.scripts.cli import entrypoint  # noqa: PLC0415

        entrypoint()
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
        "runner_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed to flashdreams-run. Prefix them with '--' when "
            "the first runner argument is an option."
        ),
    )
    args = parser.parse_args(argv)
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    if not args.runner_args:
        parser.error("pass a flashdreams-run scenario and arguments after '--'")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
