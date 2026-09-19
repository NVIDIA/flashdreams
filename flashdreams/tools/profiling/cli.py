# SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
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

"""Launch flashdreams-run-v2 under Nsight Systems with NVTX ranges enabled.

nsys must own the traced process, so the runner is spawned as a subprocess.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

_RUNNER = "flashdreams-run-v2"
_DEFAULT_TRACE = "cuda,nvtx,osrt,vulkan"
_REPORT_SUFFIX = ".nsys-rep"


def resolve_runner() -> str | None:
    """Return the runner's console script, preferring this venv over ``PATH``.

    The v2 CLI module has no ``__main__`` guard, so ``python -m`` would exit
    without running anything.
    """
    sibling = Path(sys.executable).parent / _RUNNER
    # is_file(): a directory also passes the X_OK check.
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which(_RUNNER)


def build_command(
    *,
    runner_args: Sequence[str],
    runner: str,
    report_path: Path,
    trace: str,
) -> list[str]:
    """Return the ``nsys profile`` argv wrapping one runner invocation."""
    return [
        "nsys",
        "profile",
        f"--trace={trace}",
        "--output",
        # nsys appends the .nsys-rep extension itself.
        str(report_path.with_suffix("")),
        "--force-overwrite",
        "true",
        runner,
        *runner_args,
    ]


def default_report_path() -> Path:
    """Default report path; the PID keeps concurrent runs from sharing a file."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{_RUNNER}-{stamp}-{os.getpid()}{_REPORT_SUFFIX}"
    return Path("artifacts", "profiles", name)


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one runner invocation and write an ``.nsys-rep`` report."""
    args = _parse_args(argv)
    if shutil.which("nsys") is None:
        print("nsys not found on PATH; install Nsight Systems.", file=sys.stderr)
        return 127
    runner = resolve_runner()
    if runner is None:
        print(f"{_RUNNER} not found; run `uv sync` first.", file=sys.stderr)
        return 127

    report_path = args.report_path or default_report_path()
    if report_path.suffix != _REPORT_SUFFIX:
        report_path = report_path.with_name(report_path.name + _REPORT_SUFFIX)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        runner_args=args.runner_args,
        runner=runner,
        report_path=report_path,
        trace=args.trace,
    )
    # infra.nvtx reads this once at import, so the child must start with it set.
    env = {**os.environ, "FLASHDREAMS_NVTX": "1"}
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode == 0:
        print(f"report: {report_path}")
    return completed.returncode


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flashdreams-profile",
        description=__doc__,
        epilog=(
            "Arguments after -- go to flashdreams-run-v2, and the application's "
            "own arguments follow a second --."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Where to write the report; .nsys-rep is added if missing. "
        "Default: artifacts/profiles/<runner>-<timestamp>-<pid>.nsys-rep.",
    )
    parser.add_argument(
        "--trace",
        default=_DEFAULT_TRACE,
        metavar="LIST",
        help="Value for nsys --trace. Default: %(default)s.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the runner, after '--'.",
    )
    args = parser.parse_args(argv)
    if args.runner_args and args.runner_args[0] == "--":
        args.runner_args = args.runner_args[1:]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
