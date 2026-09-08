# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot Omnidreams attention-module and DiT-block benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark.common import add_plot_io_arguments, benchmark_artifact_dir

from ._plot import plot_modules

_DEFAULT_OUTPUT_DIR = benchmark_artifact_dir(Path("artifacts/benchmark/omnidreams"))
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "module.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_io_arguments(parser, _DEFAULT_INPUT, _DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    plot_modules(args.input, args.output_dir)


if __name__ == "__main__":
    main()
