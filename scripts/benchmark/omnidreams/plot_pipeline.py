# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot Omnidreams network and pipeline benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark.common import add_plot_io_arguments, benchmark_artifact_dir

from ._plot import plot_pipeline

_DEFAULT_OUTPUT_DIR = benchmark_artifact_dir(Path("artifacts/benchmark/omnidreams"))
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "pipeline.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_io_arguments(parser, _DEFAULT_INPUT, _DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pipeline-generate-only",
        action="store_true",
        help="write only the pipeline generate benchmark figure",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    plot_pipeline(
        args.input,
        args.output_dir,
        pipeline_generate_only=args.pipeline_generate_only,
    )


if __name__ == "__main__":
    main()
