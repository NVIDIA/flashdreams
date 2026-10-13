# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first benchmark harness for FlashDreams runners and demos."""

from tools.benchmarks.harness import ScenarioRunResult, run_benchmark_suite
from tools.benchmarks.quality import QualityBaselineConfig
from tools.benchmarks.scenarios import (
    BenchmarkScenario,
    QualityCommandConfig,
    built_in_scenarios,
    load_scenario_file,
)

__all__ = [
    "BenchmarkScenario",
    "QualityBaselineConfig",
    "QualityCommandConfig",
    "ScenarioRunResult",
    "built_in_scenarios",
    "load_scenario_file",
    "run_benchmark_suite",
]
