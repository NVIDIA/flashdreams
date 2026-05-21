# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner config literals for the SANA-WM plugin."""

from __future__ import annotations

from flashdreams.infra.runner import RunnerConfig

from sana_wm.runner import SanaWMRunnerConfig

RUNNER_SANA_WM_BIDIRECTIONAL = SanaWMRunnerConfig(
    runner_name="sana-wm-bidirectional",
    description=(
        "SANA-WM bidirectional image-to-video world model "
        "(camera trajectory + first frame)."
    ),
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    RUNNER_SANA_WM_BIDIRECTIONAL.runner_name: RUNNER_SANA_WM_BIDIRECTIONAL,
}

__all__ = ["RUNNER_CONFIGS", "RUNNER_SANA_WM_BIDIRECTIONAL"]
