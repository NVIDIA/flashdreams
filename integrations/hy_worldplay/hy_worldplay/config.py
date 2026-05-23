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

"""Runner configs for the HY-WorldPlay integration (WAN-5B I2V)."""

from __future__ import annotations

from flashdreams.infra.runner import RunnerConfig
from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

# Callers must override ``ar_model_path`` / ``ckpt_path`` /
# ``hy_worldplay_repo_root`` (and ``image_path``) at runtime; real paths
# are machine-specific and intentionally not baked in here.
RUNNER_HY_WORLDPLAY_WAN_I2V_5B = HyWorldPlayWanI2VRunnerConfig(
    runner_name="hy-worldplay-wan-i2v-5b",
    description=(
        "HY-WorldPlay WAN-5B I2V (Wan 2.2 TI2V backbone, action + camera "
        "trajectory conditioning, reconstituted-context memory)."
    ),
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg for cfg in (RUNNER_HY_WORLDPLAY_WAN_I2V_5B,)
}
"""Shipped HY-WorldPlay runner configs keyed by ``runner_name``."""
