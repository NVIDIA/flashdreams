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

"""Configs for the HY-WorldPlay integration.

The phase-1 plugin only ships the WAN-5B I2V variant, since the WAN
backbone is the only one that maps cleanly onto flashdreams' existing
Wan recipe family. The HunyuanVideo-1.5 8B variant
(``hyvideo/generate.py`` upstream) is a much heavier integration --
multiple text encoders (Qwen2.5-VL-7B, ByT5, Glyph-SDXL-v2), gated
vision encoder (FLUX.1-Redux-dev), 8-way SP -- and is tracked as a
follow-up. See the integration ``README.md`` for the staging plan.

The runner config is registered with ``flashdreams-run`` via the
``flashdreams.runner_configs`` entry-point group declared in this
package's ``pyproject.toml``; the registry key always comes from
``cfg.runner_name``.
"""

from __future__ import annotations

from flashdreams.infra.runner import RunnerConfig
from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

# Default literal: every field at its upstream-matching default. Users
# *must* override at least ``ar_model_path`` / ``ckpt_path`` /
# ``hy_worldplay_repo_root`` (and ``image_path``) at runtime; we don't
# bake in real paths here because they're machine-specific.
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
"""All shipped HY-WorldPlay runners, keyed by ``runner_name``."""
