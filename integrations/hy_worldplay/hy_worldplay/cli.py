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

"""Standalone CLI for the HY-WorldPlay WAN-5B I2V runner.

Phase-1 entry point. Once the recipe-level integration lands (see
integration ``README.md``), the same config will be served as a
``flashdreams-run hy-worldplay-wan-i2v-5b`` subcommand and this
module can be removed.

Usage::

    # single-GPU
    uv run python -m hy_worldplay.cli \\
        --image-path /path/to/first_frame.jpg \\
        --ar-model-path /path/to/wan_transformer \\
        --ckpt-path /path/to/wan_distilled_model/model.pt \\
        --hy-worldplay-repo-root /path/to/HY-WorldPlay

    # 4 GPUs (matches HY-WorldPlay/wan/README.md)
    torchrun --nproc_per_node=4 --no-python --module hy_worldplay.cli \\
        --image-path ... --ar-model-path ... --ckpt-path ... \\
        --hy-worldplay-repo-root ... --num-chunk 4 --pose "w-16"
"""

from __future__ import annotations

import os

from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig


def main(config: HyWorldPlayWanI2VRunnerConfig) -> None:
    """Print the resolved config and run."""
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Resolved config for {config.runner_name!r}:")
        print(config)
    runner = config.setup()
    runner.run()


def entrypoint() -> None:
    # Deferred so ``import hy_worldplay.cli`` works without tyro
    # installed (CPU-only smoke tests).
    import tyro

    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    cfg = tyro.cli(
        HyWorldPlayWanI2VRunnerConfig,
        prog="python -m hy_worldplay.cli",
        description=__doc__,
        console_outputs=is_rank_zero,
    )
    main(cfg)


if __name__ == "__main__":
    entrypoint()
