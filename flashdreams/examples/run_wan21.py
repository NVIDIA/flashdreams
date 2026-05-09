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

"""Wan 2.1 demo entry point.

The body lives on the runners in :mod:`flashdreams.recipes.wan.runner`;
this script is now a thin wrapper that selects T2V vs I2V from the
presence of ``--image_path`` and dispatches to ``runner.run()``. The
preferred entry point going forward is ``flashdreams-run``::

    flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."
    flashdreams-run wan21-i2v-14b-480p --prompt "..." --image_path frame.png

This script is kept as a discoverability anchor for users who land on
``examples/`` from the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from flashdreams.infra.config import derive_config
from flashdreams.recipes.wan.runner import (
    WAN21_I2V_14B_480P_RUNNER,
    WAN21_T2V_1PT3B_480P_RUNNER,
    Wan21I2VRunnerConfig,
    Wan21T2VRunnerConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wan 2.1 demo. Without --image_path runs T2V (1.3B); "
            "with --image_path runs I2V (14B 480P). For the unified "
            "CLI surface use ``flashdreams-run wan21-t2v-1.3b-480p`` / "
            "``flashdreams-run wan21-i2v-14b-480p``."
        )
    )
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    common = {
        k: v
        for k, v in dict(
            prompt=args.prompt,
            pixel_height=args.height,
            pixel_width=args.width,
        ).items()
        if v is not None
    }

    if args.image_path is None:
        config = cast(
            Wan21T2VRunnerConfig,
            derive_config(WAN21_T2V_1PT3B_480P_RUNNER, **common),
        )
    else:
        config = cast(
            Wan21I2VRunnerConfig,
            derive_config(
                WAN21_I2V_14B_480P_RUNNER,
                image_path=Path(args.image_path),
                **common,
            ),
        )

    config.setup().run()


if __name__ == "__main__":
    main()
