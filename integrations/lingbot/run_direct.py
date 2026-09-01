# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the LingBot WebRTC server without going through ``flashdreams-run``.

``flashdreams-run`` builds its CLI parser from the plugin/runner registry via
tyro, which is broken in this environment (empty registry). This calls
:func:`flashdreams.scripts.cli.main` directly with the pre-built
``RUNNER_LINGBOT_WORLD_FAST`` config, the same way ``tests/test_launch.py``
drives it, skipping the registry and tyro parser entirely.
"""

from __future__ import annotations

import argparse
import dataclasses

from flashdreams.scripts.cli import main
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--warmup-chunks", type=int, default=1)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    config = dataclasses.replace(
        RUNNER_LINGBOT_WORLD_FAST,
        device=args.device,
        example_idx=args.example_idx,
    )
    main(
        config,
        mode="webrtc",
        host=args.host,
        port=args.port,
        scenario_overrides={"example_idx": args.example_idx},
        output_overrides={"warmup_chunks": args.warmup_chunks},
    )


if __name__ == "__main__":
    run()
