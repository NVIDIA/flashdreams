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

"""Command-line report for the automatic native WebM codec policy."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flashdreams_webm.policy import codec_selection


def main(argv: Sequence[str] | None = None) -> None:
    """Run codec selection and emit reproducible JSON and Markdown evidence."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark VP9 at 768x768/24 fps and select VP8 when p90 encode "
            "latency exceeds one frame interval."
        )
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached evidence."
    )
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--markdown-path", type=Path, default=None)
    parsed = parser.parse_args(argv)
    record = codec_selection(refresh=parsed.refresh, cache_path=parsed.cache_path)
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if parsed.json_path is not None:
        parsed.json_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.json_path.write_text(serialized, encoding="utf-8")
    if parsed.markdown_path is not None:
        parsed.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.markdown_path.write_text(_markdown(record), encoding="utf-8")


def _markdown(record: dict[str, Any]) -> str:
    """Return a compact human-readable codec acceptance summary."""
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict):
        return (
            "# FlashDreams native WebM codec selection\n\n"
            f"Selected `{record['codec']}` from `{record['source']}`.\n"
        )
    return "\n".join(
        [
            "# FlashDreams native WebM codec selection",
            "",
            "| Result | Codec | Median | p90 | 24-fps budget |",
            "| --- | --- | ---: | ---: | ---: |",
            (
                f"| {record['status']} | {record['codec']} | "
                f"{benchmark['median_ms']:.3f} ms | {benchmark['p90_ms']:.3f} ms | "
                f"{benchmark['frame_budget_ms']:.3f} ms |"
            ),
            "",
            f"Reason: {record['reason']}",
            "",
            f"Reproduce with `{record['command']}`.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
