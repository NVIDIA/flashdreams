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

"""Multipart test client for the queued Omnidreams video service."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path

from aiohttp import ClientSession, FormData

DEFAULT_GENERATE_URL = "http://127.0.0.1:8090/generate"
"""Default local service endpoint."""


async def submit_request(
    *,
    url: str,
    prompt: str,
    first_frame: Path,
    hdmap_videos: tuple[Path, ...],
    output: Path,
    runner_name: str | None = None,
) -> None:
    """Upload inputs to the service and write the returned MP4."""
    with ExitStack() as stack:
        form = FormData()
        form.add_field("prompt", prompt)
        if runner_name:
            form.add_field("runner_name", runner_name)
        form.add_field(
            "first_frame",
            stack.enter_context(first_frame.open("rb")),
            filename=first_frame.name,
            content_type="image/png",
        )
        for video in hdmap_videos:
            form.add_field(
                "hdmap_video",
                stack.enter_context(video.open("rb")),
                filename=video.name,
                content_type="video/mp4",
            )

        async with ClientSession() as session:
            async with session.post(url, data=form) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"service returned HTTP {response.status}: {body}")
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        f.write(chunk)
                job_id = response.headers.get("X-Job-Id", "")
                total_blocks = response.headers.get("X-Total-Blocks", "")
                print(f"wrote {output} (job_id={job_id}, total_blocks={total_blocks})")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_GENERATE_URL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument(
        "--hdmap-video",
        type=Path,
        action="append",
        required=True,
        help=(
            "HDMap conditioning MP4. Repeat for per-view inputs; a single "
            "upload is reused for every view by the service."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("omnidreams-output.mp4"))
    parser.add_argument("--runner-name", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line upload client."""
    args = _parse_args(argv)
    try:
        asyncio.run(
            submit_request(
                url=args.url,
                prompt=args.prompt,
                first_frame=args.first_frame,
                hdmap_videos=tuple(args.hdmap_video),
                output=args.output,
                runner_name=args.runner_name,
            )
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

