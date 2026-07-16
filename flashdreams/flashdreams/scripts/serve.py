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

"""``flashdreams-serve`` CLI for model-slug serving endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from flashdreams.serving.backend import LocalWorkerScheduler
from flashdreams.serving.config import discover_serve_configs
from flashdreams.serving.service import SessionService
from flashdreams.serving.transport import (
    GRPCTransport,
    ServingTransport,
    WebRTCTransport,
    WebSocketTransport,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the central serving command-line parser."""
    parser = argparse.ArgumentParser(
        prog="flashdreams-serve",
        description="Serve one or more registered FlashDreams model slugs.",
    )
    parser.add_argument("models", nargs="*", help="Model slugs to expose.")
    parser.add_argument(
        "--protocol",
        choices=("webrtc", "websocket", "grpc"),
        default="webrtc",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-workers-per-model", type=int, default=1)
    parser.add_argument("--lease-seconds", type=float, default=300.0)
    parser.add_argument(
        "--eager-load",
        action="store_true",
        help="Load and prewarm one worker per selected model before listening.",
    )
    parser.add_argument("--list-models", action="store_true")
    return parser


def make_transport(args: argparse.Namespace) -> ServingTransport:
    """Resolve model slugs and construct the selected serving transport."""
    available = discover_serve_configs()
    selected_slugs = args.models or list(available)
    unknown = sorted(set(selected_slugs) - set(available))
    if unknown:
        raise SystemExit(
            f"Unknown serving model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available)) or '(none)'}"
        )
    selected = {slug: available[slug] for slug in selected_slugs}
    unsupported = [
        slug
        for slug, config in selected.items()
        if args.protocol not in config.descriptor.capabilities.transports
    ]
    if unsupported:
        raise SystemExit(
            f"Protocol {args.protocol!r} is not supported by: "
            f"{', '.join(sorted(unsupported))}."
        )
    scheduler = LocalWorkerScheduler(
        {slug: config.worker_factory for slug, config in selected.items()},
        max_workers_per_model=args.max_workers_per_model,
    )
    service = SessionService(
        {slug: config.descriptor for slug, config in selected.items()},
        scheduler,
        default_lease_seconds=args.lease_seconds,
    )
    transport_types: dict[str, type[ServingTransport]] = {
        "websocket": WebSocketTransport,
        "webrtc": WebRTCTransport,
        "grpc": GRPCTransport,
    }
    return transport_types[args.protocol](service)


async def _serve(transport: ServingTransport, args: argparse.Namespace) -> None:
    """Preload selected models when requested and run the listener."""
    try:
        if args.eager_load:
            model_ids = [str(model["id"]) for model in transport.list_models()]
            print(
                f"Eagerly loading model worker(s): {', '.join(model_ids)}", flush=True
            )
            await transport.preload_models()
            print("Model worker prewarming complete.", flush=True)
        await transport.serve(args.host, args.port)
    finally:
        await transport.service.close()


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the serving CLI and handle interactive shutdown without a traceback."""
    try:
        _entrypoint(argv)
    except KeyboardInterrupt:
        print("FlashDreams server stopped.", flush=True)


def _entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the selected serving transport or list registered models."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        configs = discover_serve_configs()
        print(
            json.dumps(
                [configs[slug].descriptor.to_dict() for slug in sorted(configs)],
                indent=2,
            )
        )
        return
    if not discover_serve_configs():
        parser.error(
            "no serving models are registered; install an integration exposing "
            "flashdreams.serve_configs or set FLASHDREAMS_SERVE_CONFIGS"
        )
    transport = make_transport(args)
    asyncio.run(_serve(transport, args))


if __name__ == "__main__":
    entrypoint()
