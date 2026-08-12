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

"""Smoke test: user-spawned actors in a headless WebRTC-runtime drive.

Drives the Omnidreams WebRTC runtime synchronously (no browser, no
networking): hold W, spawn a car ahead mid-drive via the same
``/spawn`` command the datachannel uses, spawn a cone later, and save the
rollout. Verifies the full chain scene -> Ludus bbox render -> HDMap
conditioning -> model materializes an object.

Env knobs: ``N_CHUNKS``, ``SPAWN_AT``, ``SPAWN_CMD``, ``SPAWN2_AT``,
``SPAWN2_CMD``, ``EDIT_PROMPT`` (optional prompt swap alongside the first
spawn), ``HDMAP_ONLY=1`` (skip the model, save the rendered conditioning —
fast check that the bbox actually lands in the HDMap stream), ``OUT_DIR``.

Run from the repo root::

    .venv/bin/python integrations/omnidreams/scripts/smoke_spawn_actor.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner_io import write_video_tensor

# Register an eager variant before the runtime resolves the name: probing
# scripts skip compile / CUDA graphs to trade steady-state latency for
# startup time.
_EAGER_NAME = "omnidreams-sv-2steps-chunk2-smoke-eager"
OMNIDREAMS_CONFIGS[_EAGER_NAME] = derive_config(
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    name=_EAGER_NAME,
    enable_sync_and_profile=False,
    diffusion_model=dict(
        seed=42,
        transformer=dict(compile_network=False, use_cuda_graph=False),
    ),
)

from omnidreams.webrtc.session import (  # noqa: E402  (needs the config registered)
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
)

FPS = 30
N_CHUNKS = int(os.environ.get("N_CHUNKS", "24"))
SPAWN_AT = int(os.environ.get("SPAWN_AT", "6"))
SPAWN_CMD = os.environ.get("SPAWN_CMD", "/spawn car 16 0 0")
SPAWN2_AT = int(os.environ.get("SPAWN2_AT", "14"))
SPAWN2_CMD = os.environ.get("SPAWN2_CMD", "/spawn cone 10 0 -2")
EDIT_PROMPT = os.environ.get("EDIT_PROMPT", "")
HDMAP_ONLY = os.environ.get("HDMAP_ONLY", "0") == "1"
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/scripts/outputs/spawn_smoke")
)


def main() -> None:
    config = OmnidreamsRuntimeConfig(
        pipeline_config_name=_EAGER_NAME,
        debug_serve_hdmaps=HDMAP_ONLY,
    )
    runtime = OmnidreamsInferenceRuntime(config)
    print("initializing runtime (scene + pipeline)...", flush=True)
    runtime._initialize_sync()

    chunks: list[torch.Tensor] = []
    t = 0.0
    for ar_idx in range(N_CHUNKS):
        if ar_idx == SPAWN_AT:
            print(runtime._trigger_event_sync(event_id=SPAWN_CMD, state="trigger"))
            if EDIT_PROMPT:
                print(
                    runtime._trigger_event_sync(event_id=EDIT_PROMPT, state="trigger")
                )
        if ar_idx == SPAWN2_AT:
            print(runtime._trigger_event_sync(event_id=SPAWN2_CMD, state="trigger"))

        num_frames = runtime.peek_next_chunk_num_frames()
        t_end = t + num_frames / FPS
        segments = [(t, t_end, frozenset({"w"}))]  # hold W: drive forward
        frame_times = [t + i / FPS for i in range(num_frames)]
        result = runtime._generate_one_chunk_sync(
            segments=segments, frame_times=frame_times
        )
        chunks.append(result.video_chunk[0, 0])  # [T, 3, H, W] uint8
        t = t_end
        if ar_idx % 4 == 0:
            print(f"chunk {ar_idx} done", flush=True)

    video = torch.cat(chunks, dim=0).float() / 127.5 - 1.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "hdmap.mp4" if HDMAP_ONLY else "drive.mp4"
    write_video_tensor(video, OUT_DIR / name, fps=FPS, layout="tchw")
    print(
        f"{video.shape[0]} frames -> {OUT_DIR / name} "
        f"(spawn at chunk {SPAWN_AT}: {SPAWN_CMD!r}; "
        f"chunk {SPAWN2_AT}: {SPAWN2_CMD!r})"
    )


if __name__ == "__main__":
    main()
