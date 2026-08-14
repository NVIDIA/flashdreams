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

"""Probe: pedestrian-clone density ladder (1 / 5 / 20 ahead of the ego).

Car-sized template clones materialize (ghost-to-solid depending on motion
and guidance); this probe asks whether PEDESTRIAN tracks cloned from the
scene's own pools materialize at all, and where the density ceiling is —
a dense crowd on a residential road is far off the AV training manifold,
so the expectation is a few near materializations and mush beyond.

Placements form a deterministic grid ahead of the ego (rows every 6 m
from ``FWD0``, lateral spread across the lane), cycling the available
pedestrian templates. Masks come from the HDMAP arm as usual.

Env knobs: ``PED_COUNT`` (default 5), ``FWD0`` (default 22), ``HDMAP_ONLY``,
``N_CHUNKS``, ``OUT_DIR``, ``DEBUG_TRACKS``.

Run from the repo root (venv bin on PATH for the Ludus ninja build)::

    HDMAP_ONLY=1 PED_COUNT=5 OUT_DIR=.../ped5_hdmap python probe_pedestrians.py
    PED_COUNT=5              OUT_DIR=.../ped5       python probe_pedestrians.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
)
from omnidreams.webrtc.actors import (
    TrackTemplate,
    _ego_frame,
    _pool_track_slices,
    clone_template_pool,
)

from flashdreams.infra.config import derive_config

_EAGER_NAME = "omnidreams-sv-2steps-chunk2-ped-probe"
OMNIDREAMS_CONFIGS[_EAGER_NAME] = derive_config(
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    name=_EAGER_NAME,
    enable_sync_and_profile=False,
    diffusion_model=dict(
        seed=42,
        transformer=dict(compile_network=False, use_cuda_graph=False),
    ),
)

from flashdreams.infra.runner_io import write_video_tensor  # noqa: E402
from omnidreams.webrtc import session as webrtc_session  # noqa: E402
from omnidreams.webrtc.session import (  # noqa: E402
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
)


FPS = 30
N_CHUNKS = int(os.environ.get("N_CHUNKS", "26"))
PED_COUNT = int(os.environ.get("PED_COUNT", "5"))
FWD0 = float(os.environ.get("FWD0", "22"))
HDMAP_ONLY = os.environ.get("HDMAP_ONLY", "0") == "1"
DEBUG_TRACKS = os.environ.get("DEBUG_TRACKS", "0") == "1"
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/scripts/outputs/ped_probe")
)

_LATERALS = tuple(
    float(x) for x in os.environ.get("PED_LATERALS", "-3.2,-1.1,1.1,3.2").split(",")
)
"""Lateral columns of the placement grid (default: across the ego lane;
override with PED_LATERALS for sidewalk-band placements)."""


def _extract_ped_templates(
    pools, *, ego_pose: np.ndarray, t0_us: int
) -> list[TrackTemplate]:
    """Pedestrian-sized tracks covering the early window (walking allowed)."""
    origin, forward, left = _ego_frame(ego_pose)
    out: list[tuple[float, TrackTemplate]] = []
    for pool_index, pool in enumerate(pools):
        scales = pool.scales.cpu().numpy()
        for track_index, (a, b) in enumerate(_pool_track_slices(pool)):
            ts = pool.track_timestamps_us[a:b].cpu().numpy()
            length = float(scales[track_index].max())
            if DEBUG_TRACKS and length <= 2.2:
                tr0 = pool.translations[a].cpu().numpy()[:2] - origin
                print(
                    f"pool{pool_index} track{track_index}: n={b - a} "
                    f"len={length:.2f} t=[{(ts[0] - t0_us) / 1e6:.1f},"
                    f"{(ts[-1] - t0_us) / 1e6:.1f}]s "
                    f"fwd={float(tr0 @ forward):.1f} lat={float(tr0 @ left):.1f}",
                    flush=True,
                )
            if len(ts) < 6 or ts[0] > t0_us + 1_500_000 or ts[-1] < t0_us + 3_500_000:
                continue
            # Person-sized: tallest dim is the ~1.7 m height (scale.max()
            # is NOT the footprint), the other dims are sub-metre.
            dims = np.sort(scales[track_index])
            if not (1.2 <= dims[-1] <= 2.1 and dims[-2] <= 1.2):
                continue
            rel = pool.translations[a].cpu().numpy()[:2] - origin
            template = TrackTemplate(
                timestamps_us=pool.track_timestamps_us[a:b].clone(),
                translations=pool.translations[a:b].clone(),
                quaternions=pool.quaternions[a:b].clone(),
                scale=pool.scales[track_index : track_index + 1].clone(),
                colors=pool.colors[track_index : track_index + 1].clone(),
                prim_type_id=pool.prim_type_id,
                render_flags=pool.render_flags,
                source_fwd_m=float(rel @ forward),
                source_lateral_m=float(rel @ left),
            )
            out.append((float(np.linalg.norm(rel)), template))
    out.sort(key=lambda item: item[0])
    return [t for _, t in out]


def main() -> None:
    config = OmnidreamsRuntimeConfig(
        pipeline_config_name=_EAGER_NAME, debug_serve_hdmaps=HDMAP_ONLY
    )
    runtime = OmnidreamsInferenceRuntime(config)
    print("initializing runtime (scene + pipeline)...", flush=True)
    runtime._initialize_sync()

    renderer = runtime._renderer
    assert renderer is not None and runtime._initial_ego_pose is not None
    pools = list(renderer._base_timestamped_scene.cube_pools or [])
    ego0 = runtime._initial_ego_pose
    assert runtime._scene_data is not None
    t0_us = int(runtime._scene_data.ego_poses[0].timestamp)

    templates = _extract_ped_templates(pools, ego_pose=ego0, t0_us=t0_us)
    assert templates, "no pedestrian-sized tracks cover the window (try DEBUG_TRACKS=1)"
    print(
        f"{len(templates)} pedestrian templates; nearest at "
        f"fwd {templates[0].source_fwd_m:.1f} lat {templates[0].source_lateral_m:.1f}",
        flush=True,
    )

    # Deterministic grid: rows every 6 m, columns across the lane.
    placements = []
    i = 0
    while len(placements) < PED_COUNT:
        fwd = FWD0 + 6.0 * (i // len(_LATERALS))
        lateral = _LATERALS[i % len(_LATERALS)]
        placements.append((templates[i % len(templates)], fwd, lateral))
        i += 1
    clone = clone_template_pool(placements, ego_pose=ego0)
    print(f"placed {len(placements)} pedestrian clones from fwd {FWD0} m", flush=True)

    runtime._spawned_actors = [object()]  # ty: ignore[invalid-assignment]
    webrtc_session.actors_to_cube_pool = (  # ty: ignore[invalid-assignment]
        lambda actors, ts, device: clone
    )

    if os.environ.get("EDIT_PROMPT"):
        # Scene-class synergy: align the text channel with the box channel
        # (e.g. a street-festival prompt to make mid-road crowds plausible).
        print(
            runtime._trigger_event_sync(
                event_id=os.environ["EDIT_PROMPT"], state="trigger"
            ),
            flush=True,
        )

    chunks: list[torch.Tensor] = []
    t = 0.0
    for ar_idx in range(N_CHUNKS):
        num_frames = runtime.peek_next_chunk_num_frames()
        t_end = t + num_frames / FPS
        segments = [(t, t_end, frozenset({"w"}))]
        frame_times = [t + i / FPS for i in range(num_frames)]
        result = runtime._generate_one_chunk_sync(
            segments=segments, frame_times=frame_times
        )
        chunks.append(result.video_chunk[0, 0])
        t = t_end
        if ar_idx % 4 == 0:
            print(f"chunk {ar_idx} done", flush=True)

    video = torch.cat(chunks, dim=0).float() / 127.5 - 1.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "hdmap.mp4" if HDMAP_ONLY else "drive.mp4"
    write_video_tensor(video, OUT_DIR / name, fps=FPS, layout="tchw")
    print(f"{video.shape[0]} frames -> {OUT_DIR / name}")


if __name__ == "__main__":
    main()
