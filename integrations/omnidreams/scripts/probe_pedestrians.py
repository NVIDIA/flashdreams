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
``N_CHUNKS``, ``OUT_DIR``, ``DEBUG_TRACKS``, ``PED_WALK_SPEED`` (m/s toward
the ego: a linear drift along -forward added to each clone's per-frame
translations, so the boxes TRANSLATE at walking speed instead of holding
the source track's — often near-static — trajectory).

Run from the repo root (venv bin on PATH for the Ludus ninja build)::

    HDMAP_ONLY=1 PED_COUNT=5 OUT_DIR=.../ped5_hdmap python probe_pedestrians.py
    PED_COUNT=5              OUT_DIR=.../ped5       python probe_pedestrians.py
"""

from __future__ import annotations

import os
from dataclasses import replace
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

TEACHER = os.environ.get("TEACHER", "0") == "1"

if TEACHER:
    # 35-step bidirectional teacher (chunk48), light TAE decoder — the
    # door-closer arm: does the teacher materialize mid-road pedestrian
    # clones under a scene-class prompt where the student refuses?
    from pathlib import Path as _P

    from omnidreams.config import (
        SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
    )
    from omnidreams.transformer.impl.network import (
        CosmosDiTNetwork,
        CosmosDiTNetworkConfig,
    )

    def _teacher_key_filter():
        with torch.device("meta"):
            ref = CosmosDiTNetwork(CosmosDiTNetworkConfig(additional_concat_ch=16))
        valid = set(ref.state_dict().keys())
        del ref

        def transform(sd):
            out = {
                (k[4:] if k.startswith("net.") else k): v
                for k, v in sd.items()
                if (k[4:] if k.startswith("net.") else k) in valid
            }
            assert len(out) == len(valid)
            return out

        return transform

    _EAGER_NAME = "omnidreams-sv-teacher-ped-probe"
    OMNIDREAMS_CONFIGS[_EAGER_NAME] = derive_config(
        SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        name=_EAGER_NAME,
        enable_sync_and_profile=False,
        decoder=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE.decoder,
        diffusion_model=dict(
            seed=42,
            transformer=dict(
                checkpoint_path=str(
                    _P.home()
                    / ".cache/huggingface/hub/models--nvidia--omni-dreams-models"
                    / "snapshots/253701787e2f99efec31aaab665d0d9e0cc1eb4a"
                    / "single_view/teacher/3b4c21d0-7b77-4694-9d9d-6ac9b6dbba51_model.pt"
                ),
                state_dict_transform=_teacher_key_filter(),
                compile_network=False,
                use_cuda_graph=False,
            ),
        ),
    )
else:
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
PED_WALK_SPEED = float(os.environ.get("PED_WALK_SPEED", "0"))
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

    def _walking(template: TrackTemplate, ego_pose: np.ndarray) -> TrackTemplate:
        """Drift the clone's translations along -forward at PED_WALK_SPEED."""
        if PED_WALK_SPEED == 0.0:
            return template
        _, forward, _ = _ego_frame(ego_pose)
        ts = template.timestamps_us
        dt_s = (ts - ts[0]).double() / 1e6
        moved = template.translations.clone()
        drift = (-PED_WALK_SPEED * dt_s)[:, None] * torch.as_tensor(
            forward, dtype=moved.dtype, device=moved.device
        )
        moved[:, :2] += drift.to(moved.dtype)
        return replace(template, translations=moved)

    # Deterministic grid: rows every 6 m, columns across the lane.
    def _grid(ego_pose):
        out = []
        j = 0
        while len(out) < PED_COUNT:
            fw = FWD0 + 6.0 * (j // len(_LATERALS))
            out.append(
                (
                    _walking(templates[j % len(templates)], ego_pose),
                    fw,
                    _LATERALS[j % len(_LATERALS)],
                )
            )
            j += 1
        return clone_template_pool(out, ego_pose=ego_pose)

    runtime._spawned_actors = [object()]  # ty: ignore[invalid-assignment]
    if os.environ.get("CROWD_FOLLOW") == "1":
        # Rebuild the grid ahead of the CURRENT ego each chunk: a driving ego
        # keeps the crowd in front (a static grid is driven past in ~6 s).
        # This is the conditioning mode for drive-through-crowd SFT data.
        def _dyn_pool(actors, ts, device):
            ego = runtime._last_ego_pose
            return _grid(ego if ego is not None else ego0)

        webrtc_session.actors_to_cube_pool = _dyn_pool  # ty: ignore[invalid-assignment]
        print(f"crowd-follow: {PED_COUNT} clones re-anchored per chunk", flush=True)
    else:
        clone = _grid(ego0)
        print(f"placed {PED_COUNT} pedestrian clones from fwd {FWD0} m", flush=True)
        webrtc_session.actors_to_cube_pool = (  # ty: ignore[invalid-assignment]
            lambda actors, ts, device: clone
        )

    GUIDE_SCALE = float(os.environ.get("GUIDE_SCALE", "0"))
    if GUIDE_SCALE > 0 and not HDMAP_ONLY:
        wrapper = runtime._wrapper
        assert wrapper is not None
        pipe = wrapper.pipeline
        transformer = pipe.diffusion_model.transformer
        encoder = pipe.encoder
        assert encoder is not None
        shadow_cache = encoder.initialize_autoregressive_cache()
        state: dict = {"alt_input": None, "ar_idx": 0}
        orig_render = wrapper._render_condition_frames

        def dual_render(renderer, camera_names, poses, timestamps, pool=None):
            frames_box = orig_render(renderer, camera_names, poses, timestamps, pool)
            frames_nobox = orig_render(renderer, camera_names, poses, timestamps, None)
            with torch.no_grad():
                model_in = wrapper._to_model_range(
                    wrapper._normalize_condition_input(frames_nobox)
                )
                encoded = encoder(
                    input=model_in,
                    autoregressive_index=state["ar_idx"],
                    cache=shadow_cache,
                )
                state["alt_input"] = transformer.patchify_and_maybe_split_cp(encoded)
            state["ar_idx"] += 1
            return frames_box

        wrapper._render_condition_frames = (  # ty: ignore[invalid-assignment]
            dual_render
        )
        orig_pf = transformer.predict_flow

        def guided_pf(noisy_latent, timestep, cache, input=None):
            if transformer._finalizing_kv_cache or state["alt_input"] is None:
                return orig_pf(noisy_latent, timestep, cache, input=input)
            flow_box = orig_pf(noisy_latent, timestep, cache, input=input)
            flow_nobox = orig_pf(
                noisy_latent, timestep, cache, input=state["alt_input"]
            )
            return flow_nobox + GUIDE_SCALE * (flow_box - flow_nobox)

        transformer.predict_flow = guided_pf
        print(f"box-axis guidance active at s={GUIDE_SCALE}", flush=True)

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
        keys = frozenset() if os.environ.get("EGO_STOP") == "1" else frozenset({"w"})
        segments = [(t, t_end, keys)]
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
