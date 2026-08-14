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

"""Probe: does a cloned MOVING track materialize more solidly than a parked one?

Parked-template clones materialize at only ~20% render strength (box
darkening 9-10 uint8 vs ~52 for a scene-native car). Hypothesis: a parked
clone is pinned against contradicting history at one spot (the first-frame
photo and every prior render show that curb empty), while recorded moving
traffic materializes solidly from boxes alone — the object is never at odds
with the same pixels for long. This probe clones a real MOVING car track
(drift >= ``MIN_DRIFT_M``) and rigidly shifts it along the ego heading so it
repeats its recorded motion offset in space.

Env knobs: ``SHIFT_FWD_M`` (default 25), ``MTEMPLATE_IDX`` (default 0 =
nearest at window start), ``MIN_DRIFT_M`` (default 15), ``GUIDE_SCALE``
(default 0 = off; >0 adds the box-axis guidance combine over the clone,
same recipe as ``probe_spawn_guidance``), ``HDMAP_ONLY``, ``N_CHUNKS``,
``OUT_DIR``.

Run from the repo root (venv bin on PATH for the Ludus ninja build)::

    HDMAP_ONLY=1 OUT_DIR=.../mclone_hdmap python probe_moving_clone.py
    OUT_DIR=.../mclone                    python probe_moving_clone.py
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

_EAGER_NAME = "omnidreams-sv-2steps-chunk2-moving-clone-probe"
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
SHIFT_FWD_M = float(os.environ.get("SHIFT_FWD_M", "25"))
MTEMPLATE_IDX = int(os.environ.get("MTEMPLATE_IDX", "0"))
MIN_DRIFT_M = float(os.environ.get("MIN_DRIFT_M", "15"))
GUIDE_SCALE = float(os.environ.get("GUIDE_SCALE", "0"))
HDMAP_ONLY = os.environ.get("HDMAP_ONLY", "0") == "1"
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/scripts/outputs/mclone")
)


def _extract_moving_templates(
    pools, *, ego_pose: np.ndarray, t0_us: int
) -> list[TrackTemplate]:
    """Car-sized tracks that MOVE >= MIN_DRIFT_M and cover the window."""
    origin, forward, left = _ego_frame(ego_pose)
    out: list[tuple[float, TrackTemplate]] = []
    for pool in pools:
        scales = pool.scales.cpu().numpy()
        for track_index, (a, b) in enumerate(_pool_track_slices(pool)):
            ts = pool.track_timestamps_us[a:b].cpu().numpy()
            if len(ts) < 8 or ts[0] > t0_us + 1_000_000 or ts[-1] < t0_us + 5_500_000:
                continue
            length = float(scales[track_index].max())
            if not 3.4 <= length <= 5.6:
                continue
            tr = pool.translations[a:b].cpu().numpy()
            drift = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
            if drift < MIN_DRIFT_M:
                continue
            rel = tr[0, :2] - origin
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
    assert renderer is not None
    pools = list(renderer._base_timestamped_scene.cube_pools or [])
    ego0 = runtime._initial_ego_pose
    assert ego0 is not None
    assert runtime._scene_data is not None
    t0_us = int(runtime._scene_data.ego_poses[0].timestamp)
    origin, forward, left = _ego_frame(ego0)

    templates = _extract_moving_templates(pools, ego_pose=ego0, t0_us=t0_us)
    assert templates, "no moving car-sized tracks cover the rollout window"
    for i, tpl in enumerate(templates[:6]):
        tr = tpl.translations.cpu().numpy()
        rel_end = tr[-1, :2] - origin
        print(
            f"moving template {i}: start fwd {tpl.source_fwd_m:.1f} "
            f"lat {tpl.source_lateral_m:.1f} -> end fwd "
            f"{float(rel_end @ forward):.1f} lat {float(rel_end @ left):.1f} "
            f"({tr.shape[0]} samples, len {float(tpl.scale.max()):.1f} m)",
            flush=True,
        )
    template = templates[MTEMPLATE_IDX % len(templates)]

    # Rigid shift along the ego heading: same recorded motion, offset start.
    target_fwd = template.source_fwd_m + SHIFT_FWD_M
    clone = clone_template_pool(
        [(template, target_fwd, template.source_lateral_m)], ego_pose=ego0
    )
    print(
        f"cloned moving template {MTEMPLATE_IDX % len(templates)} shifted "
        f"+{SHIFT_FWD_M} m fwd: starts fwd {target_fwd:.1f} m, "
        f"lat {template.source_lateral_m:.1f} m",
        flush=True,
    )

    # Route through the standard overlay path: sentinel actor list so the
    # session builds a pool, patched builder returns the clone.
    runtime._spawned_actors = [object()]  # ty: ignore[invalid-assignment]
    webrtc_session.actors_to_cube_pool = (  # ty: ignore[invalid-assignment]
        lambda actors, ts, device: clone
    )

    if GUIDE_SCALE > 0 and not HDMAP_ONLY:
        wrapper = runtime._wrapper
        assert wrapper is not None
        pipe = wrapper.pipeline
        transformer = pipe.diffusion_model.transformer
        encoder = pipe.encoder
        assert encoder is not None
        shadow_encoder_cache = encoder.initialize_autoregressive_cache()
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
                    cache=shadow_encoder_cache,
                )
                state["alt_input"] = transformer.patchify_and_maybe_split_cp(encoded)
            state["ar_idx"] += 1
            return frames_box

        wrapper._render_condition_frames = dual_render  # ty: ignore[invalid-assignment]
        orig_pf = transformer.predict_flow

        def guided_pf(noisy_latent, timestep, cache, input=None):
            if transformer._finalizing_kv_cache or state["alt_input"] is None:
                return orig_pf(noisy_latent, timestep, cache, input=input)
            flow_box = orig_pf(noisy_latent, timestep, cache, input=input)
            flow_nobox = orig_pf(
                noisy_latent, timestep, cache, input=state["alt_input"]
            )
            return flow_nobox + GUIDE_SCALE * (flow_box - flow_nobox)

        transformer.predict_flow = guided_pf  # ty: ignore[invalid-assignment]
        print(f"box-axis guidance active at s={GUIDE_SCALE}", flush=True)

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
