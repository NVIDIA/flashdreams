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

"""Rollout capture and counterfactual x0 probes for the HY-WorldPlay drift corrector."""

from __future__ import annotations

import dataclasses
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must land before the first CUDA allocation: long captures fragment the
# allocator; expandable segments keep the probe phase inside the VRAM share
# left over by co-tenant jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import Tensor

from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21TransformerCache
from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B
from hy_worldplay.runner import (
    HyWorldPlayWanI2VRunner,
    _resolve_prompt,
    _write_mp4,
    preprocess_first_frame,
)

## Runner construction

_POINT_CLOUD_SEED = 0
"""Global-RNG seed used right before ``_bind_memory_config``: the FOV
selector's Monte-Carlo point cloud draws from the global RNG, so pinning it
makes ``memory_frame_indices`` reproducible across processes."""


def build_runner(
    *,
    num_chunk: int,
    pose: str,
    output_dir: Path,
    image_path: Path | None = None,
    compile_network: bool = True,
) -> HyWorldPlayWanI2VRunner:
    """Build the HY-WorldPlay runner with the rollout geometry fixed up front.

    Args:
        num_chunk: AR chunks per rollout; the pose source must cover
            ``num_chunk * 4`` latents.
        pose: Pose-string or trajectory-JSON path (upstream grammar).
        output_dir: Directory the runner (and callers) write outputs into.
        image_path: First-frame override; ``None`` lazy-downloads the
            upstream sample image.
        compile_network: Inductor-compile the DiT. Disable for training,
            where LoRA module surgery + checkpointed autograd must run on
            the raw module.

    Returns:
        Runner with the pipeline built and the checkpoint loaded.
    """
    from flashdreams.infra.config import derive_config

    cfg = dataclasses.replace(
        RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
        example_data=image_path is None,
        image_path=image_path,
        pose=pose,
        num_chunk=num_chunk,
        output_dir=output_dir,
    )
    # Eager VAE: the encoder/decoder CUDAGraphWrapper private pools cost
    # ~35 GiB at 704x1280, which doesn't fit next to co-tenant jobs; these
    # scripts trade decode speed for headroom.
    cfg = derive_config(
        cfg,
        pipeline=dict(
            encoder=dict(encoder=dict(use_cuda_graph=False)),
            decoder=dict(use_cuda_graph=False),
            diffusion_model=dict(transformer=dict(compile_network=compile_network)),
        ),
    )
    runner = cfg.setup()
    assert isinstance(runner, HyWorldPlayWanI2VRunner)
    if runner.config.example_data and runner.config.image_path is None:
        runner.config.image_path = runner._fetch_example_image()
    return runner


## Per-step alpha*(t) gating

GATE_ALPHA = {1000.0: 0.81, 960.0: 0.53, 888.8889: 0.53, 727.2728: 0.58}
"""Unbiased alpha*(t) from the faithful step-0 gate
(``outputs/gate/gate_faithful.json``): the systematic fraction of the
drift-induced error per inference timestep. Gate configs deploy the LoRA at
``alpha*(t) * scale`` — correction strength follows how systematic the
error actually is at each step."""


def parse_gain_token(token: str) -> float | tuple[str, float]:
    """Parse a config gain token.

    ``"0.7"`` -> flat gain ``0.7``; ``"gate"`` -> ``("gate", 1.0)``;
    ``"gate0.5"`` -> ``("gate", 0.5)`` (per-step ``alpha*(t) * 0.5``).
    """
    token = token.strip()
    if token.startswith("gate"):
        return ("gate", float(token[4:]) if len(token) > 4 else 1.0)
    return float(token)


def install_alpha_gate(runner: Any, network: Any, mode: dict) -> None:
    """Wrap ``predict_flow`` so gate configs rescale the LoRA every step.

    When ``mode["gain"]`` is ``("gate", scale)``, each denoise step sets the
    LoRA to ``alpha*(t) * scale`` via nearest-t lookup in
    :data:`GATE_ALPHA`; flat-gain configs pass through untouched (the caller
    sets the scale once per rollout). Per-token timesteps (AR0) include the
    first-frame stabilization value; the max is always the scheduler step.
    """
    from _lora import set_lora_scale

    transformer = runner.pipeline.diffusion_model.transformer
    orig_pf = transformer.predict_flow

    def gated_pf(*args, **kwargs):
        gain = mode["gain"]
        if isinstance(gain, tuple):
            t = float(kwargs["timestep"].reshape(-1).max())
            alpha = min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]
            set_lora_scale(network, alpha * gain[1])
        return orig_pf(*args, **kwargs)

    transformer.predict_flow = gated_pf


## Rollout capture


@dataclass
class ChunkSnapshot:
    """Per-chunk state captured from a rollout, sufficient to replay probes."""

    history: Tensor | None
    """Patchified clean-latent history the chunk was conditioned on
    (``clean_latent_history`` right before this chunk's ``generate``);
    ``None`` for chunk 0."""

    clean_latent: Tensor
    """Patchified x0 the sampler produced for this chunk."""

    ctrl: HyWorldPlayCtrl
    """Patchified per-AR-step control payload (action / camera / memory
    indices) exactly as ``predict_flow`` consumed it."""

    def to(self, device: torch.device | str) -> "ChunkSnapshot":
        """Return a copy with every tensor (including ctrl fields) on ``device``."""
        moved = {}
        for f in dataclasses.fields(self.ctrl):
            v = getattr(self.ctrl, f.name)
            moved[f.name] = v.to(device) if isinstance(v, Tensor) else v
        return ChunkSnapshot(
            history=None if self.history is None else self.history.to(device),
            clean_latent=self.clean_latent.to(device),
            ctrl=HyWorldPlayCtrl(**moved),
        )


def capture_rollout(
    runner: HyWorldPlayWanI2VRunner,
    *,
    noise_seed: int,
    save_path: Path | None = None,
    mp4_path: Path | None = None,
) -> list[ChunkSnapshot]:
    """Roll out ``num_chunk`` chunks and snapshot the per-chunk corrector inputs.

    Reproduces the runner's ``run()`` flow (bindings included) but records,
    for every chunk, the conditioning history, the produced clean latent,
    and the patchified ctrl payload. The diffusion RNG is re-seeded with
    ``noise_seed`` so captures are reproducible per seed.

    Args:
        runner: Runner from :func:`build_runner`.
        noise_seed: Seed for the diffusion model's noise generator.
        save_path: When set, snapshots are also saved (CPU tensors) here.
        mp4_path: When set, the decoded rollout is written here for eyeballing.

    Returns:
        One :class:`ChunkSnapshot` per chunk, tensors on the compute device.
    """
    pipe = runner.pipeline
    cfg = runner.config
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype

    assert cfg.image_path is not None
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)

    runner._bind_action_labels()
    runner._bind_camera_data()
    torch.manual_seed(_POINT_CLOUD_SEED)
    runner._bind_memory_config(device=device)

    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(noise_seed)

    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)
    snaps: list[ChunkSnapshot] = []
    chunks: list[Tensor] = []
    for ar_idx in range(cfg.num_chunk):
        history = tc.clean_latent_history
        history = None if history is None else history.detach().clone()
        chunk = pipe.generate(ar_idx, cache)
        chunks.append(chunk)
        fs = cache.final_state
        assert fs is not None and isinstance(fs.input, HyWorldPlayCtrl)
        snaps.append(
            ChunkSnapshot(
                history=history,
                clean_latent=fs.clean_latent.detach().clone(),
                ctrl=fs.input,
            )
        )
        pipe.finalize(ar_idx, cache)

    if mp4_path is not None:
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        _write_mp4(torch.cat(chunks, dim=-4), mp4_path, fps=cfg.fps)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"snaps": [s.to("cpu") for s in snaps], "noise_seed": noise_seed},
            save_path,
        )

    # Drop this rollout's caches (rolling KV buffers, VAE stream state)
    # before the caller starts the next one.
    del cache, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return snaps


def load_rollout(path: Path, device: torch.device | str) -> list[ChunkSnapshot]:
    """Load :func:`capture_rollout` snapshots and move them to ``device``."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return [s.to(device) for s in data["snaps"]]


## Counterfactual x0 probes


def start_probe_chunk(
    tc: HyWorldPlayWan21TransformerCache,
    *,
    ar_idx: int,
    history: Tensor,
) -> None:
    """Rewind the AR cache to chunk ``ar_idx`` conditioned on ``history``.

    Substitutes the patchified clean-latent history and resets the rolling
    caches + prefill latch, so the next ``predict_flow`` call re-runs the
    memory KV prefill against the substituted history. Subsequent calls at
    the same chunk reuse the prefilled memory (matching in-chunk sampler
    steps).
    """
    assert ar_idx > 0, "probing requires a non-empty history, so ar_idx >= 1"
    tc.clean_latent_history = history
    tc.start(ar_idx)


def finish_probe_chunk(tc: HyWorldPlayWan21TransformerCache, *, ar_idx: int) -> None:
    """Close the ``start`` / ``finalize`` bracket after a probe sweep.

    ``BlockKVCache`` enforces a strict ``before_update`` / ``after_update``
    alternation; :func:`start_probe_chunk` opened it, so every sweep must
    call this before the next one (the pipeline's ``finalize`` plays this
    role during generation).
    """
    tc.finalize(ar_idx)


def predict_x0(
    transformer: Any,
    tc: HyWorldPlayWan21TransformerCache,
    *,
    ctrl: HyWorldPlayCtrl,
    z_t: Tensor,
    timestep: Tensor,
    sigma: float,
) -> Tensor:
    """Predict x0 at a matched noisy state ``z_t`` under the cache's current history.

    Args:
        transformer: The pipeline's ``HyWorldPlayWan21Transformer``.
        tc: AR cache positioned via :func:`start_probe_chunk`.
        ctrl: The probed chunk's captured (patchified) ctrl payload.
        z_t: Patchified noisy latent ``(1 - sigma) * x0 + sigma * eps``.
        timestep: Scalar timestep tensor in the network dtype.
        sigma: Noise level matching ``timestep`` on the inference schedule.

    Returns:
        fp32 ``x0_hat = z_t - sigma * flow``.
    """
    flow = transformer.predict_flow(
        noisy_latent=z_t, timestep=timestep, cache=tc, input=ctrl
    )
    return z_t.float() - sigma * flow.float()
