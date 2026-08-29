# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-side guidance for track-backed obstacle events."""

from __future__ import annotations

from typing import Any

from loguru import logger

from crazy_robotaxi.live_edit.obstacle_events import (
    OBSTACLE_ENTITY_PREFIX,
    ObstacleAbility,
    ObstacleEvent,
    ObstaclePhase,
    ObstacleTemplate,
    ObstacleTemplateCatalog,
    build_obstacle_event,
    local_ground_z,
    road_ahead_pose,
)

## Box-axis guidance (model side, GPU only)


class ObstacleGuidance:
    """Guide the flow along the with-box/without-box conditioning axis.

    Render the conditioning twice (with and without the obstacle event), encode the no-box branch
    through a shadow encoder cache whose temporal state tracks the real one
    from chunk 0, and combine ``flow_nobox + s * (flow_box - flow_nobox)``
    per denoising step while an event is on screen. Costs one extra lightVAE
    encode per chunk always, plus one raster and one extra network forward
    per step during events.
    """

    def __init__(self, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError("ObstacleGuidance requires a positive scale")
        self._scale = float(scale)
        self._alt_frames: list[Any] | None = None
        self._alt_input: Any | None = None
        self._shadow_cache: Any | None = None
        self._ar_index = 0

    def install(self, backend: Any) -> None:
        """Hook the warmed backend's raster and session seams."""
        session = backend._session
        self._guard_transformer(session)
        rasterizer = backend._rasterizer

        original_first = backend.render_first_chunk
        original_next = backend.render_next_chunk

        def render_first_chunk(trajectory: Any) -> Any:
            self._stash_alt_frames(rasterizer, trajectory)
            return original_first(trajectory)

        def render_next_chunk(trajectory: Any) -> Any:
            self._stash_alt_frames(rasterizer, trajectory)
            return original_next(trajectory)

        backend.render_first_chunk = render_first_chunk
        backend.render_next_chunk = render_next_chunk

        original_start = session.start
        original_continue = session.continue_generation

        def start(initial_rgb: Any, condition_frames: Any, prompt: str) -> Any:
            self._reset_shadow(session)
            self._encode_shadow(session, condition_frames)
            return original_start(initial_rgb, condition_frames, prompt)

        def continue_generation(condition_frames: Any) -> Any:
            self._encode_shadow(session, condition_frames)
            return original_continue(condition_frames)

        session.start = start
        session.continue_generation = continue_generation
        self._wrap_predict_flow(session)
        logger.info(f"[live-edit] obstacle box-axis guidance armed s={self._scale}")

    def install_v2(self, pipeline: Any) -> None:
        """Attach guidance directly to an API-v2 OmniDreams pipeline."""
        transformer = pipeline.diffusion_model.transformer
        if getattr(transformer, "_optimized_dit_executor", None) is not None:
            raise RuntimeError(
                "Obstacle guidance requires native_dit_acceleration='disabled'"
            )
        self._shadow_cache = pipeline.encoder.initialize_autoregressive_cache()
        original_predict_flow = transformer.predict_flow

        def guided_predict_flow(
            noisy_latent: Any, timestep: Any, cache: Any, input: Any = None
        ) -> Any:
            alt = self._alt_input
            if alt is None or transformer._finalizing_kv_cache:
                return original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_box = original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_nobox = original_predict_flow(noisy_latent, timestep, cache, input=alt)
            return flow_nobox + self._scale * (flow_box - flow_nobox)

        transformer.predict_flow = guided_predict_flow
        logger.info(f"[live-edit] V2 obstacle guidance armed s={self._scale}")

    def reset_v2(self, pipeline: Any) -> None:
        """Reset the shadow encoder cache without reinstalling model hooks."""
        self._shadow_cache = pipeline.encoder.initialize_autoregressive_cache()
        self._alt_frames = None
        self._alt_input = None
        self._ar_index = 0

    def prepare_v2(
        self,
        pipeline: Any,
        autoregressive_index: int,
        hdmap: Any,
        alternate_hdmap: Any | None,
    ) -> None:
        """Advance the shadow encoder and publish obstacle-free conditioning."""
        import torch

        from flashdreams.core.distributed.context_parallel import split_inputs_cp

        source = hdmap if alternate_hdmap is None else alternate_hdmap
        source = split_inputs_cp(source, seq_dim=1, cp_group=pipeline.V_group)
        with torch.no_grad(), _eager_vae_scope(pipeline.encoder):
            encoded = pipeline.encoder(
                input=source,
                autoregressive_index=autoregressive_index,
                cache=self._shadow_cache,
            )
        transformer = pipeline.diffusion_model.transformer
        self._alt_input = (
            transformer.patchify_and_maybe_split_cp(encoded)
            if alternate_hdmap is not None
            else None
        )

    def _stash_alt_frames(self, rasterizer: Any, trajectory: Any) -> None:
        """Render the obstacle-free conditioning when an event is present."""
        actors = trajectory.dynamic_actors
        others = tuple(
            actor
            for actor in actors
            if not actor.entity_id.startswith(OBSTACLE_ENTITY_PREFIX)
        )
        if len(others) == len(actors):
            self._alt_frames = None
            return
        chunk = rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
            dynamic_actors=others,
        )
        self._alt_frames = [frame.rgb_host_uint8 for frame in chunk.frames]

    def _reset_shadow(self, session: Any) -> None:
        self._shadow_cache = session.pipeline.encoder.initialize_autoregressive_cache()
        self._ar_index = 0
        self._alt_frames = None
        self._alt_input = None

    def _encode_shadow(self, session: Any, condition_frames: Any) -> None:
        """Advance the shadow encoder; publish the patchified no-box input.

        Runs every chunk (with identical conditioning when no event is
        active) so the shadow cache's temporal state matches the real
        encoder's — an event can then start mid-run without a history
        mismatch between the two branches.

        The encode runs EAGERLY (:func:`_eager_vae_scope`): the encoder's
        CUDA-graph wrapper captures against one streaming cache's buffer
        addresses, so a captured replay fed the shadow cache would silently
        operate on the real cache's state. The eager shadow encode also
        keeps the wrapper's warmup/capture stream fed by the real cache
        only, so the real branch captures correctly.
        """
        import torch

        from flashdreams.core.distributed.context_parallel import split_inputs_cp

        pipeline = session.pipeline
        if self._shadow_cache is None:
            self._reset_shadow(session)
        frames = self._alt_frames if self._alt_frames is not None else condition_frames
        with torch.no_grad():
            hdmap = session._condition_tensor(frames)
            hdmap = split_inputs_cp(hdmap, seq_dim=1, cp_group=pipeline.V_group)
            with _eager_vae_scope(pipeline.encoder):
                encoded = pipeline.encoder(
                    input=hdmap,
                    autoregressive_index=self._ar_index,
                    cache=self._shadow_cache,
                )
            transformer = pipeline.diffusion_model.transformer
            self._alt_input = (
                transformer.patchify_and_maybe_split_cp(encoded)
                if self._alt_frames is not None
                else None
            )
        self._ar_index += 1

    def _wrap_predict_flow(self, session: Any) -> None:
        transformer = session.pipeline.diffusion_model.transformer
        original_predict_flow = transformer.predict_flow

        def guided_predict_flow(
            noisy_latent: Any, timestep: Any, cache: Any, input: Any = None
        ) -> Any:
            alt = self._alt_input
            if alt is None or transformer._finalizing_kv_cache:
                return original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_box = original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_nobox = original_predict_flow(noisy_latent, timestep, cache, input=alt)
            return flow_nobox + self._scale * (flow_box - flow_nobox)

        transformer.predict_flow = guided_predict_flow

    @staticmethod
    def _guard_transformer(session: Any) -> None:
        """Reject executors the predict_flow dispatch cannot intercept.

        CUDA graphs and ``compile_network`` are fine: the dispatch wraps the
        transformer's eager ``predict_flow`` (outside any capture), and the
        graph wrapper stages the ``hdmap_condition`` kwarg into its static
        buffers per call — the two forwards of a guided step are two replays
        of the same captured graph with different conditioning staged in.
        The native optimized-DiT executor is the one seam that bypasses the
        Python conditioning path.
        """
        transformer = session.pipeline.diffusion_model.transformer
        if getattr(transformer, "_optimized_dit_executor", None) is not None:
            raise RuntimeError(
                "obstacle guidance is not wired for the native optimized-DiT "
                "executor; set native_dit_acceleration: disabled in the "
                "world-model manifest."
            )


class _eager_vae_scope:
    """Route a graph-wrapped Wan VAE's calls through its eager encoder.

    The VAE's ``CUDAGraphWrapper`` passes the streaming cache dict through
    verbatim, binding captured kernels to ONE cache's buffer addresses; a
    replay fed a different cache would silently read/write the capture-time
    cache. Flipping ``_use_cuda_graph`` off for the duration makes the
    encode dispatch to the (possibly compiled) eager module with the cache
    that was actually passed. No-op for encoders without the knob (pixel
    shuffle, fakes).
    """

    def __init__(self, encoder: Any) -> None:
        self._vae = getattr(encoder, "vae", None)
        if self._vae is not None and not hasattr(self._vae, "_use_cuda_graph"):
            self._vae = None
        self._saved: bool | None = None

    def __enter__(self) -> None:
        if self._vae is not None:
            self._saved = self._vae._use_cuda_graph
            self._vae._use_cuda_graph = False

    def __exit__(self, *exc: object) -> None:
        if self._vae is not None and self._saved is not None:
            self._vae._use_cuda_graph = self._saved


__all__ = [
    "OBSTACLE_ENTITY_PREFIX",
    "ObstacleAbility",
    "ObstacleEvent",
    "ObstacleGuidance",
    "ObstaclePhase",
    "ObstacleTemplate",
    "ObstacleTemplateCatalog",
    "build_obstacle_event",
    "local_ground_z",
    "road_ahead_pose",
]
