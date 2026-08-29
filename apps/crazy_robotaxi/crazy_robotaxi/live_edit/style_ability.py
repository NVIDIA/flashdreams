# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live game-skin switching on the flashdreams world-model session.

Ports the attach recipe from
``integrations/omnidreams/scripts/smoke_text_edit.py`` onto
:class:`omnidreams_game_engine.world_model.flashdreams_adapter.FlashdreamsWorldModelSession`:

- a pre-merged :class:`omnidreams.impl._edit_lora.TextEditLoRA` on the
  transformer (zero steady-state cost; ``replace_text`` opens its edit
  window automatically),
- the rank-16 drift corrector. Default (``corrector_mode="fused"``): the
  CUDA-graph-safe per-state
  :class:`omnidreams.impl._drift_corrector.DriftCorrectorDispatch` — one
  pre-merged weight-set family per (base | skin | weather) state,
  ``compile_network`` and ``use_cuda_graph`` stay ON. The edit LoRA hands
  its self-attention deltas to the dispatch (``release_targets``), whose
  skin sets carry LoRA + corrector in one ``copy_`` source, resolving the
  old last-writer-wins clash; the LoRA keeps toggling only the
  cross-attention projections. ``corrector_mode="unfused"`` (or
  ``LIVE_EDIT_CORRECTOR_MODE=unfused``) restores the eager scale-gated
  fallback, which still forces the graph-free pipeline;
  ``corrector_mode="off"`` deploys no corrector at all (configured
  corrector checkpoints are ignored and no weight sets are snapshotted),
- prompt swaps applied strictly between chunks by wrapping the session's
  ``start`` / ``continue_generation``; corrector-state selection rides the
  same boundary.

Vanilla behavior is untouched until :func:`attach_style_ability` runs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from loguru import logger

from crazy_robotaxi.live_edit.config import (
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
)
from crazy_robotaxi.live_edit.weather_ability import compose_swap_target

_NO_PENDING = object()
"""Sentinel distinguishing "no request" from "revert to base" (None)."""


class _V2PromptSession:
    """Small adapter exposing the legacy swap seam over a V2 rollout cache."""

    def __init__(self, pipeline: Any, cache: Any) -> None:
        self.pipeline = pipeline
        self._cache = cache
        self._pending_finalization_index = None

    def replace_prompt(
        self,
        prompt: str,
        *,
        guidance_scale: float,
        guidance_chunks: int,
    ) -> None:
        """Replace text directly on the V2 rollout cache."""
        self.pipeline.replace_text(
            self._cache,
            [[prompt]],
            guidance_scale=guidance_scale,
            guidance_chunks=guidance_chunks,
        )


class StyleAbility:
    """Cycle world skins and weather states on a running flashdreams session.

    One object owns the prompt state machine for both abilities so the
    mutual-exclusion rule holds at one seam: weather is base-world-only,
    the weather key is rejected while a skin is active, and activating a
    skin clears any active weather. A single ``replace_text`` per boundary
    carries the one active prompt; see
    :mod:`crazy_robotaxi.live_edit.weather_ability` for the state matrix.
    """

    def __init__(
        self,
        config: LiveEditStyleConfig,
        weather_config: LiveEditWeatherConfig | None = None,
    ) -> None:
        weather_enabled = weather_config is not None and weather_config.enabled
        if not config.enabled and not weather_enabled:
            raise ValueError(
                "StyleAbility requires live_edit.style or live_edit.weather"
            )
        self._config = config
        self._weather_config = weather_config if weather_enabled else None
        self._session: Any | None = None
        self._transformer: Any | None = None
        self._lora_attached = False
        self._base_prompt: str | None = None
        self._active_index: int | None = None
        self._pending_index: int | None | object = _NO_PENDING
        self._active_weather: int | None = None
        self._pending_weather: int | None | object = _NO_PENDING
        self._chunks_since_swap = 0
        self._skin_hold_chunks = 0
        self._weather_hold_chunks = 0
        # Per-activation timed-skin duration: item pickups (mystery box)
        # override the global skin_duration_chunks for one activation.
        self._active_skin_duration = config.skin_duration_chunks
        self._pending_skin_duration: int | None = None
        self._seconds_per_chunk = 8.0 / 30.0  # attach() reads the manifest
        self._set_corrector_gain: Callable[[float], None] = lambda _: None
        self._dispatch: Any | None = None
        self._corrector_states: set[str] = set()
        self._prompt_embeddings: dict[str, Any] = {}

    @property
    def active_skin_name(self) -> str:
        """Return the HUD label of the active skin (``base`` when off)."""
        if self._active_index is None or not self._config.enabled:
            return "base"
        return self._config.skins[self._active_index].name

    @property
    def skin_chunks_remaining(self) -> int | None:
        """Chunks left on the active timed skin (``None`` when untimed/off)."""
        duration = self._active_skin_duration
        if duration <= 0 or self._active_index is None:
            return None
        return max(duration - self._skin_hold_chunks, 0)

    @property
    def skin_seconds_remaining(self) -> float | None:
        """Seconds left on the active timed skin (``None`` when untimed/off).

        Derived from the manifest's chunk length at attach time; ticks at
        chunk granularity (~0.27 s for the shipped 8-frame recipe), which is
        plenty for a HUD countdown chip.
        """
        remaining = self.skin_chunks_remaining
        if remaining is None:
            return None
        return remaining * self._seconds_per_chunk

    @property
    def active_weather_name(self) -> str:
        """Return the HUD label of the active weather (``clear`` when off)."""
        if self._active_weather is None or self._weather_config is None:
            return "clear"
        return self._weather_config.weathers[self._active_weather].name

    @property
    def weather_chunks_remaining(self) -> int | None:
        """Chunks left on the active timed weather (``None`` when untimed/off)."""
        if self._weather_config is None or self._active_weather is None:
            return None
        duration = self._weather_config.duration_chunks
        if duration <= 0:
            return None
        return max(duration - self._weather_hold_chunks, 0)

    @property
    def weather_seconds_remaining(self) -> float | None:
        """Seconds left on the active timed weather (chunk granularity)."""
        remaining = self.weather_chunks_remaining
        if remaining is None:
            return None
        return remaining * self._seconds_per_chunk

    @property
    def skin_names(self) -> tuple[str, ...]:
        """Selectable skin names (empty when the style ability is off)."""
        if not self._config.enabled:
            return ()
        return tuple(skin.name for skin in self._config.skins)

    @property
    def weather_names(self) -> tuple[str, ...]:
        """Selectable weather names (empty when the weather ability is off)."""
        if self._weather_config is None:
            return ()
        return tuple(weather.name for weather in self._weather_config.weathers)

    def attach(self, session: Any) -> None:
        """Attach the LoRA + corrector and hook the chunk boundaries.

        Args:
            session: A warmed-up ``FlashdreamsWorldModelSession``. Accessing
                its pipeline before ``warmup_model()`` raises.

        Raises:
            RuntimeError: The manifest enables an acceleration mode the
                prompt-swap machinery (or the configured corrector) cannot
                ride; the message names the flags to drop.
        """
        self._guard_manifest(session.manifest)
        frames_per_chunk = getattr(session.manifest, "num_frames_per_block", 8)
        fps = getattr(session.manifest, "fps", 30) or 30
        self._seconds_per_chunk = float(frames_per_chunk) / float(fps)
        pipeline = session.pipeline
        transformer = pipeline.diffusion_model.transformer
        self._guard_transformer(transformer)

        mode = self._config.corrector_mode
        if mode == "unfused" and self._config.gate_alpha_json is not None:
            # The unfused path reads GATE_ALPHA_JSON at _drift_corrector
            # import time; the fused dispatch takes per-state profiles
            # directly, leaving the module default (photoreal) for the
            # base-state corrector.
            os.environ["GATE_ALPHA_JSON"] = str(self._config.gate_alpha_json)

        self._transformer = transformer
        edit_lora = None
        if self._config.enabled and self._config.lora_checkpoint is not None:
            from omnidreams.impl._edit_lora import TextEditLoRA

            edit_lora = TextEditLoRA(
                transformer.network, str(self._config.lora_checkpoint)
            )
            transformer.set_text_edit_lora(edit_lora)
            self._lora_attached = True
            logger.info(f"[live-edit] deployed {edit_lora.describe()}")

        if self._corrector_enabled():
            if mode == "fused":
                self._attach_corrector_fused(pipeline, transformer, edit_lora)
            else:
                self._attach_corrector(pipeline, transformer)
        elif mode == "off" and self._any_corrector_configured():
            logger.info(
                "[live-edit] corrector mode 'off': configured corrector "
                "checkpoints are ignored; transformer weights stay untouched"
            )

        self._precompute_prompt_embeddings(pipeline)
        self.hook_session(session)
        skins = (
            [skin.name for skin in self._config.skins] if self._config.enabled else []
        )
        weathers = (
            [weather.name for weather in self._weather_config.weathers]
            if self._weather_config is not None
            else []
        )
        logger.info(
            f"[live-edit] style ability attached skins={skins} weathers={weathers}"
        )

    def attach_v2(
        self,
        pipeline: Any,
        cache: Any,
        base_prompt: str,
        *,
        seconds_per_chunk: float,
    ) -> None:
        """Attach prompt editing to a direct API-v2 rollout cache.

        Args:
            pipeline: Session-shared OmniDreams pipeline.
            cache: Session-local autoregressive cache.
            base_prompt: Prompt used to initialize the rollout.
            seconds_per_chunk: Generated duration of one steady-state chunk.
        """
        transformer = pipeline.diffusion_model.transformer
        if getattr(transformer, "_optimized_dit_executor", None) is not None:
            raise RuntimeError("Live text editing requires a non-native model preset")
        self._guard_transformer(transformer)
        self._transformer = transformer
        self._base_prompt = base_prompt
        self._seconds_per_chunk = seconds_per_chunk
        edit_lora = None
        if self._config.enabled and self._config.lora_checkpoint is not None:
            from omnidreams.impl._edit_lora import TextEditLoRA

            edit_lora = TextEditLoRA(transformer.network, self._config.lora_checkpoint)
            transformer.set_text_edit_lora(edit_lora)
            self._lora_attached = True
        if self._corrector_enabled():
            if self._config.corrector_mode == "fused":
                self._attach_corrector_fused(pipeline, transformer, edit_lora)
            else:
                self._attach_corrector(pipeline, transformer)
        self._precompute_prompt_embeddings(pipeline)
        self._encode_prompt(pipeline, base_prompt)
        self._session = _V2PromptSession(pipeline, cache)
        self.reset_v2(cache)

    def reset_v2(self, cache: Any) -> None:
        """Bind a new V2 cache and reset all live-edit state."""
        if isinstance(self._session, _V2PromptSession):
            self._session._cache = cache
        self._active_index = None
        self._pending_index = _NO_PENDING
        self._active_weather = None
        self._pending_weather = _NO_PENDING
        self._chunks_since_swap = 0
        self._skin_hold_chunks = 0
        self._weather_hold_chunks = 0
        self._active_skin_duration = self._config.skin_duration_chunks
        self._pending_skin_duration = None
        self._update_corrector(None, None, 0.0)

    def before_v2_chunk(self) -> None:
        """Apply queued edits at the model-thread chunk boundary."""
        if self._timed_skin_expired():
            self._pending_index = None
        if self._timed_weather_expired():
            self._pending_weather = None
        refresh_due = self._reswap_due()
        if (
            self._pending_index is not _NO_PENDING
            or self._pending_weather is not _NO_PENDING
            or refresh_due
        ):
            self._apply_pending(refresh=refresh_due)

    def after_v2_chunk(self) -> None:
        """Advance timed ability counters after one generated chunk."""
        if self._active_index is not None or self._active_weather is not None:
            self._chunks_since_swap += 1
        if self._active_index is not None:
            self._skin_hold_chunks += 1
        if self._active_weather is not None:
            self._weather_hold_chunks += 1

    def _precompute_prompt_embeddings(self, pipeline: Any) -> None:
        """Encode every configured swap prompt once at session start.

        A swap's dominant cost is the text-encoder forward inside
        ``replace_text`` (450-930 ms at the chunk boundary); the prompts are
        all known up front, so encoding them here lets ``_replace_text``
        inject cached embeddings through the pipeline's
        ``replace_text_from_embeddings`` and skip the encoder entirely.

        No-op when the pipeline has no resident text encoder (the offload
        path releases it); swaps then fall back to ``replace_text``, whose
        own assertion reports the missing encoder.
        """
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            logger.warning(
                "[live-edit] no resident text encoder; prompt swaps will "
                "re-encode per swap (pre-encoding skipped)"
            )
            return
        prompts: list[str] = []
        if self._config.enabled:
            prompts.extend(skin.prompt for skin in self._config.skins)
        if self._weather_config is not None:
            prompts.extend(weather.prompt for weather in self._weather_config.weathers)
        start = time.perf_counter()
        for prompt in prompts:
            self._encode_prompt(pipeline, prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"[live-edit] pre-encoded {len(prompts)} swap prompts in "
            f"{elapsed_ms:.0f} ms (swaps now inject cached embeddings)"
        )

    def _encode_prompt(self, pipeline: Any, prompt: str) -> None:
        """Cache the ``[B=1, V=1, L, D]`` embeddings of one prompt."""
        if prompt in self._prompt_embeddings:
            return
        import torch

        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        with torch.no_grad():
            self._prompt_embeddings[prompt] = text_encoder([prompt]).unsqueeze(0)

    def request_cycle(self) -> None:
        """Queue base -> skin[0] -> skin[1] -> ... -> base for the next chunk.

        Weather is base-world-only, so activating any skin also queues the
        weather back to clear (documented state-machine rule: K wins over an
        active weather; V is rejected while a skin is active).

        Timed power-up mode (``skin_duration_chunks > 0``) keeps these exact
        K semantics: pressing K during an active timed skin cycles to the
        NEXT skin with a fresh timer (the last skin cycles to base early).
        Chosen over extend/reset-in-place because it keeps K meaning one
        thing in both modes, every skin stays reachable mid-power-up, and a
        cycle re-lands the swap anyway — so the new skin's timer is
        naturally fresh; a dedicated "extend" would add a second behavior
        for the same key with no gameplay the cycle doesn't already give.
        """
        if not self._config.enabled:
            return
        current = self._skin_state()
        if current is None:
            self._pending_index = 0
        elif current + 1 < len(self._config.skins):
            self._pending_index = current + 1
        else:
            self._pending_index = None
        # A K press always uses the global duration, even when it races a
        # queued mystery-box burst at the same boundary (last request wins).
        self._pending_skin_duration = None
        if self._pending_index is not None and self._weather_state() is not None:
            self._pending_weather = None
            logger.info(
                "[live-edit] skin activation clears weather (base-only ability)"
            )

    def request_skin_burst(self, name: str, duration_chunks: int) -> str | None:
        """Queue a specific skin with a per-activation duration override.

        The mystery-box pickup path: lands at the next chunk boundary
        through the exact machinery :meth:`request_cycle` uses, so it
        composes with the K key — a burst during a key-held skin behaves
        like a K cycle (switch, fresh timer), and rolling the skin that is
        already active re-lands the swap with a fresh burst timer.

        Args:
            name: Skin name from :attr:`skin_names`.
            duration_chunks: Auto-revert after this many chunks (0 = the
                granted skin is untimed).

        Returns:
            The queued skin name, or ``None`` when the style ability is off
            or the name is unknown (logged, never raises: pickups must not
            crash the frame loop).
        """
        if not self._config.enabled:
            return None
        names = [skin.name for skin in self._config.skins]
        if name not in names:
            logger.warning(f"[live-edit] unknown skin burst {name!r}; ignoring")
            return None
        self._pending_index = names.index(name)
        self._pending_skin_duration = max(0, int(duration_chunks))
        if self._weather_state() is not None:
            self._pending_weather = None
            logger.info("[live-edit] skin burst clears weather (base-only ability)")
        return name

    def request_weather(self, name: str) -> bool:
        """Queue a specific weather preset (item-pickup path).

        Same base-world-only rule as the V key: rejected while a skin is
        active or queued (the caller shows the HUD hint). Re-requesting the
        active weather refreshes its timed-weather timer without re-landing
        the swap (a same-prompt guided re-swap has a zero guidance
        direction — pure 2x cost).

        Returns:
            ``True`` when the weather was queued (or its timer refreshed);
            ``False`` when rejected (skin active, ability off, unknown name).
        """
        if self._weather_config is None:
            return False
        names = [weather.name for weather in self._weather_config.weathers]
        if name not in names:
            logger.warning(f"[live-edit] unknown weather {name!r}; ignoring")
            return False
        if self._skin_state() is not None:
            logger.info(
                f"[live-edit] weather is base-skin only; ignoring {name} pickup"
            )
            return False
        self._pending_weather = names.index(name)
        return True

    def request_weather_cycle(self) -> None:
        """Queue clear -> rain -> snow -> clear for the next chunk.

        Ignored while a skin is active or queued: weather only runs over
        the base world (skin+weather combo prompts were dropped 2026-08-20).
        """
        if self._weather_config is None:
            return
        skin_state = self._skin_state()
        if skin_state is not None:
            logger.info(
                "[live-edit] weather is base-skin only; ignoring V "
                f"(skin={self._config.skins[skin_state].name})"
            )
            return
        current = self._weather_state()
        if current is None:
            self._pending_weather = 0
        elif current + 1 < len(self._weather_config.weathers):
            self._pending_weather = current + 1
        else:
            self._pending_weather = None

    def _skin_state(self) -> int | None:
        """Effective skin index once any pending request lands."""
        if self._pending_index is _NO_PENDING:
            return self._active_index
        return cast(int | None, self._pending_index)

    def _weather_state(self) -> int | None:
        """Effective weather index once any pending request lands."""
        if self._pending_weather is _NO_PENDING:
            return self._active_weather
        return cast(int | None, self._pending_weather)

    def _corrector_enabled(self) -> bool:
        """Whether any drift corrector will actually attach to the session.

        ``corrector_mode == "off"`` disables every corrector even when
        checkpoints are configured; ``fused`` needs at least one registered
        state beyond ``base``; ``unfused`` rides the single style
        checkpoint.
        """
        mode = self._config.corrector_mode
        if mode == "off":
            return False
        if mode == "fused":
            return self._any_corrector_configured()
        return self._config.corrector_checkpoint is not None

    def _any_corrector_configured(self) -> bool:
        """Whether any state of the fused dispatch would carry a corrector."""
        weather = self._weather_config
        return (
            self._config.corrector_checkpoint is not None
            or self._config.base_corrector_checkpoint is not None
            or (
                weather is not None
                and weather.corrector_gain > 0.0
                and weather.corrector_checkpoint is not None
            )
        )

    def _attach_corrector_fused(
        self, pipeline: Any, transformer: Any, edit_lora: Any | None
    ) -> None:
        """Deploy the CUDA-graph-safe per-state corrector dispatch.

        One pre-merged weight-set family per (base | skin | weather)
        state; :meth:`_apply_pending` selects the state at chunk
        boundaries. The edit LoRA releases its self-attention projections
        to the dispatch, whose skin sets fold the LoRA delta into every
        alpha set — one ``copy_`` source carries LoRA + corrector, so the
        two mechanisms no longer race on the same weights; the LoRA keeps
        toggling only the cross-attention projections. Consequence: while
        a skin state is selected, the self-attention LoRA delta stays
        applied even after the cross-attention edit window ages out
        (the 8-chunk re-swap reopens the skin window before long holds
        soften, so the split is invisible in practice).
        """
        from types import SimpleNamespace

        from omnidreams.impl._drift_corrector import (
            DriftCorrectorDispatch,
            _target_linears,
        )

        dispatch = DriftCorrectorDispatch(SimpleNamespace(pipeline=pipeline))
        lines: list[str] = []
        lora_delta = None
        if edit_lora is not None:
            network = transformer.network
            if hasattr(network, "_orig_mod"):  # unwrap torch.compile
                network = network._orig_mod
            lora_delta = edit_lora.release_targets(_target_linears(network))

        config = self._config
        if config.base_corrector_checkpoint is not None:
            # Photoreal corrector over the base world, module-default gate.
            lines.append(
                dispatch.register_state(
                    "base",
                    checkpoint=config.base_corrector_checkpoint,
                    gain=config.base_corrector_gain,
                )
            )
        if config.enabled and (
            config.corrector_checkpoint is not None or lora_delta is not None
        ):
            lines.append(
                dispatch.register_state(
                    "skin",
                    checkpoint=config.corrector_checkpoint,
                    gain=(
                        config.corrector_gain
                        if config.corrector_checkpoint is not None
                        else 0.0
                    ),
                    gate_alpha=config.gate_alpha_json,
                    lora_delta=lora_delta,
                )
            )
            self._corrector_states.add("skin")
        weather = self._weather_config
        if weather is not None and weather.corrector_gain > 0.0:
            ckpt = weather.corrector_checkpoint or config.corrector_checkpoint
            if ckpt is not None:
                lines.append(
                    dispatch.register_state(
                        "weather",
                        checkpoint=ckpt,
                        gain=weather.corrector_gain,
                        gate_alpha=config.gate_alpha_json,
                    )
                )
                self._corrector_states.add("weather")
        self._corrector_states.add("base")
        self._dispatch = dispatch
        for line in lines:
            logger.info(f"[live-edit] fused {line}")

    def _update_corrector(
        self, skin: int | None, weather: int | None, gain: float
    ) -> None:
        """Route the new (skin | weather) state to the corrector backend.

        Fused: select the dispatch state (states without a registration
        fall back to ``base``). Unfused: apply the absolute gain via the
        scale-gated predict_flow dispatch.
        """
        if self._dispatch is not None:
            name = (
                "skin"
                if skin is not None
                else "weather"
                if weather is not None
                else "base"
            )
            if name not in self._corrector_states:
                name = "base"
            self._dispatch.set_active_corrector(name)
        else:
            self._set_corrector_gain(gain)

    def _attach_corrector(self, pipeline: Any, transformer: Any) -> None:
        """Deploy the unfused corrector behind a per-state gain dispatch.

        Legacy fallback (``corrector_mode="unfused"``): requires the
        graph-free pipeline; see :meth:`_attach_corrector_fused` for the
        real-time path.

        The dispatch supports three regimes per (skin | weather) state:
        the configured style gain rides the validated ``gated_pf`` wrapper
        unchanged; gain 0 short-circuits to the bit-clean base forward; any
        other gain (e.g. a reduced weather gain) re-derives the per-step
        LoRA scale ``alpha*(t) * gain`` here before calling the base
        forward — identical math to ``gated_pf`` at a different gain, since
        the unfused _LoRALinear wrappers stay installed permanently and
        only the scale changes.
        """
        from types import SimpleNamespace

        from omnidreams.impl._drift_corrector import (
            _nearest_alpha,
            _set_scale,
            apply_drift_corrector,
        )

        base_predict_flow = transformer.predict_flow
        style_gain = self._config.corrector_gain
        checkpoint = self._config.corrector_checkpoint
        assert checkpoint is not None
        summary = apply_drift_corrector(
            SimpleNamespace(pipeline=pipeline),
            checkpoint,
            style_gain,
            unfused=True,
        )
        corrected_predict_flow = transformer.predict_flow
        active_gain = [0.0]

        # The unfused deployment installs _LoRALinear wrappers permanently;
        # only the predict_flow wrapper re-scales them per step. Dispatching
        # to the base predict_flow therefore leaves the LAST scale applied,
        # so gain 0 must also zero the LoRA scale (scale == 0 is an exact
        # short-circuit in _LoRALinear.forward -> bit-clean base output).
        network = transformer.network
        if hasattr(network, "_orig_mod"):  # unwrap torch.compile
            network = network._orig_mod
        _set_scale(network, 0.0)

        def dispatched_predict_flow(*args: Any, **kwargs: Any) -> Any:
            gain = active_gain[0]
            if gain <= 0.0:
                return base_predict_flow(*args, **kwargs)
            if gain == style_gain:
                return corrected_predict_flow(*args, **kwargs)
            timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            t = float(timestep.reshape(-1).max())
            _set_scale(network, _nearest_alpha(t) * gain)
            return base_predict_flow(*args, **kwargs)

        def set_gain(value: float) -> None:
            active_gain[0] = float(value)
            if active_gain[0] <= 0.0:
                _set_scale(network, 0.0)

        transformer.predict_flow = dispatched_predict_flow
        self._set_corrector_gain = set_gain
        logger.info(f"[live-edit] {summary} (dispatch-gated, gain 0)")

    def hook_session(self, session: Any) -> None:
        """Wrap the session's chunk boundaries (model-free; CPU-testable).

        ``attach`` calls this after deploying the LoRA/corrector; tests can
        call it directly with a fake session to exercise the swap protocol.
        """
        self._session = session
        original_start = session.start
        original_continue = session.continue_generation

        def start(initial_rgb: Any, condition_frames: Any, prompt: str) -> Any:
            self._base_prompt = prompt
            # The base (revert) prompt is only known here; encode it once so
            # reverting to base/clear is also an embedding injection.
            self._encode_prompt(session.pipeline, prompt)
            self._active_index = None
            self._pending_index = _NO_PENDING
            self._active_weather = None
            self._pending_weather = _NO_PENDING
            self._chunks_since_swap = 0
            self._skin_hold_chunks = 0
            self._weather_hold_chunks = 0
            self._active_skin_duration = self._config.skin_duration_chunks
            self._pending_skin_duration = None
            self._update_corrector(None, None, 0.0)
            return original_start(initial_rgb, condition_frames, prompt)

        def continue_generation(condition_frames: Any) -> Any:
            if self._timed_skin_expired():
                # Auto-revert rides the exact K-cycle revert path: a queued
                # None lands as the plain base swap (guidance 1.0/0) and
                # moves the corrector dispatch back to the base state. A
                # user K press queued this boundary wins (it re-lands a
                # fresh swap with a fresh timer).
                self._pending_index = None
                logger.info(
                    f"[live-edit] timed skin {self.active_skin_name} expired "
                    f"after {self._skin_hold_chunks} chunks; reverting to base"
                )
            if self._timed_weather_expired():
                # Rides the V-cycle wrap-to-clear path; the landing is
                # GUIDED (see _apply_pending) because clear is itself a
                # weather transition. A user request queued this boundary
                # wins.
                self._pending_weather = None
                logger.info(
                    f"[live-edit] timed weather {self.active_weather_name} "
                    f"expired after {self._weather_hold_chunks} chunks; "
                    "landing clear"
                )
            refresh_due = self._reswap_due()
            if (
                self._pending_index is not _NO_PENDING
                or self._pending_weather is not _NO_PENDING
                or refresh_due
            ):
                # The validated swap semantics are finalize -> replace_text
                # -> generate (otherwise finalize re-commits the previous
                # chunk under the NEW text, an implicit recache). The swap
                # path (:meth:`_replace_text`) flushes the adapter's deferred
                # finalize before every swap, so no flush is needed here.
                self._apply_pending(refresh=refresh_due)
            result = original_continue(condition_frames)
            if self._active_index is not None or self._active_weather is not None:
                self._chunks_since_swap += 1
            if self._active_index is not None:
                self._skin_hold_chunks += 1
            if self._active_weather is not None:
                self._weather_hold_chunks += 1
            return result

        session.start = start
        session.continue_generation = continue_generation

    def _timed_skin_expired(self) -> bool:
        """Whether the active timed skin is due its auto-revert to base.

        False when the mode is off (duration 0), no skin is active, or the
        user already queued a request this boundary (their cycle wins).
        """
        duration = self._active_skin_duration
        return (
            duration > 0
            and self._active_index is not None
            and self._pending_index is _NO_PENDING
            and self._skin_hold_chunks >= duration
        )

    def _timed_weather_expired(self) -> bool:
        """Whether the active timed weather is due its guided clear landing.

        False when the mode is off (duration 0), no weather is active, or a
        user request already queued this boundary (their request wins; a
        queued skin also clears weather through its own path).
        """
        if self._weather_config is None:
            return False
        duration = self._weather_config.duration_chunks
        return (
            duration > 0
            and self._active_weather is not None
            and self._pending_weather is _NO_PENDING
            and self._pending_index is _NO_PENDING
            and self._weather_hold_chunks >= duration
        )

    def _reswap_due(self) -> bool:
        """Whether the active edit window is due a duty-cycle refresh.

        Skins refresh on the style ``reswap_interval_chunks`` (the LoRA
        realizes the window single-branch, so a refresh is free per chunk).
        Weather holds land-then-release: the landing window expires and the
        state persists through KV history + the swapped text, so weather
        only refreshes when a maintenance interval is explicitly configured
        (each pulse costs ``maintain_chunks`` chunks at 2x).
        """
        if self._active_index is not None:
            interval = self._config.reswap_interval_chunks
            duration = self._active_skin_duration
            if 0 < duration <= interval:
                # Timed skin expires at or before the first refresh would
                # fire — the re-swap would only ever land on the revert
                # boundary, so skip the duty cycle entirely.
                return False
        elif self._active_weather is not None and self._weather_config is not None:
            interval = self._weather_config.maintain_interval_chunks
        else:
            return False
        return interval > 0 and self._chunks_since_swap >= interval

    def _apply_pending(self, *, refresh: bool = False) -> None:
        """Swap the prompt between chunks when a request or refresh is due."""
        pending_skin = self._pending_index
        pending_weather = self._pending_weather
        pending_duration = self._pending_skin_duration
        self._pending_index = _NO_PENDING
        self._pending_weather = _NO_PENDING
        self._pending_skin_duration = None
        target_skin: int | None = (
            self._active_index
            if pending_skin is _NO_PENDING
            else cast(int | None, pending_skin)
        )
        target_weather: int | None = (
            self._active_weather
            if pending_weather is _NO_PENDING
            else cast(int | None, pending_weather)
        )
        changed = (
            target_skin != self._active_index or target_weather != self._active_weather
        )
        # An explicit request for the already-active skin (mystery box
        # rolling it again) re-lands the swap with a fresh timer; the same
        # request for the active weather only refreshes the timer (a
        # same-prompt guided re-swap has a zero guidance direction).
        explicit_same_skin = (
            pending_skin is not _NO_PENDING
            and pending_skin is not None
            and pending_skin == self._active_index
        )
        if (
            pending_weather is not _NO_PENDING
            and pending_weather is not None
            and pending_weather == self._active_weather
        ):
            self._weather_hold_chunks = 0
        if (
            not changed
            and not explicit_same_skin
            and (not refresh or (target_skin is None and target_weather is None))
        ):
            return
        session = self._session
        if session is None or session._cache is None or self._base_prompt is None:
            logger.warning("[live-edit] prompt swap requested before first chunk")
            return

        target = compose_swap_target(
            base_prompt=self._base_prompt,
            skin=None if target_skin is None else self._config.skins[target_skin],
            weather=(
                None
                if target_weather is None or self._weather_config is None
                else self._weather_config.weathers[target_weather]
            ),
            style_config=self._config,
            weather_config=self._weather_config,
            lora_available=self._lora_attached,
        )
        if (
            self._active_weather is not None
            and target_skin is None
            and target_weather is None
            and self._weather_config is not None
        ):
            # Weather -> clear is itself a weather transition: a plain swap
            # leaves the precipitation running on KV-history momentum, so
            # the clear lands GUIDED (both the timed auto-revert and the
            # V-cycle wrap). Accumulated scene change (wet roads, settled
            # snow) is NOT undone — it decays naturally, by design.
            target = replace(
                target,
                guidance_scale=self._weather_config.guidance_scale,
                guidance_chunks=self._weather_config.clear_guidance_chunks,
            )
        if not changed and target_weather is not None and target_skin is None:
            # Weather maintenance pulse. A same-prompt re-swap would clone
            # its "old" KV from buffers that already hold the weather text,
            # collapsing the guidance direction to zero (paying 2x for
            # nothing); rebase to the base prompt first so the pulse pushes
            # weather-minus-base again, for maintain_chunks chunks only.
            assert self._weather_config is not None
            rebase = compose_swap_target(
                base_prompt=self._base_prompt,
                skin=None,
                weather=None,
                style_config=self._config,
                weather_config=self._weather_config,
                lora_available=self._lora_attached,
            )
            self._replace_text(session, rebase)
            target = replace(
                target, guidance_chunks=self._weather_config.maintain_chunks
            )
        self._replace_text(session, target)
        verb = "re-swap" if not (changed or explicit_same_skin) else "state ->"
        if target_skin != self._active_index or explicit_same_skin:
            # Fresh activation (or skin->skin cycle, or an explicit re-roll
            # of the active skin): a timed skin starts a fresh timer with
            # its per-activation duration. A duty-cycle re-swap keeps the
            # timer running.
            self._skin_hold_chunks = 0
            self._active_skin_duration = (
                self._config.skin_duration_chunks
                if pending_duration is None
                else pending_duration
            )
        if target_weather != self._active_weather:
            self._weather_hold_chunks = 0
        self._active_index = target_skin
        self._active_weather = target_weather
        self._chunks_since_swap = 0
        self._update_corrector(target_skin, target_weather, target.corrector_gain)
        logger.info(
            f"[live-edit] {verb} skin={self.active_skin_name} "
            f"weather={self.active_weather_name}"
        )

    def _replace_text(self, session: Any, target: Any) -> None:
        """Issue the swap, bypassing the edit LoRA for two-prompt windows.

        A guided ``replace_text`` routes through the pre-merged text-edit
        LoRA whenever one is attached; weather-only windows must instead run
        the two-prompt KV-snapshot guidance (the LoRA was trained on the
        style prompts), so the LoRA is detached around the call. Plain swaps
        (scale 1.0) never open a LoRA window and need no bypass.

        Prompts pre-encoded at attach time (see
        :meth:`_precompute_prompt_embeddings`) inject their cached
        embeddings through ``replace_text_from_embeddings`` — no text
        encoder forward at the boundary; anything else falls back to the
        session's encode-per-swap ``replace_prompt``.

        Both paths flush the adapter's deferred chunk finalize first:
        finalize must run under the OLD text. ``session.replace_prompt``
        does its own flush; the embeddings fast path (no upstream
        equivalent yet) flushes here before touching the cache.
        """
        transformer = self._transformer
        bypass_lora = (
            not target.use_lora
            and target.guidance_scale != 1.0
            and transformer is not None
            and getattr(transformer, "_text_edit_lora", None) is not None
        )
        edit_lora = None
        if bypass_lora:
            edit_lora = transformer._text_edit_lora
            transformer.set_text_edit_lora(None)
        embeddings = self._prompt_embeddings.get(target.prompt)
        replace_from_embeddings = getattr(
            session.pipeline, "replace_text_from_embeddings", None
        )
        cached = embeddings is not None and callable(replace_from_embeddings)
        start = time.perf_counter()
        try:
            if cached:
                self._flush_pending_finalize(session)
                import torch

                with torch.no_grad():
                    replace_from_embeddings(
                        session._cache,
                        embeddings,
                        guidance_scale=target.guidance_scale,
                        guidance_chunks=target.guidance_chunks,
                        recache_last_chunk=False,
                    )
            else:
                session.replace_prompt(
                    target.prompt,
                    guidance_scale=target.guidance_scale,
                    guidance_chunks=target.guidance_chunks,
                )
        finally:
            if bypass_lora:
                transformer.set_text_edit_lora(edit_lora)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"[live-edit] swap issued cached_embeddings={cached} "
            f"swap_ms={elapsed_ms:.1f}"
        )

    @staticmethod
    def _flush_pending_finalize(session: Any) -> None:
        """Flush the adapter's deferred chunk finalize under the OLD text.

        Only the embeddings fast path needs this; ``session.replace_prompt``
        performs the same flush itself.
        """
        pending_finalize = getattr(session, "_pending_finalization_index", None)
        if pending_finalize is None or session._cache is None:
            return
        import torch

        with torch.no_grad():
            session.pipeline.finalize(pending_finalize, session._cache)
        session._pending_finalization_index = None

    def _guard_manifest(self, manifest: Any) -> None:
        """Reject native-DIT manifests with a message naming the fix.

        Prompt-swap abilities fundamentally need the Python transformer
        forward today, for two independent reasons (verified 2026-08-21):

        - ``CosmosTransformer.replace_text_embeddings`` raises
          ``NotImplementedError`` under the native optimized-DiT executor
          (the cross-attention KV rebuild is not wired for it), and every
          skin/weather swap goes through it;
        - the pre-merged ``TextEditLoRA`` toggles by ``copy_``-ing into the
          ``nn.Linear`` weights, but the native fp8 executor quantizes a
          one-time weight snapshot (and then releases the PyTorch network),
          so those toggles would silently never reach the native forward.

        Correctors add a third reason (same snapshot bypass), but they are
        gated separately: with no corrector enabled the message does not
        ask the user to change corrector flags. Abilities that never touch
        the model (coins, obstacle without ``--live-edit-obstacle-guide-scale``)
        do not construct this ability and stay perf-neutral under native
        DIT.
        """
        if getattr(manifest, "native_dit_acceleration", "disabled") in (
            "disabled",
            None,
            False,
        ):
            return
        flags = []
        if self._config.enabled:
            flags.append("--live-edit-style")
        if self._weather_config is not None:
            flags.append("--live-edit-weather")
        corrector_note = (
            " The configured drift corrector also merges into the PyTorch "
            "network's weights, which the native executor bypasses "
            "(--live-edit-corrector-mode off would disable it, but the "
            "prompt-swap limitation above still applies)."
            if self._corrector_enabled()
            else ""
        )
        raise RuntimeError(
            "Prompt-swap live-edit abilities need the Python transformer "
            "forward: replace_text_embeddings is not wired for the native "
            "optimized-DiT executor, and the pre-merged text-edit LoRA "
            "toggles weights the native fp8 snapshot never re-reads. Either "
            f"drop {' / '.join(flags)} (coins and other pixel-only "
            "abilities stay available and perf-neutral), or set "
            "native_dit_acceleration: disabled in the world-model manifest."
            + corrector_note
        )

    def _guard_transformer(self, transformer: Any) -> None:
        """Reject built pipeline configs the unfused corrector cannot ride.

        Fused mode needs no rejection (the per-state dispatch copies into
        fixed parameter storages, which captured CUDA graphs and the
        compiled network read by address — the whole point of the mode),
        and ``off`` deploys no corrector at all.

        The manifest only carries ``compile_net`` / ``native_dit_*``; the
        transformer's ``use_cuda_graph`` defaults to True in the recipe, so
        it must be checked on the live config. CUDA-graph capture would bake
        the corrector's scale-0 short-circuit (and the predict_flow dispatch
        runs outside any captured graph), and ``compile_network`` re-traces
        around the _LoRALinear wrap.
        """
        if self._config.corrector_mode != "unfused":
            return
        config = getattr(transformer, "config", None)
        if config is None:
            return
        needs_corrector = self._config.corrector_checkpoint is not None
        if needs_corrector and getattr(config, "use_cuda_graph", False):
            raise RuntimeError(
                "live_edit.style with the drift corrector requires "
                "use_cuda_graph=False on the transformer (unfused LoRA "
                "scale-gating is not graph-safe)."
            )
        if needs_corrector and getattr(config, "compile_network", False):
            raise RuntimeError(
                "live_edit.style with the drift corrector requires "
                "compile_network=False (bring-up parity with the validated "
                "smoke-harness configuration)."
            )


def attach_style_ability(session: Any, config: LiveEditStyleConfig) -> StyleAbility:
    """Create and attach the style ability to a warmed-up session."""
    ability = StyleAbility(config)
    ability.attach(session)
    return ability
