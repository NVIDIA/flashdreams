# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live game-skin switching on the flashdreams world-model session.

Ports the attach recipe from
``integrations/omnidreams/scripts/smoke_text_edit.py`` onto
:class:`omnidreams_game_engine.world_model.flashdreams_adapter.FlashdreamsWorldModelSession`:

- a pre-merged :class:`omnidreams._edit_lora.TextEditLoRA` on the
  transformer (zero steady-state cost; ``replace_text`` opens its edit
  window automatically),
- the rank-16 drift corrector. Default (``corrector_mode="fused"``): the
  CUDA-graph-safe per-state
  :class:`omnidreams._drift_corrector.DriftCorrectorDispatch` — one
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
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from loguru import logger

from crazy_robotaxi.live_edit.config import (
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
)
from crazy_robotaxi.live_edit.weather_ability import compose_swap_target

_NO_PENDING = object()
"""Sentinel distinguishing "no request" from "revert to base" (None)."""


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
        self._set_corrector_gain: Callable[[float], None] = lambda _: None
        self._dispatch: Any | None = None
        self._corrector_states: set[str] = set()

    @property
    def active_skin_name(self) -> str:
        """Return the HUD label of the active skin (``base`` when off)."""
        if self._active_index is None or not self._config.enabled:
            return "base"
        return self._config.skins[self._active_index].name

    @property
    def active_weather_name(self) -> str:
        """Return the HUD label of the active weather (``clear`` when off)."""
        if self._active_weather is None or self._weather_config is None:
            return "clear"
        return self._weather_config.weathers[self._active_weather].name

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
            from omnidreams._edit_lora import TextEditLoRA

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

    def request_cycle(self) -> None:
        """Queue base -> skin[0] -> skin[1] -> ... -> base for the next chunk.

        Weather is base-world-only, so activating any skin also queues the
        weather back to clear (documented state-machine rule: K wins over an
        active weather; V is rejected while a skin is active).
        """
        if not self._config.enabled:
            return
        current = (
            self._active_index
            if self._pending_index is _NO_PENDING
            else self._pending_index
        )
        if current is None:
            self._pending_index = 0
        elif current + 1 < len(self._config.skins):
            self._pending_index = current + 1
        else:
            self._pending_index = None
        if self._pending_index is not None and self._weather_state() is not None:
            self._pending_weather = None
            logger.info(
                "[live-edit] skin activation clears weather (base-only ability)"
            )

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
        return self._pending_index  # type: ignore[return-value]

    def _weather_state(self) -> int | None:
        """Effective weather index once any pending request lands."""
        if self._pending_weather is _NO_PENDING:
            return self._active_weather
        return self._pending_weather  # type: ignore[return-value]

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

        from omnidreams._drift_corrector import (
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

        from omnidreams._drift_corrector import (
            _nearest_alpha,
            _set_scale,
            apply_drift_corrector,
        )

        base_predict_flow = transformer.predict_flow
        style_gain = self._config.corrector_gain
        summary = apply_drift_corrector(
            SimpleNamespace(pipeline=pipeline),
            self._config.corrector_checkpoint,
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
            self._active_index = None
            self._pending_index = _NO_PENDING
            self._active_weather = None
            self._pending_weather = _NO_PENDING
            self._chunks_since_swap = 0
            self._update_corrector(None, None, 0.0)
            return original_start(initial_rgb, condition_frames, prompt)

        def continue_generation(condition_frames: Any) -> Any:
            refresh_due = self._reswap_due()
            if (
                self._pending_index is not _NO_PENDING
                or self._pending_weather is not _NO_PENDING
                or refresh_due
            ):
                # The adapter defers finalize of chunk N into the next
                # continue_generation call; the validated swap semantics are
                # finalize -> replace_text -> generate (otherwise finalize
                # re-commits the previous chunk under the NEW text, an
                # implicit recache). Flush the pending finalize first; the
                # adapter's own finalize branch then no-ops.
                pending_finalize = getattr(session, "_pending_finalization_index", None)
                if pending_finalize is not None and session._cache is not None:
                    import torch

                    with torch.no_grad():
                        session.pipeline.finalize(pending_finalize, session._cache)
                    session._pending_finalization_index = None
                self._apply_pending(refresh=refresh_due)
            result = original_continue(condition_frames)
            if self._active_index is not None or self._active_weather is not None:
                self._chunks_since_swap += 1
            return result

        session.start = start
        session.continue_generation = continue_generation

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
        elif self._active_weather is not None and self._weather_config is not None:
            interval = self._weather_config.maintain_interval_chunks
        else:
            return False
        return interval > 0 and self._chunks_since_swap >= interval

    def _apply_pending(self, *, refresh: bool = False) -> None:
        """Swap the prompt between chunks when a request or refresh is due."""
        pending_skin = self._pending_index
        pending_weather = self._pending_weather
        self._pending_index = _NO_PENDING
        self._pending_weather = _NO_PENDING
        target_skin = (
            self._active_index if pending_skin is _NO_PENDING else pending_skin
        )
        target_weather = (
            self._active_weather if pending_weather is _NO_PENDING else pending_weather
        )
        changed = (
            target_skin != self._active_index or target_weather != self._active_weather
        )
        if not changed and (
            not refresh or (target_skin is None and target_weather is None)
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
        verb = "re-swap" if not changed else "state ->"
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
        """
        import torch

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
        try:
            # TODO(upstream): session._cache is private; ask for a public
            # replace_text passthrough on FlashdreamsWorldModelSession.
            with torch.no_grad():
                session.pipeline.replace_text(
                    session._cache,
                    [[target.prompt]],
                    guidance_scale=target.guidance_scale,
                    guidance_chunks=target.guidance_chunks,
                    recache_last_chunk=False,
                )
        finally:
            if bypass_lora:
                transformer.set_text_edit_lora(edit_lora)

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


def install_style_ability_on_backend(backend: Any, ability: StyleAbility) -> None:
    """Prepare a ``WorldModelRenderBackend`` for the style ability.

    Called at the composition root BEFORE model warmup starts:

    - in ``unfused`` corrector mode only, replaces the backend's session
      with one whose pipeline factory disables CUDA graphs (the manifest
      has no such knob, and the unfused corrector's scale gating is not
      graph-safe). The default ``fused`` mode keeps the accelerated
      pipeline (compile_network + use_cuda_graph) untouched;
    - defers :meth:`StyleAbility.attach` until the session's own
      ``warmup_model`` has built the pipeline (attach needs the live
      transformer), by wrapping that method.
    """
    session = getattr(backend, "_session", None)
    if session is None:
        raise ValueError(
            "--live-edit-style requires the omnidreams world-model backend "
            "(the raster backend has no flashdreams session)."
        )
    needs_graph_free = (
        ability._config.corrector_checkpoint is not None
        and ability._config.corrector_mode == "unfused"
    )
    if needs_graph_free and not getattr(session, "_live_edit_cuda_graph_free", False):
        session = _corrector_safe_session(session)
        backend._session = session
    original_warmup = session.warmup_model

    def warmup_and_attach() -> None:
        original_warmup()
        ability.attach(session)

    session.warmup_model = warmup_and_attach


def _corrector_safe_session(session: Any) -> Any:
    """Rebuild a not-yet-warmed session with CUDA graphs disabled.

    Mirrors the validated bring-up factory: same manifest / profile /
    offload / postprocess settings, but the transformer is built with
    ``use_cuda_graph=False`` so the unfused drift corrector's per-step LoRA
    scale gating stays outside any captured graph.
    """
    from omnidreams_game_engine.world_model.flashdreams_adapter import (
        FlashdreamsWorldModelSession,
        _build_pipeline_config,
        _setup_pipeline_from_config,
    )

    def cuda_graph_free_factory(manifest: Any, profile: Any) -> Any:
        from flashdreams.infra.config import derive_config

        config = _build_pipeline_config(manifest, profile)
        config = derive_config(
            config, diffusion_model={"transformer": {"use_cuda_graph": False}}
        )
        return _setup_pipeline_from_config(config, manifest)

    rebuilt = FlashdreamsWorldModelSession(
        session.manifest,
        profile=session._profile_config,
        offload_text_encoder=session._offload_text_encoder,
        pipeline_factory=cuda_graph_free_factory,
        postprocess=session._postprocess,
    )
    # Marker so cooperating installers (obstacle guidance) skip a re-swap.
    rebuilt._live_edit_cuda_graph_free = True
    return rebuilt
