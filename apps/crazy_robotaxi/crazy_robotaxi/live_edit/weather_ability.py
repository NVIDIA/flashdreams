# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weather events: composing the prompt swap for the (skin | weather) state.

There is no weather LoRA. Weather uses the plain two-prompt edit-guidance
mechanism (the PR #431 ``replace_text`` path: the old prompt anchors the
scene, the flow is pushed along the new-minus-old text direction) to LAND
the state (guidance scale 2.5 over a short landing window), then holds
unguided: the weather persists through the KV history and the swapped
cross-attention text, so the steady-state cost of an active weather is a
single forward per step ("land-then-release", A/B'd 2026-08-21). Because
the transformer
routes any guided swap through the pre-merged text-edit LoRA when one is
attached, weather swaps must *bypass* the LoRA (it was trained on the four
style prompts, not weather); :class:`~.style_ability.StyleAbility` detaches
it around the ``replace_text`` call when ``use_lora`` is False.

Weather is a **base-world-only** ability (design decision 2026-08-20):
skin+weather combo prompts produced rain that was not attributable as rain
under the neon skins, so :class:`~.style_ability.StyleAbility` rejects the
weather key while a skin is active and clears weather when a skin is
activated. Exactly one of (skin, weather) is active at any time:

===========  ==========  ==============================  ========  =========
skin         weather     prompt                          LoRA      corrector
===========  ==========  ==============================  ========  =========
none         none        base scene prompt (plain 1/0)   off       off
active       none        skin prompt                     on        style gain
none         active      weather standalone prompt       BYPASS    weather gain*
===========  ==========  ==============================  ========  =========

``*`` the corrector gate profile was calibrated on style v6, not weather;
``LiveEditWeatherConfig.corrector_gain`` defaults to 0 (off) and can be
raised to a small absolute gain (e.g. 0.10) if base-world drift under a
long weather window proves worse than a mild corrector wash.
"""

from __future__ import annotations

from dataclasses import dataclass

from crazy_robotaxi.live_edit.config import (
    LiveEditStyleConfig,
    LiveEditWeatherConfig,
    StyleSkin,
    WeatherPreset,
)


@dataclass(frozen=True)
class SwapTarget:
    """One fully-resolved ``replace_text`` call plus its side policies."""

    prompt: str
    """Full prompt to swap in."""

    guidance_scale: float
    """``replace_text`` guidance scale (1.0 = plain swap)."""

    guidance_chunks: int
    """``replace_text`` guidance window length."""

    use_lora: bool
    """Whether the pre-merged text-edit LoRA may realize the window. False
    forces the two-prompt KV-snapshot guidance (LoRA detached for the call)."""

    corrector_gain: float
    """Absolute style-drift-corrector gain for this state (0 = corrector
    off, an exact base-forward short-circuit)."""


def compose_swap_target(
    *,
    base_prompt: str,
    skin: StyleSkin | None,
    weather: WeatherPreset | None,
    style_config: LiveEditStyleConfig,
    weather_config: LiveEditWeatherConfig | None,
    lora_available: bool,
) -> SwapTarget:
    """Resolve the single active prompt for a (skin | weather) state.

    Raises:
        ValueError: Both a skin and a weather are requested — weather is
            base-world-only; the :class:`~.style_ability.StyleAbility`
            state machine must never produce this combination.
    """
    if skin is not None and weather is not None:
        raise ValueError("weather is base-world-only and cannot compose with a skin")
    if skin is None and weather is None:
        # Plain swap back to the base world; guidance 1.0/0 also deactivates
        # the pre-merged edit LoRA.
        return SwapTarget(
            prompt=base_prompt,
            guidance_scale=1.0,
            guidance_chunks=0,
            use_lora=False,
            corrector_gain=0.0,
        )
    if skin is not None:
        return SwapTarget(
            prompt=skin.prompt,
            guidance_scale=style_config.guidance_scale,
            guidance_chunks=style_config.guidance_chunks,
            use_lora=lora_available,
            corrector_gain=style_config.corrector_gain,
        )
    assert weather is not None
    assert weather_config is not None, "weather state requires a weather config"
    return SwapTarget(
        prompt=weather.prompt,
        guidance_scale=weather_config.guidance_scale,
        guidance_chunks=weather_config.guidance_chunks,
        use_lora=False,
        corrector_gain=weather_config.corrector_gain,
    )
