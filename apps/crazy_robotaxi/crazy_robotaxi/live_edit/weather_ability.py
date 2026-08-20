# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weather events: composing the prompt swap for skin/weather combinations.

There is no weather LoRA. Weather uses the plain two-prompt edit-guidance
mechanism (the PR #431 ``replace_text`` path: the old prompt anchors the
scene, the flow is pushed along the new-minus-old text direction) with the
same validated 2.5/20 kwargs as the skin deployment. Because the transformer
routes any guided swap through the pre-merged text-edit LoRA when one is
attached, weather-only swaps must *bypass* the LoRA (it was trained on the
four style prompts, not weather); :class:`~.style_ability.StyleAbility`
detaches it around the ``replace_text`` call when ``use_lora`` is False.

Composition matrix (single active prompt at any time):

===========  ==========  ==============================  ========  =========
skin         weather     prompt                          LoRA      corrector
===========  ==========  ==============================  ========  =========
none         none        base scene prompt (plain 1/0)   off       off
active       none        skin prompt                     on        on
none         active      weather standalone prompt       BYPASS    off*
active       active      skin prompt + weather clause    on        off*
===========  ==========  ==============================  ========  =========

``*`` the corrector gate profile was calibrated on style v6, not weather;
``LiveEditWeatherConfig.allow_corrector`` keeps it on only after a
weather+corrector bring-up shows no artifacts.
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

    corrector_enabled: bool
    """Whether the style-drift corrector should run for this state."""


def compose_swap_target(
    *,
    base_prompt: str,
    skin: StyleSkin | None,
    weather: WeatherPreset | None,
    style_config: LiveEditStyleConfig,
    weather_config: LiveEditWeatherConfig | None,
    lora_available: bool,
) -> SwapTarget:
    """Resolve the single active prompt for a (skin, weather) state."""
    if skin is None and weather is None:
        # Plain swap back to the base world; guidance 1.0/0 also deactivates
        # the pre-merged edit LoRA.
        return SwapTarget(
            prompt=base_prompt,
            guidance_scale=1.0,
            guidance_chunks=0,
            use_lora=False,
            corrector_enabled=False,
        )
    if skin is not None and weather is None:
        return SwapTarget(
            prompt=skin.prompt,
            guidance_scale=style_config.guidance_scale,
            guidance_chunks=style_config.guidance_chunks,
            use_lora=lora_available,
            corrector_enabled=True,
        )
    assert weather_config is not None, "weather state requires a weather config"
    if skin is None:
        return SwapTarget(
            prompt=weather.prompt,
            guidance_scale=weather_config.guidance_scale,
            guidance_chunks=weather_config.guidance_chunks,
            use_lora=False,
            corrector_enabled=False,
        )
    # Combo: one compositional prompt describes both; the style LoRA (when
    # present) realizes the window since the skin needs it.
    return SwapTarget(
        prompt=f"{skin.prompt} {weather.combo_clause}",
        guidance_scale=style_config.guidance_scale,
        guidance_chunks=style_config.guidance_chunks,
        use_lora=lora_available,
        corrector_enabled=weather_config.allow_corrector,
    )
