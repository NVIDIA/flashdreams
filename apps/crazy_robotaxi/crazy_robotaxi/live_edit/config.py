# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration for the flag-gated live-edit abilities.

Follows the ``TaxiGameConfig`` pattern (frozen dataclass, docstring per
field, validation in ``__post_init__``) plus argparse helpers mirroring the
``--taxi-*`` flag flow in ``runtime_cli.py`` / ``app.taxi_config_from_args``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

_CORRECTOR_MODES = ("fused", "unfused")


@dataclass(frozen=True)
class StyleSkin:
    """One selectable world skin driven by a prompt swap."""

    name: str
    """Short HUD label, e.g. ``arcade``."""

    prompt: str
    """Full edit prompt swapped into the text cross-attention cache."""


# The v6 style LoRA is prompt-selected: these are the exact declarative
# prompts its four styles were trained on (edit_sft/style_prompts.py).
_DEFAULT_SKINS: tuple[StyleSkin, ...] = (
    StyleSkin(
        name="arcade",
        prompt=(
            "A bright arcade racing game world with exaggerated saturated "
            "colors, clean stylized surfaces, and a cheerful sunny palette."
        ),
    ),
    StyleSkin(
        name="comic",
        prompt=(
            "A comic book style world with bold black ink outlines, halftone "
            "shading, and vivid flat colors."
        ),
    ),
    StyleSkin(
        name="cyberpunk",
        prompt=(
            "A neon-lit cyberpunk night city with glowing signs and "
            "rain-slicked streets."
        ),
    ),
    StyleSkin(
        name="pixel",
        prompt=(
            "Retro 16-bit pixel art video game graphics with visible pixels "
            "and a bright limited color palette."
        ),
    ),
)


@dataclass(frozen=True)
class LiveEditStyleConfig:
    """Live game-skin switching (text-edit LoRA + drift corrector)."""

    enabled: bool = False
    """Whether the style ability is attached to the world-model session."""

    lora_checkpoint: Path | None = None
    """Pre-merged text-edit LoRA checkpoint (``guidance_distill`` format)."""

    corrector_checkpoint: Path | None = None
    """Style-drift corrector LoRA checkpoint (``train_v2`` format)."""

    corrector_gain: float = 0.15
    """Global corrector gain composed with the alpha*(t) gate profile."""

    corrector_mode: str = "fused"
    """Drift-corrector deploy mode. ``fused`` rides the CUDA-graph-safe
    per-state ``DriftCorrectorDispatch`` (compile_network + use_cuda_graph
    stay ON; validated 207 ms/chunk vs 203.9 no-corrector); ``unfused``
    falls back to the eager scale-gated path, which forces the graph-free
    pipeline (~1.4 s/chunk in-game). ``LIVE_EDIT_CORRECTOR_MODE`` sets the
    CLI default."""

    base_corrector_checkpoint: Path | None = None
    """Optional photoreal drift corrector for the BASE world state (fused
    mode only; the shipped ``lora_v2_v3_valpeak.pt`` deploy). ``None``
    leaves the base world uncorrected."""

    base_corrector_gain: float = 0.25
    """Gain for the base-state photoreal corrector (``corrgate025``)."""

    gate_alpha_json: Path | None = None
    """Measured per-timestep gate profile (``edit_sft/gate_style.py`` output)."""

    guidance_scale: float = 2.5
    """Edit-window strength marker for skin swaps. With the pre-merged edit
    LoRA deployed, any value > 1.0 (together with ``guidance_chunks`` > 0)
    opens the single-branch LoRA window; exactly 1.0 falls back to a plain
    swap, which *deactivates* the LoRA. 2.5/20 is the validated skin
    deployment from the smoke harness."""

    guidance_chunks: int = 20
    """Number of chunks the LoRA edit window stays open after a swap."""

    reswap_interval_chunks: int = 8
    """Re-issue the active skin's ``replace_text`` every N generated chunks.

    Long holds soften after ~8-10 chunks as the edit window ages out of the
    KV cache; a periodic duty-cycled re-swap keeps the style crisp. ``0``
    disables the refresh."""

    skins: tuple[StyleSkin, ...] = _DEFAULT_SKINS
    """Selectable skins, cycled by the switch-skin key."""

    def __post_init__(self) -> None:
        """Validate style values at configuration time."""
        if self.enabled and self.lora_checkpoint is None:
            raise ValueError("live_edit.style requires --live-edit-style-lora")
        if not 0.0 <= self.corrector_gain <= 1.0:
            raise ValueError("corrector_gain must be in [0, 1]")
        if self.corrector_mode not in _CORRECTOR_MODES:
            raise ValueError(f"corrector_mode must be one of {_CORRECTOR_MODES}")
        if not 0.0 <= self.base_corrector_gain <= 1.0:
            raise ValueError("base_corrector_gain must be in [0, 1]")
        if self.guidance_scale < 1.0:
            raise ValueError("guidance_scale must be at least 1.0")
        if self.guidance_chunks < 0:
            raise ValueError("guidance_chunks must be non-negative")
        if self.reswap_interval_chunks < 0:
            raise ValueError("reswap_interval_chunks must be non-negative")
        if self.enabled and not self.skins:
            raise ValueError("live_edit.style requires at least one skin")


@dataclass(frozen=True)
class WeatherPreset:
    """One selectable weather state driven by a prompt swap.

    Weather is a base-world-only ability (design decision 2026-08-20): it
    never composes with a skin prompt, so each preset carries exactly one
    standalone scene prompt.
    """

    name: str
    """Short HUD label, e.g. ``rain``."""

    prompt: str
    """Full standalone scene prompt describing the weather over the base
    world. Scene-native declarative phrasing lands much stronger than
    instruction-style wording (calibration sweeps, 2026-08-08)."""


# Daytime-rain phrasing follows the validated RAIN_NIGHT_NATIVE structure
# (sweep_text_edit.py) adapted to the daylight suburban scenes, with the
# visible-precipitation cues front-loaded (streaks in the air, droplets on
# the windshield/lens, tire spray) — the 2026-08-20 recapture showed wording
# that leans on wet-road looks alone reads as "no rain" to viewers. Snow
# extends the scene bundle's snowstorm wording with the same front-loaded
# falling-precipitation cues (heavy snowfall, flakes in the air, accumulation
# on the hood) after the 2.5-guidance capture read as a light dusting. Storm
# is an experimental heavy-weather preset: appearance cues (dark sky,
# torrential rain, fog, headlights) are expected to land; dynamic wind
# effects (bending trees, flying debris) are unlikely to materialize in a
# history-anchored world model and are included only as steering pressure.
_DEFAULT_WEATHERS: tuple[WeatherPreset, ...] = (
    WeatherPreset(
        name="rain",
        prompt=(
            "A dashcam perspective of a suburban street in a heavy daytime "
            "downpour under a dark gray overcast sky. Dense visible rain "
            "streaks slice through the air across the whole frame, and "
            "raindrops and water droplets bead and run down the windshield "
            "and camera lens. The asphalt road is saturated with sheeting "
            "water, a glossy wet mirror breaking up reflections, and mist "
            "and spray kick up from the tires of vehicles. The car's wet "
            "hood is covered with rain droplets. Photorealistic dashcam "
            "footage in pouring rain."
        ),
    ),
    WeatherPreset(
        name="snow",
        prompt=(
            "A dashcam perspective from inside a vehicle driving down a "
            "wide suburban residential street in heavy snowfall during a "
            "snowstorm. Thick white snowflakes fall densely and visibly "
            "through the air across the whole frame, streaking past the "
            "windshield. The road is heavily covered in white snow with "
            "visible parallel tire tracks, and fresh snow keeps "
            "accumulating on the asphalt. Vehicles parked along the curb "
            "and the roadsides are coated in a thick layer of snow. The "
            "surrounding houses, lawns, and large trees are completely "
            "blanketed in winter snow. The sky is a bright white-out "
            "overcast winter sky. In the foreground, the bottom of the "
            "windshield and the car's snow-dusted hood are visible, with "
            "thick snowflakes and snow accumulating on the hood and around "
            "the windshield wipers."
        ),
    ),
    WeatherPreset(
        name="storm",
        prompt=(
            "A dashcam perspective of a suburban street in a violent "
            "hurricane-force storm. Torrential rain hammers down in dense "
            "sheets, thick rain streaks slice through the air, and water "
            "sprays across the windshield and camera lens. The sky is a "
            "dark green-black wall of storm clouds, so dark that oncoming "
            "vehicles have their headlights on. Low fog and wind-driven "
            "mist blow across the road, trees bend hard in the violent "
            "wind, and loose leaves and debris fly through the air. The "
            "flooded asphalt sheets with water and heavy spray kicks up "
            "from the tires. Photorealistic dashcam footage inside a "
            "severe storm."
        ),
    ),
)


def weathers_starting_with(name: str | None) -> tuple[WeatherPreset, ...]:
    """Rotate the default presets so ``name`` leads the V-key cycle.

    The weather key steps clear -> presets in order -> clear, so putting a
    preset first lets one confirmed key press select it directly (no brief
    pass through the presets ahead of it in the default order).

    Raises:
        ValueError: ``name`` is not a known preset name.
    """
    if name is None:
        return _DEFAULT_WEATHERS
    names = [weather.name for weather in _DEFAULT_WEATHERS]
    if name not in names:
        raise ValueError(f"unknown weather preset {name!r}; choose from {names}")
    index = names.index(name)
    return _DEFAULT_WEATHERS[index:] + _DEFAULT_WEATHERS[:index]


@dataclass(frozen=True)
class LiveEditWeatherConfig:
    """Live weather events (plain guided prompt swaps, no LoRA needed).

    Weather is only available over the base world: the V key is ignored
    while a skin is active, and activating a skin clears any active
    weather (design decision 2026-08-20 — skin+weather combo prompts
    produced unattributable rain and were dropped).
    """

    enabled: bool = False
    """Whether the weather ability responds to the weather-cycle key."""

    guidance_scale: float = 2.5
    """Two-prompt edit-guidance strength for weather swaps (the PR #431
    mechanism: flow pushed along the new-minus-old text direction). 2.5/20
    is the validated skin deployment; earlier sweeps needed 3.0 for snow,
    so this is exposed as ``--live-edit-weather-guidance``."""

    guidance_chunks: int = 20
    """Number of chunks the two-prompt guidance window stays open."""

    corrector_gain: float = 0.0
    """Absolute style-drift-corrector gain while weather is active. ``0``
    keeps the corrector off during weather (its gate profile was calibrated
    on style v6, not weather, and base-world drift is mild); a small value
    such as 0.10 trades a possible mild wash for less late-run drift."""

    corrector_checkpoint: Path | None = None
    """Dedicated corrector checkpoint for the weather state (fused mode).
    ``None`` reuses the style corrector at :attr:`corrector_gain`."""

    weathers: tuple[WeatherPreset, ...] = _DEFAULT_WEATHERS
    """Selectable weathers, cycled clear -> rain -> snow -> storm -> clear
    by default; :func:`weathers_starting_with` rotates the order for direct
    one-press selection."""

    def __post_init__(self) -> None:
        """Validate weather values at configuration time."""
        if self.guidance_scale < 1.0:
            raise ValueError("weather guidance_scale must be at least 1.0")
        if self.guidance_chunks < 0:
            raise ValueError("weather guidance_chunks must be non-negative")
        if not 0.0 <= self.corrector_gain <= 1.0:
            raise ValueError("weather corrector_gain must be in [0, 1]")
        if self.enabled and not self.weathers:
            raise ValueError("live_edit.weather requires at least one preset")


@dataclass(frozen=True)
class LiveEditObstacleConfig:
    """Obstacle events: a real scene actor track cloned into the lane ahead.

    Synthetic boxes render correctly in the conditioning but never
    materialize (mask-verified 2026-08-10); a bit-faithful clone of a real
    perception track does, and MOVING clones materialize solidly
    (probe_moving_clone.py, 2026-08-13). The event clones a moving vehicle
    track from the scene bundle, retimes it to "now", and rigidly shifts it
    to start ahead of the ego.
    """

    enabled: bool = False
    """Whether the obstacle ability responds to the spawn key."""

    spawn_ahead_m: float = 16.0
    """Meters ahead of the ego (along its heading) where the clone starts."""

    lateral_m: float = 0.0
    """Meters to the left (+) / right (-) of the ego heading at spawn."""

    active_chunks: int = 10
    """Despawn after this many generated chunks (also capped by the
    template track's own coverage)."""

    min_drift_m: float = 15.0
    """Minimum ground-plane travel for a track to count as moving."""

    min_coverage_s: float = 4.0
    """Minimum template track duration."""

    length_range_m: tuple[float, float] = (3.4, 5.6)
    """Car-sized bbox length filter for template tracks."""

    collision_radius_m: float = 3.0
    """Ego XY distance at which a hit is logged (visual/log only; the
    clone is not registered with PhysX)."""

    guide_scale: float = 0.0
    """Box-axis guidance strength (flow extrapolated along the
    with-box/without-box conditioning direction). ``0`` disables the
    guidance hook entirely (clone renders at ghost strength); ``2.0`` is the
    validated in-game operating point (solid vehicle, in-box |diff| ~18 vs
    ~7 unguided, out-box clean; ``3.0`` breaks up at near range). Requires
    ``use_cuda_graph=False`` on the transformer."""

    annotate: bool = False
    """Draw the clone's projected 3D box outline into presented frames
    (evidence/demo aid)."""

    def __post_init__(self) -> None:
        """Validate obstacle values at configuration time."""
        if self.spawn_ahead_m <= 0.0:
            raise ValueError("spawn_ahead_m must be positive")
        if self.active_chunks <= 0:
            raise ValueError("active_chunks must be positive")
        if self.min_drift_m < 0.0:
            raise ValueError("min_drift_m must be non-negative")
        if self.guide_scale < 0.0:
            raise ValueError("guide_scale must be non-negative")
        if not 0.0 < self.length_range_m[0] <= self.length_range_m[1]:
            raise ValueError("length_range_m must be a positive (lo, hi) pair")


@dataclass(frozen=True)
class LiveEditCoinsConfig:
    """Collectible coin course composited into the presented frames."""

    enabled: bool = False
    """Whether coins are laid out, rendered, and collectible."""

    spacing_m: float = 25.0
    """Arc-length spacing between coin groups along each navigation lane."""

    group_offsets_m: tuple[float, ...] = (-1.1, 0.0, 1.1)
    """Lateral offsets of the coins in one group, metres across the lane."""

    hover_height_m: float = 0.8
    """Coin center height above the waypoint ground point."""

    coin_diameter_m: float = 0.62
    """World-space coin diameter used for sprite scaling."""

    pickup_radius_m: float = 2.5
    """XY distance at which the ego collects a coin."""

    points_per_coin: int = 50
    """Score awarded per collected coin (HUD counter only for now)."""

    max_render_distance_m: float = 120.0
    """Coins farther than this are not composited."""

    fade_start_distance_m: float = 100.0
    """Alpha ramps to zero between this distance and the render limit."""

    sprite_path: Path | None = None
    """RGBA coin sprite; ``None`` renders a procedural coin."""

    def __post_init__(self) -> None:
        """Validate coin values at configuration time."""
        if self.spacing_m <= 0.0:
            raise ValueError("coin spacing_m must be positive")
        if self.pickup_radius_m <= 0.0:
            raise ValueError("coin pickup_radius_m must be positive")
        if self.coin_diameter_m <= 0.0:
            raise ValueError("coin_diameter_m must be positive")
        if not 0.0 < self.fade_start_distance_m <= self.max_render_distance_m:
            raise ValueError(
                "fade_start_distance_m must be in (0, max_render_distance_m]"
            )


@dataclass(frozen=True)
class LiveEditConfig:
    """Top-level live-edit ability switchboard."""

    style: LiveEditStyleConfig = field(default_factory=LiveEditStyleConfig)
    """Live skin-switching ability."""

    coins: LiveEditCoinsConfig = field(default_factory=LiveEditCoinsConfig)
    """Coin-pickup ability."""

    weather: LiveEditWeatherConfig = field(default_factory=LiveEditWeatherConfig)
    """Weather-event ability."""

    obstacle: LiveEditObstacleConfig = field(default_factory=LiveEditObstacleConfig)
    """Obstacle-event ability."""

    sharpen_amount: float = 0.8
    """Unsharp-mask strength applied to styled frames (0 disables)."""

    sharpen_sigma: float = 2.0
    """Gaussian sigma of the unsharp mask."""

    @property
    def any_enabled(self) -> bool:
        """Return whether any ability needs the presenter wrapper."""
        return (
            self.style.enabled
            or self.coins.enabled
            or self.weather.enabled
            or self.obstacle.enabled
        )

    def __post_init__(self) -> None:
        """Validate presenter-filter values at configuration time."""
        if self.sharpen_amount < 0.0:
            raise ValueError("sharpen_amount must be non-negative")
        if self.sharpen_sigma <= 0.0:
            raise ValueError("sharpen_sigma must be positive")


def add_live_edit_args(parser: argparse.ArgumentParser) -> None:
    """Register the ``--live-edit-*`` flags next to the ``--taxi-*`` flags."""
    group = parser.add_argument_group("live edit")
    group.add_argument(
        "--live-edit-style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mid-run world-skin switching (requires the LoRA checkpoints).",
    )
    group.add_argument(
        "--live-edit-style-lora",
        type=Path,
        default=None,
        help="Pre-merged text-edit LoRA checkpoint for the style ability.",
    )
    group.add_argument(
        "--live-edit-style-corrector",
        type=Path,
        default=None,
        help="Style-drift corrector checkpoint (optional but recommended).",
    )
    group.add_argument(
        "--live-edit-style-gain",
        type=float,
        default=0.15,
        help="Drift-corrector gain (composed with the gate profile).",
    )
    group.add_argument(
        "--live-edit-corrector-mode",
        type=str,
        choices=_CORRECTOR_MODES,
        default=os.environ.get("LIVE_EDIT_CORRECTOR_MODE", "fused"),
        help=(
            "Drift-corrector deploy mode: 'fused' keeps CUDA graphs + "
            "compile_network on (real-time); 'unfused' is the old eager "
            "fallback. Env default: LIVE_EDIT_CORRECTOR_MODE."
        ),
    )
    group.add_argument(
        "--live-edit-base-corrector",
        type=Path,
        default=None,
        help=(
            "Photoreal drift-corrector checkpoint for the base world state "
            "(fused mode only; omit to leave the base world uncorrected)."
        ),
    )
    group.add_argument(
        "--live-edit-base-corrector-gain",
        type=float,
        default=0.25,
        help="Gain for the base-state photoreal corrector (shipped: 0.25).",
    )
    group.add_argument(
        "--live-edit-gate-alpha-json",
        type=Path,
        default=None,
        help="Measured per-timestep corrector gate profile JSON.",
    )
    group.add_argument(
        "--live-edit-style-reswap-chunks",
        type=int,
        default=8,
        help=(
            "Re-issue the active skin swap every N generated chunks so long "
            "holds stay crisp (0 disables the refresh)."
        ),
    )
    group.add_argument(
        "--live-edit-weather",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mid-run weather events (guided prompt swaps; V key).",
    )
    group.add_argument(
        "--live-edit-weather-guidance",
        type=float,
        default=2.5,
        help=(
            "Two-prompt guidance scale for weather swaps (2.5 = validated "
            "default; snow needed 3.0 in earlier sweeps)."
        ),
    )
    group.add_argument(
        "--live-edit-weather-first",
        type=str,
        default=None,
        help=(
            "Rotate the weather cycle so this preset comes first (direct "
            "one-press select, e.g. 'snow'; default keeps rain first)."
        ),
    )
    group.add_argument(
        "--live-edit-weather-corrector-gain",
        type=float,
        default=0.0,
        help=(
            "Absolute drift-corrector gain while weather is active "
            "(0 = corrector off during weather, the calibrated-safe default)."
        ),
    )
    group.add_argument(
        "--live-edit-weather-corrector",
        type=Path,
        default=None,
        help=(
            "Dedicated corrector checkpoint for the weather state (fused "
            "mode; default reuses the style corrector)."
        ),
    )
    group.add_argument(
        "--live-edit-obstacle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable obstacle events (cloned moving scene vehicle; O key).",
    )
    group.add_argument(
        "--live-edit-obstacle-ahead-m",
        type=float,
        default=16.0,
        help="Meters ahead of the ego where the obstacle clone starts.",
    )
    group.add_argument(
        "--live-edit-obstacle-chunks",
        type=int,
        default=10,
        help="Chunks the obstacle event stays active before despawn.",
    )
    group.add_argument(
        "--live-edit-obstacle-guide-scale",
        type=float,
        default=0.0,
        help=(
            "Box-axis guidance strength over the clone (0 disables; 3.0 was "
            "the fully-opaque probe operating point)."
        ),
    )
    group.add_argument(
        "--live-edit-obstacle-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the obstacle clone's projected box outline (evidence aid).",
    )
    group.add_argument(
        "--live-edit-coins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable collectible coins composited along the route.",
    )
    group.add_argument(
        "--live-edit-coin-sprite",
        type=Path,
        default=None,
        help="RGBA coin sprite path (default: procedural coin).",
    )


def live_edit_config_from_args(args: argparse.Namespace) -> LiveEditConfig:
    """Build the live-edit configuration at the application composition root."""
    return LiveEditConfig(
        style=LiveEditStyleConfig(
            enabled=bool(args.live_edit_style),
            lora_checkpoint=args.live_edit_style_lora,
            corrector_checkpoint=args.live_edit_style_corrector,
            corrector_gain=float(args.live_edit_style_gain),
            corrector_mode=str(args.live_edit_corrector_mode),
            base_corrector_checkpoint=args.live_edit_base_corrector,
            base_corrector_gain=float(args.live_edit_base_corrector_gain),
            gate_alpha_json=args.live_edit_gate_alpha_json,
            reswap_interval_chunks=int(args.live_edit_style_reswap_chunks),
        ),
        coins=LiveEditCoinsConfig(
            enabled=bool(args.live_edit_coins),
            sprite_path=args.live_edit_coin_sprite,
        ),
        weather=LiveEditWeatherConfig(
            enabled=bool(args.live_edit_weather),
            guidance_scale=float(args.live_edit_weather_guidance),
            corrector_gain=float(args.live_edit_weather_corrector_gain),
            corrector_checkpoint=args.live_edit_weather_corrector,
            weathers=weathers_starting_with(args.live_edit_weather_first),
        ),
        obstacle=LiveEditObstacleConfig(
            enabled=bool(args.live_edit_obstacle),
            spawn_ahead_m=float(args.live_edit_obstacle_ahead_m),
            active_chunks=int(args.live_edit_obstacle_chunks),
            guide_scale=float(args.live_edit_obstacle_guide_scale),
            annotate=bool(args.live_edit_obstacle_annotate),
        ),
    )
