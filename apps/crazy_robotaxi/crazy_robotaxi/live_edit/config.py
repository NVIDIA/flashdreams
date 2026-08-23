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

_CORRECTOR_MODES = ("fused", "unfused", "off")


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
    pipeline (~1.4 s/chunk in-game); ``off`` disables every corrector even
    when checkpoints are configured — no corrector machinery is built and
    no transformer weights are snapshotted or copied.
    ``LIVE_EDIT_CORRECTOR_MODE`` sets the CLI default."""

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

    guidance_chunks: int = 6
    """Number of chunks the LoRA edit window stays open after a swap.

    With the pre-merged edit LoRA deployed this window is realized
    single-branch (merged weights toggled at the boundaries), so its length
    is NOT a per-chunk cost — only a stacked deployment without the LoRA
    would fall back to the two-prompt guidance whose window doubles the
    chunk cost. A/B on a cyberpunk swap (2026-08-21): 6 lands the style as
    fast and as strong as the old 20 (the 8-chunk re-swap refresh re-opens
    the window before long holds soften), so 6 is the default; it also caps
    the exposure of any stacked no-LoRA deployment to the 2x window.
    Exposed as ``--live-edit-skin-guidance-chunks``."""

    reswap_interval_chunks: int = 8
    """Re-issue the active skin's ``replace_text`` every N generated chunks.

    Long holds soften after ~8-10 chunks as the edit window ages out of the
    KV cache; a periodic duty-cycled re-swap keeps the style crisp. ``0``
    disables the refresh. Skipped entirely when a timed skin
    (:attr:`skin_duration_chunks`) expires at or before the first refresh
    would fire — the re-swap would land on an already-reverted world."""

    skin_duration_chunks: int = 0
    """Timed "power-up" mode: auto-revert an activated skin to the base
    world after this many generated chunks (at a chunk boundary, through
    the same plain-swap revert path the K cycle uses). ``0`` (default)
    keeps the current hold-until-cycled behavior. 11 chunks is ~3 s at the
    shipped 8-frames-per-chunk / 30 fps recipe. Pressing K while a timed
    skin is active cycles to the NEXT skin with a fresh timer (same K
    semantics as untimed mode; mashing K to extend simply re-lands the
    cycle). Exposed as ``--live-edit-skin-duration-chunks``. Also holds
    ~10+ chunk scene-content drift in check: the skin never outlives the
    crisp window."""

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
        if self.skin_duration_chunks < 0:
            raise ValueError("skin_duration_chunks must be non-negative")
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
# Hurricane escalates storm along the axes that DO land (2026-08-21 A/B):
# visibility collapse, spray/mist walls, debris lying statically ON the
# flooded road, black-green sky — no flying-debris or bending-tree wording,
# which never materializes.
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
    WeatherPreset(
        name="hurricane",
        prompt=(
            "A dashcam perspective of a suburban street in the eyewall of "
            "a landfalling hurricane, visibility collapsed to almost "
            "nothing. Blinding torrential rain bands and solid walls of "
            "white spray and mist swallow the street, so only the nearest "
            "stretch of road is visible before everything dissolves into "
            "gray-white murk. Fallen tree branches, palm fronds, leaves, "
            "and scattered debris litter the flooded road surface, lying "
            "across the lanes in standing water. The sky is an oppressive "
            "black-green hurricane sky, dark as night at midday, and the "
            "whole scene is drowned in emergency gloom. Oncoming headlights "
            "smear into halos through the deluge, windshield wipers thrash "
            "at full speed, and sheets of water crash over the windshield "
            "and camera lens. Photorealistic dashcam footage inside a "
            "catastrophic hurricane."
        ),
    ),
)


def skins_starting_with(name: str | None) -> tuple[StyleSkin, ...]:
    """Rotate the default skins so ``name`` leads the K-key cycle.

    Mirrors :func:`weathers_starting_with`: one confirmed K press selects
    the named skin directly — important for timed power-up demos where
    cycling through the skins ahead of it would burn transitional chunks.

    Raises:
        ValueError: ``name`` is not a known skin name.
    """
    if name is None:
        return _DEFAULT_SKINS
    names = [skin.name for skin in _DEFAULT_SKINS]
    if name not in names:
        raise ValueError(f"unknown skin {name!r}; choose from {names}")
    index = names.index(name)
    return _DEFAULT_SKINS[index:] + _DEFAULT_SKINS[:index]


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

    guidance_chunks: int = 6
    """Number of chunks the two-prompt LANDING window stays open.

    TRANSIENT COST: weather has no LoRA, so every denoise step inside this
    window runs a second network forward — a swap costs ~2x per chunk for
    this many chunks. Weather then persists through the KV history and the
    swapped cross-attention text with NO guidance ("land-then-release",
    A/B'd 2026-08-21: a 6-chunk landing matches the old always-guided 20
    within noise over a 27-chunk hold), so the steady-state cost of an
    active weather is ~1x. Exposed as
    ``--live-edit-weather-guidance-chunks``."""

    maintain_interval_chunks: int = 0
    """Re-open a short guidance window every N chunks while weather holds.

    ``0`` (default) holds with no guidance at all — the validated
    land-then-release policy. A positive interval issues a maintenance
    pulse of :attr:`maintain_chunks` guided chunks every N chunks, REBASED
    first (plain swap to the base prompt, then the guided weather swap):
    a same-prompt re-swap snapshots its old KV from buffers that already
    hold the weather text, making the guidance direction exactly zero —
    pure wasted 2x (this is what the old style re-swap refresh did to
    weather). Exposed as ``--live-edit-weather-maintain-interval``."""

    maintain_chunks: int = 2
    """Guided chunks per maintenance pulse (used when
    :attr:`maintain_interval_chunks` > 0). Exposed as
    ``--live-edit-weather-maintain-chunks``."""

    duration_chunks: int = 90
    """Timed weather: auto-revert an active weather to clear after this many
    generated chunks (~24 s at the shipped 8-frames-per-chunk / 30 fps
    recipe). Applies to every activation path (V key and pickup items).
    ``0`` holds until cycled. The revert lands GUIDED (see
    :attr:`clear_guidance_chunks`): unlike a skin revert, clear is itself a
    weather transition and a plain swap leaves the precipitation running on
    KV-history momentum. Accepted physics: the revert stops NEW
    precipitation but does not undo accumulated scene change — wet roads dry
    gradually and snow lingers then fades, which reads as realistic weather
    passing. Exposed as ``--live-edit-weather-duration-chunks``."""

    clear_guidance_chunks: int = 8
    """Guided chunks for the weather -> clear landing (both the timed
    auto-revert and a V-cycle wrap to clear). Slightly longer than the
    6-chunk activation landing because dense states (hurricane fog walls)
    dissipate slower than they land. Exposed as
    ``--live-edit-weather-clear-guidance-chunks``."""

    corrector_gain: float = 0.0
    """Absolute style-drift-corrector gain while weather is active. ``0``
    (default) keeps the corrector off during weather — policy decision
    2026-08-23: the clean-forcing corrector runs ONLY for game-skin states
    (0.15), base and weather states stay uncorrected. A/B note: 0.10
    measured slightly crisper late-run under long weather holds, but with
    timed weather (~24 s default) the window is short, so the knob stays
    for A/B while the default is off."""

    corrector_checkpoint: Path | None = None
    """Dedicated corrector checkpoint for the weather state (fused mode).
    ``None`` reuses the style corrector at :attr:`corrector_gain`."""

    weathers: tuple[WeatherPreset, ...] = _DEFAULT_WEATHERS
    """Selectable weathers, cycled clear -> rain -> snow -> storm ->
    hurricane -> clear by default; :func:`weathers_starting_with` rotates the order for direct
    one-press selection."""

    def __post_init__(self) -> None:
        """Validate weather values at configuration time."""
        if self.guidance_scale < 1.0:
            raise ValueError("weather guidance_scale must be at least 1.0")
        if self.guidance_chunks < 0:
            raise ValueError("weather guidance_chunks must be non-negative")
        if self.maintain_interval_chunks < 0:
            raise ValueError("weather maintain_interval_chunks must be non-negative")
        if self.maintain_chunks < 0:
            raise ValueError("weather maintain_chunks must be non-negative")
        if self.duration_chunks < 0:
            raise ValueError("weather duration_chunks must be non-negative")
        if self.clear_guidance_chunks < 0:
            raise ValueError("weather clear_guidance_chunks must be non-negative")
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

    count: int = 1
    """Clones per spawn request. ``1`` is the classic single obstacle; more
    makes a "traffic" event: each clone uses a DIFFERENT crossing/oncoming
    template track (never a pace-matched lead car — those render at ghost
    strength) staggered ahead of the ego by :attr:`spacing_m`."""

    spawn_ahead_m: float = 16.0
    """Meters ahead of the ego (along its heading) where the first clone
    starts; clone ``i`` starts ``i * spacing_m`` further out."""

    spacing_m: float = 8.0
    """Extra ahead-distance per additional clone (count > 1). The default
    puts a 4-clone burst across a 16-40 m band — the model's validated
    materialization range."""

    stagger_chunks: int = 1
    """Chunks between consecutive clone spawns in one burst. ``0`` spawns
    the whole burst in one chunk; a small stagger both eases the model into
    the event and spreads the passes out on screen."""

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

    static_count: int = 0
    """Static roadblock: this many PARKED-track clones placed midroad ahead
    of the spawn pose, in the conditioning from the session's first chunk,
    persisting until a rollout reset (which re-anchors them). Slots start
    ``static_ahead_m`` out, ``spacing_m`` apart, laterals alternating
    right/left by ``static_lateral_m`` so the ego can weave between them.
    Pair with ``guide_scale`` ~2.0: probed 2026-08-23 — unguided static
    clones render at ghost strength even when present from chunk 0 (the
    initial camera frame shows the road empty), s=2.0 materializes solid
    stopped cars in the 5-25 m band; mid-stream static spawns at s=2.5
    break up. ``0`` disables."""

    static_ahead_m: float = 28.0
    """Meters ahead of the spawn pose where the first static clone sits
    (nearer slots fight the initial frame hardest and stay ghost)."""

    static_lateral_m: float = 2.8
    """Lateral offset magnitude of the alternating static-clone slots."""

    guide_scale: float = 0.0
    """Box-axis guidance strength (flow extrapolated along the
    with-box/without-box conditioning direction). ``0`` disables the
    guidance hook entirely (clone renders at ghost strength); ``2.0`` is the
    validated in-game operating point (solid vehicle, in-box |diff| ~18 vs
    ~7 unguided, out-box clean; ``3.0`` breaks up at near range).
    CUDA-graph safe (2026-08-21): during an event each denoise step replays
    the captured graph twice with the box/no-box conditioning staged in, so
    event chunks cost ~2x model time and non-event chunks are unchanged; no
    graph-free rebuild. Not wired for the native optimized-DiT executor."""

    annotate: bool = False
    """Draw the clone's projected 3D box outline into presented frames
    (evidence/demo aid)."""

    def __post_init__(self) -> None:
        """Validate obstacle values at configuration time."""
        if self.count < 1:
            raise ValueError("obstacle count must be at least 1")
        if self.spacing_m <= 0.0:
            raise ValueError("spacing_m must be positive")
        if self.stagger_chunks < 0:
            raise ValueError("stagger_chunks must be non-negative")
        if self.spawn_ahead_m <= 0.0:
            raise ValueError("spawn_ahead_m must be positive")
        if self.active_chunks <= 0:
            raise ValueError("active_chunks must be positive")
        if self.min_drift_m < 0.0:
            raise ValueError("min_drift_m must be non-negative")
        if self.static_count < 0:
            raise ValueError("static_count must be non-negative")
        if self.static_ahead_m <= 0.0:
            raise ValueError("static_ahead_m must be positive")
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

    max_visible_sprites: int = 64
    """Composite at most this many coins per frame, keeping the nearest.

    Dense courses put hundreds of coins inside the render radius (the
    shipped suburb course peaks at 211), and the compositor's per-frame
    cost is launch-bound per sprite — unbounded sprite counts are what
    actually blow the frame budget. The dropped coins are the farthest
    (small, distance-faded) ones. ``0`` disables the cap."""

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
        if self.max_visible_sprites < 0:
            raise ValueError("max_visible_sprites must be non-negative")


ITEM_TYPES = ("rain", "snow", "mystery")
"""Effect-item kinds, in course-layout cycle order."""


@dataclass(frozen=True)
class LiveEditItemsConfig:
    """Sparse pickup items along the lanes that trigger live-edit effects.

    Reuses the coin machinery (lane-course layout, FTheta projection,
    proximity pickup, GPU compositing) but places one item every
    :attr:`spacing_m` instead of a coin row every 25 m. Picking an item up
    routes the effect through the existing ability state machines at the
    next chunk boundary — the exact path the K/V key requests take; the
    keys stay fully functional alongside.
    """

    enabled: bool = False
    """Whether effect items are laid out, rendered, and collectible."""

    spacing_m: float = 200.0
    """Arc-length spacing between items along each navigation lane (items
    are rare by design; 150-300 m is the intended range)."""

    hover_height_m: float = 1.0
    """Item center height above the waypoint ground point."""

    item_diameter_m: float = 0.9
    """World-space item height used for sprite scaling (bigger than a coin
    so the rare pickups read from a distance)."""

    pickup_radius_m: float = 2.5
    """XY distance at which the ego collects an item."""

    max_render_distance_m: float = 120.0
    """Items farther than this are not composited."""

    fade_start_distance_m: float = 100.0
    """Alpha ramps to zero between this distance and the render limit."""

    rain_sprite_path: Path | None = None
    """RGBA rain-item sprite; ``None`` renders a procedural placeholder.
    Sprite files are local-only paths, never bundled (coin-sprite pattern)."""

    snow_sprite_path: Path | None = None
    """RGBA snow-item sprite; ``None`` renders a procedural placeholder."""

    mystery_sprite_path: Path | None = None
    """RGBA mystery-box sprite; ``None`` renders a procedural '?' box."""

    mystery_burst_chunks: int = 11
    """Timed-skin duration granted by a mystery box (~3 s at the shipped
    recipe). Overrides the global ``skin_duration_chunks`` per activation so
    the box grants a burst even when the global mode is hold-forever (0);
    ``0`` makes the granted skin untimed."""

    mystery_seed: int | None = None
    """Seed for the mystery-box skin roll (reproducible captures); ``None``
    draws from the OS entropy pool. Re-seeded per rollout."""

    flash_seconds: float = 2.5
    """How long the pickup HUD flash chip stays up."""

    def __post_init__(self) -> None:
        """Validate item values at configuration time."""
        if self.spacing_m <= 0.0:
            raise ValueError("item spacing_m must be positive")
        if self.pickup_radius_m <= 0.0:
            raise ValueError("item pickup_radius_m must be positive")
        if self.item_diameter_m <= 0.0:
            raise ValueError("item_diameter_m must be positive")
        if not 0.0 < self.fade_start_distance_m <= self.max_render_distance_m:
            raise ValueError(
                "item fade_start_distance_m must be in (0, max_render_distance_m]"
            )
        if self.mystery_burst_chunks < 0:
            raise ValueError("mystery_burst_chunks must be non-negative")
        if self.flash_seconds <= 0.0:
            raise ValueError("flash_seconds must be positive")

    def sprite_path(self, item_type: str) -> Path | None:
        """Configured sprite path for one item type (``None`` = procedural)."""
        paths = {
            "rain": self.rain_sprite_path,
            "snow": self.snow_sprite_path,
            "mystery": self.mystery_sprite_path,
        }
        if item_type not in paths:
            raise ValueError(f"unknown item type {item_type!r}")
        return paths[item_type]


@dataclass(frozen=True)
class LiveEditConfig:
    """Top-level live-edit ability switchboard."""

    style: LiveEditStyleConfig = field(default_factory=LiveEditStyleConfig)
    """Live skin-switching ability."""

    coins: LiveEditCoinsConfig = field(default_factory=LiveEditCoinsConfig)
    """Coin-pickup ability."""

    items: LiveEditItemsConfig = field(default_factory=LiveEditItemsConfig)
    """Effect-item pickup ability."""

    weather: LiveEditWeatherConfig = field(default_factory=LiveEditWeatherConfig)
    """Weather-event ability."""

    obstacle: LiveEditObstacleConfig = field(default_factory=LiveEditObstacleConfig)
    """Obstacle-event ability."""

    sharpen_amount: float = 0.8
    """Unsharp-mask strength applied to styled frames (0 disables)."""

    sharpen_sigma: float = 2.0
    """Gaussian sigma of the unsharp mask."""

    perf_log_every_frames: int = 0
    """Log p50/p95 of the live-edit per-frame costs (coin-update CPU ms,
    compositor enqueue CPU ms, compositor GPU ms) every N composited frames
    on the tensor path. ``0`` disables the report. Exposed as
    ``--live-edit-perf-log``; ``LIVE_EDIT_PERF_LOG`` sets the CLI default."""

    @property
    def any_enabled(self) -> bool:
        """Return whether any ability needs the presenter wrapper."""
        return (
            self.style.enabled
            or self.coins.enabled
            or self.items.enabled
            or self.weather.enabled
            or self.obstacle.enabled
        )

    def __post_init__(self) -> None:
        """Validate presenter-filter values at configuration time."""
        if self.sharpen_amount < 0.0:
            raise ValueError("sharpen_amount must be non-negative")
        if self.sharpen_sigma <= 0.0:
            raise ValueError("sharpen_sigma must be positive")
        if self.perf_log_every_frames < 0:
            raise ValueError("perf_log_every_frames must be non-negative")


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
            "fallback; 'off' disables every corrector (no transformer "
            "weights are touched even if corrector checkpoints are given). "
            "Env default: LIVE_EDIT_CORRECTOR_MODE."
        ),
    )
    group.add_argument(
        "--live-edit-skin-guidance-chunks",
        type=int,
        default=6,
        help=(
            "Chunks the skin edit window stays open after a swap (the "
            "pre-merged LoRA realizes it single-branch, so this is not a "
            "per-chunk cost; the 8-chunk re-swap refresh re-opens it)."
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
        "--live-edit-skin-first",
        type=str,
        default=None,
        help=(
            "Rotate the skin cycle so this skin comes first (direct "
            "one-press select, e.g. 'cyberpunk'; default keeps arcade first)."
        ),
    )
    group.add_argument(
        "--live-edit-skin-duration-chunks",
        type=int,
        default=0,
        help=(
            "Timed power-up mode: auto-revert an activated skin to the base "
            "world after N generated chunks (0 = hold until cycled, the "
            "default; 11 is ~3 s at 8 frames/chunk, 30 fps)."
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
        "--live-edit-weather-guidance-chunks",
        type=int,
        default=6,
        help=(
            "Chunks the two-prompt weather LANDING window stays open (2x "
            "model cost per guided chunk). After the landing the weather "
            "holds unguided at ~1x (land-then-release)."
        ),
    )
    group.add_argument(
        "--live-edit-weather-maintain-interval",
        type=int,
        default=0,
        help=(
            "Re-open a short rebased guidance window every N chunks while "
            "weather holds (0 = plain hold, the validated default)."
        ),
    )
    group.add_argument(
        "--live-edit-weather-maintain-chunks",
        type=int,
        default=2,
        help="Guided chunks per weather maintenance pulse.",
    )
    group.add_argument(
        "--live-edit-weather-duration-chunks",
        type=int,
        default=90,
        help=(
            "Timed weather: auto-revert to clear after N generated chunks "
            "via a guided clear landing (~24 s at 8 frames/chunk, 30 fps; "
            "0 = hold until cycled). Applies to V-key and item pickups."
        ),
    )
    group.add_argument(
        "--live-edit-weather-clear-guidance-chunks",
        type=int,
        default=8,
        help=(
            "Guided chunks for the weather->clear landing (auto-revert and "
            "V-cycle wrap; a bit longer than the activation landing so "
            "dense states like hurricane fog dissipate)."
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
            "Absolute drift-corrector gain while weather is active. Default "
            "0 = off (policy: the clean-forcing corrector runs only for "
            "game-skin states; 0.10 was slightly crisper on long holds but "
            "timed weather keeps windows short). Knob kept for A/B."
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
        "--live-edit-obstacle-count",
        type=int,
        default=1,
        help=(
            "Clones per obstacle spawn (1 = single obstacle; 3-5 makes a "
            "traffic event of distinct crossing/oncoming vehicles staggered "
            "ahead of the ego)."
        ),
    )
    group.add_argument(
        "--live-edit-obstacle-stagger-chunks",
        type=int,
        default=1,
        help=(
            "Chunks between consecutive clone spawns in one burst (0 = all at once)."
        ),
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
        "--live-edit-obstacle-static-count",
        type=int,
        default=0,
        help=(
            "Static roadblock: N parked-track clones placed midroad from the "
            "session's first chunk (alternating laterals; pair with "
            "--live-edit-obstacle-guide-scale 2.0; 0 disables)."
        ),
    )
    group.add_argument(
        "--live-edit-obstacle-static-ahead-m",
        type=float,
        default=28.0,
        help="Meters ahead of spawn where the first static clone sits.",
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
    group.add_argument(
        "--live-edit-coin-max-visible",
        type=int,
        default=64,
        help=(
            "Composite at most this many coins per frame (nearest win; the "
            "farthest, distance-faded ones drop first; 0 disables the cap)."
        ),
    )
    group.add_argument(
        "--live-edit-items",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable sparse effect-pickup items along the route (rain/snow "
            "icons trigger weather, mystery boxes a random timed skin burst)."
        ),
    )
    group.add_argument(
        "--live-edit-item-spacing",
        type=float,
        default=200.0,
        help="Arc-length spacing between effect items per lane, metres.",
    )
    group.add_argument(
        "--live-edit-item-rain-sprite",
        type=Path,
        default=None,
        help="RGBA rain-item sprite path (default: procedural placeholder).",
    )
    group.add_argument(
        "--live-edit-item-snow-sprite",
        type=Path,
        default=None,
        help="RGBA snow-item sprite path (default: procedural placeholder).",
    )
    group.add_argument(
        "--live-edit-item-mystery-sprite",
        type=Path,
        default=None,
        help="RGBA mystery-box sprite path (default: procedural '?' box).",
    )
    group.add_argument(
        "--live-edit-item-mystery-burst-chunks",
        type=int,
        default=11,
        help=(
            "Timed-skin duration a mystery box grants (overrides the global "
            "skin duration per activation; 0 = untimed)."
        ),
    )
    group.add_argument(
        "--live-edit-item-mystery-seed",
        type=int,
        default=None,
        help="Seed for the mystery-box skin roll (reproducible captures).",
    )
    group.add_argument(
        "--live-edit-perf-log",
        type=int,
        default=int(os.environ.get("LIVE_EDIT_PERF_LOG", "0")),
        help=(
            "Log p50/p95 of the live-edit per-frame costs (coin-update CPU "
            "ms, compositor enqueue CPU ms, compositor GPU ms) every N "
            "composited frames (0 disables; env default: LIVE_EDIT_PERF_LOG)."
        ),
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
            guidance_chunks=int(args.live_edit_skin_guidance_chunks),
            reswap_interval_chunks=int(args.live_edit_style_reswap_chunks),
            skin_duration_chunks=int(args.live_edit_skin_duration_chunks),
            skins=skins_starting_with(args.live_edit_skin_first),
        ),
        coins=LiveEditCoinsConfig(
            enabled=bool(args.live_edit_coins),
            sprite_path=args.live_edit_coin_sprite,
            max_visible_sprites=int(args.live_edit_coin_max_visible),
        ),
        items=LiveEditItemsConfig(
            enabled=bool(args.live_edit_items),
            spacing_m=float(args.live_edit_item_spacing),
            rain_sprite_path=args.live_edit_item_rain_sprite,
            snow_sprite_path=args.live_edit_item_snow_sprite,
            mystery_sprite_path=args.live_edit_item_mystery_sprite,
            mystery_burst_chunks=int(args.live_edit_item_mystery_burst_chunks),
            mystery_seed=(
                None
                if args.live_edit_item_mystery_seed is None
                else int(args.live_edit_item_mystery_seed)
            ),
        ),
        weather=LiveEditWeatherConfig(
            enabled=bool(args.live_edit_weather),
            guidance_scale=float(args.live_edit_weather_guidance),
            guidance_chunks=int(args.live_edit_weather_guidance_chunks),
            maintain_interval_chunks=int(args.live_edit_weather_maintain_interval),
            maintain_chunks=int(args.live_edit_weather_maintain_chunks),
            duration_chunks=int(args.live_edit_weather_duration_chunks),
            clear_guidance_chunks=int(args.live_edit_weather_clear_guidance_chunks),
            corrector_gain=float(args.live_edit_weather_corrector_gain),
            corrector_checkpoint=args.live_edit_weather_corrector,
            weathers=weathers_starting_with(args.live_edit_weather_first),
        ),
        perf_log_every_frames=int(args.live_edit_perf_log),
        obstacle=LiveEditObstacleConfig(
            enabled=bool(args.live_edit_obstacle),
            count=int(args.live_edit_obstacle_count),
            stagger_chunks=int(args.live_edit_obstacle_stagger_chunks),
            spawn_ahead_m=float(args.live_edit_obstacle_ahead_m),
            active_chunks=int(args.live_edit_obstacle_chunks),
            static_count=int(args.live_edit_obstacle_static_count),
            static_ahead_m=float(args.live_edit_obstacle_static_ahead_m),
            guide_scale=float(args.live_edit_obstacle_guide_scale),
            annotate=bool(args.live_edit_obstacle_annotate),
        ),
    )
