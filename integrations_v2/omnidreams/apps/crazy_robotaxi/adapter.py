# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable Crazy Robotaxi application."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from crazy_robotaxi import CrazyRobotaxiApplication, CrazyRobotaxiApplicationDefaults
from crazy_robotaxi.settings import (
    default_settings,
    format_editor_value,
    iter_setting_fields,
)
from omnidreams.config import (
    OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG,
    OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
    OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
    OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG,
    OMNIDREAMS_RTX_5090_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication


def _preset_documentation() -> str:
    """Describe OmniDreams runner inheritance using the current config values."""
    presets = (
        (OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS, None),
        (OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS, OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_FAST_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_DEFAULTS,
        ),
        (
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS,
            OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_DEFAULTS,
        ),
    )
    lines = [
        "## OmniDreams runner presets",
        "",
        "Preset defaults are applied first, then sparse settings YAML overrides, "
        "then explicit application CLI arguments. Switching runners keeps saved "
        "overrides. Compare unmodified presets with a config containing only "
        "`schema_version: 1`, and omit setting overrides. RESET TO DEFAULTS followed "
        "by SAVE restores the selected runner's defaults on every Options page.",
        "",
        "The tables use current config values. Rows marked preset-owned have no "
        "Options editor or supported YAML override; their paths identify model "
        "config fields. All other paths are full YAML keys. Only differences from "
        "the indicated parent are listed; other values are inherited.",
        "",
        "| Runner | Based on | Render size (pixels) |",
        "| --- | --- | --- |",
    ]

    def slug(defaults: CrazyRobotaxiApplicationDefaults) -> str:
        return defaults.slug.replace("crazy-robotaxi", "crazy-robotaxi-omnidreams", 1)

    def collect(
        value: object, path: tuple[str, ...] = ()
    ) -> dict[tuple[str, ...], tuple[object, str]]:
        assert is_dataclass(value)
        values = {}
        editable = {item.name for item, _annotation in iter_setting_fields(value, path)}
        for item in fields(value):
            if item.name == "_target" or (*path, item.name) == (
                "model",
                "pipeline",
                "name",
            ):
                continue
            current = getattr(value, item.name)
            item_path = (*path, item.name)
            if is_dataclass(current) and not isinstance(current, type):
                values.update(collect(current, item_path))
            else:
                location = " → ".join(name.replace("_", " ").upper() for name in path)
                label = f"{item.name.replace('_', ' ').title()}:"
                values[item_path] = (
                    current,
                    f"{location} → **{label}**"
                    if item.name in editable
                    else "Preset-owned",
                )
        return values

    def settings(
        defaults: CrazyRobotaxiApplicationDefaults,
    ) -> dict[tuple[str, ...], tuple[object, str]]:
        return collect(
            default_settings(
                defaults.pipeline_config, width=defaults.width, height=defaults.height
            )
        )

    def row(path: tuple[str, ...], location: str, *values: object) -> str:
        formatted = [
            "null" if value is None else format_editor_value(value) for value in values
        ]
        formatted = [
            value.lower() if isinstance(original, bool) else value
            for value, original in zip(formatted, values)
        ]
        cells = [
            f"`{'.'.join(path)}`",
            location,
            *(f"`{value}`" for value in formatted),
        ]
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    for defaults, parent in presets:
        inherited = "—" if parent is None else f"`{slug(parent)}`"
        lines.append(
            f"| `{slug(defaults)}` | {inherited} | {defaults.width} × {defaults.height} |"
        )
    lines.extend(
        [
            "",
            "### Standard and shared values",
            "",
            "Every preset uses the single-view distilled OmniDreams checkpoint, "
            "Cosmos-Reason1 text encoding, LightVAE first-frame and HD-map input "
            "encoders, and the BF16 TAEHV/LightTAE video decoder. Native FP8 policies "
            "select quantized computation separately from the transformer's dtype.",
            "",
            "Raster width and height determine model resolution; `--width` and "
            "`--height` override them. Presentation dimensions start at `null` and "
            "follow model output; `--display-width` and `--display-height` change "
            "display size without increasing generation resolution.",
            "",
            "| YAML key / preset field | Options location | Standard value |",
            "| --- | --- | --- |",
        ]
    )
    standard = settings(OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS)
    for path, (value, location) in standard.items():
        if path[:2] == ("model", "pipeline") and path[-1] in {
            "seed",
            "window_size_t",
            "early_short_history_block_count",
            "len_t",
            "sink_size_t",
            "guidance_scale",
            "context_noise",
            "dtype",
            "compile_network",
            "use_compile",
            "use_cuda_graph",
            "skip_finalize_kv_cache",
            "native_dit_acceleration",
            "native_dit_backend",
            "native_dit_attention_backend",
            "apply_rope_before_kvcache",
            "self_attention_backend",
            "cross_attention_backend",
            "denoising_timesteps",
            "num_inference_steps",
            "run_on_cpu",
            "embedding_cache_size",
            "native_vae_acceleration",
            "native_vae_backend",
            "native_vae_fp8_state_path",
            "native_vae_fp8_auto_export",
            "qkv_fusion_option",
            "sdpa_backend",
            "use_tma",
            "projection",
            "quantized_sdpa",
        }:
            lines.append(row(path, location, value))
    for defaults, parent in presets:
        if parent is None:
            continue
        lines.extend(
            [
                "",
                f"### `{slug(defaults)}`",
                "",
                f"Inherits `{slug(parent)}` except for these values:",
                "",
                "| YAML key / preset field | Options location | Inherited value | Preset value |",
                "| --- | --- | --- | --- |",
            ]
        )
        inherited = settings(parent)
        for path, (value, location) in settings(defaults).items():
            previous = inherited[path][0]
            if value != previous:
                lines.append(row(path, location, previous, value))
    lines.extend(
        [
            "",
            "### What the preset choices mean",
            "",
            "- `-perf` requires native FP8 DiT; startup fails if unavailable. Its "
            "second denoising timestep changes from 500 to 100 and it skips the separate "
            "cache-finalization pass. These choices can change generated video. Native "
            "DiT uses its native attention backend; NETWORK attention fields are inactive.",
            "- `-fast-perf` inherits `-perf` and requires native FP8 LightVAE for both "
            "input encoders. Their float16 dtype describes the wrapper; the preset-owned "
            "FP8 backend selects computation. The decoder stays BF16. A null model seed "
            "uses the global random-number generator; `--model-seed` sets it explicitly.",
            "- Native VAE auto-export prepares and caches FP8 state when needed. A blank "
            "state path uses `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` if set, otherwise "
            "`artifacts/native_vae/lightvae_fp8_state.pt`.",
            "- The RTX 5090 presets fit in 32 GB VRAM by moving text encoding to CPU, "
            "caching eight prompt batches, and reducing history. `prefer_sage3_fp8` uses "
            "SageAttention-3 FP8 when supported and falls back to cuDNN, including on "
            "Windows. Native DiT remains required. The fast variant keeps native FP8 "
            "input encoders and a null model seed, with smaller render dimensions.",
            "- Optimized GB300 uses quantized cuDNN self-attention without FP8 "
            "projections and unquantized FlashAttention-2 cross-attention. RTX PRO 6000 "
            "uses FP8 self-attention projections and quantized FlashAttention-2. Its "
            "OmniDreams cross-attention leaves cross optimized implementation fields inactive.",
            "- Responsive variants inherit their matching base and use one-chunk "
            "history in the first nine of 28 transformer blocks, with cache-relative "
            "RoPE. Window Size T is measured in latent temporal units before "
            "patchification; each chunk has two latent frames. Native DiT is disabled "
            "because it cannot execute that combination. Native DiT backend fields "
            "are inactive; native FP8 input encoders and optimized network policies "
            "remain inherited. Smaller history alone does not enable this policy; "
            "there are no registered RTX 5090 responsive presets.",
            "",
            "Setting descriptions and available CLI flags appear in the options "
            "reference above. Model and render changes require a process restart.",
            "",
        ]
    )
    return "\n".join(lines)


OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi",
    slug="crazy-robotaxi",
    width=1280,
    height=704,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Perf)",
    slug="crazy-robotaxi-perf",
    width=1168,
    height=640,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_PERF_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Fast Perf)",
    slug="crazy-robotaxi-fast-perf",
    width=1168,
    height=640,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (RTX 5090)",
    slug="crazy-robotaxi-rtx-5090",
    width=1168,
    height=640,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_RTX_5090_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_FAST_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (RTX 5090 Fast)",
    slug="crazy-robotaxi-rtx-5090-fast",
    width=1024,
    height=560,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Optimized GB300)",
    slug="crazy-robotaxi-optimized-gb300",
    width=1280,
    height=704,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Optimized RTX PRO 6000)",
        slug="crazy-robotaxi-optimized-rtx-pro-6000",
        width=1280,
        height=704,
        preset_documentation=_preset_documentation,
        pipeline_config=OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG,
    )
)
OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Responsive)",
    slug="crazy-robotaxi-responsive",
    width=1280,
    height=704,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS = CrazyRobotaxiApplicationDefaults(
    title="Crazy Robotaxi (Perf Responsive)",
    slug="crazy-robotaxi-perf-responsive",
    width=1168,
    height=640,
    preset_documentation=_preset_documentation,
    pipeline_config=OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG,
)
OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Fast Perf Responsive)",
        slug="crazy-robotaxi-fast-perf-responsive",
        width=1168,
        height=640,
        preset_documentation=_preset_documentation,
        pipeline_config=OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG,
    )
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Optimized GB300 Responsive)",
        slug="crazy-robotaxi-optimized-gb300-responsive",
        width=1280,
        height=704,
        preset_documentation=_preset_documentation,
        pipeline_config=OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG,
    )
)
OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS = (
    CrazyRobotaxiApplicationDefaults(
        title="Crazy Robotaxi (Optimized RTX PRO 6000 Responsive)",
        slug="crazy-robotaxi-optimized-rtx-pro-6000-responsive",
        width=1280,
        height=704,
        preset_documentation=_preset_documentation,
        pipeline_config=(OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG),
    )
)


def create_app() -> IApplication:
    """Create Crazy Robotaxi with the regular OmniDreams config."""
    return CrazyRobotaxiApplication(defaults=OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS)


def create_perf_app() -> IApplication:
    """Create Crazy Robotaxi with the performance OmniDreams config."""
    return CrazyRobotaxiApplication(defaults=OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS)


def create_fast_perf_app() -> IApplication:
    """Create Crazy Robotaxi with fast OmniDreams acceleration when available."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS
    )


def create_rtx_5090_app() -> IApplication:
    """Create Crazy Robotaxi tuned to fit a 32 GiB GeForce RTX 5090."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_DEFAULTS
    )


def create_rtx_5090_fast_app() -> IApplication:
    """Create the RTX 5090 app with the native FP8 VAE at 1024x560."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_FAST_DEFAULTS
    )


def create_optimized_gb300_app() -> IApplication:
    """Create Crazy Robotaxi with the GB300-optimized attention policy."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_DEFAULTS
    )


def create_optimized_rtx_pro_6000_app() -> IApplication:
    """Create Crazy Robotaxi with the RTX PRO 6000 attention policy."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_DEFAULTS
    )


def create_responsive_app() -> IApplication:
    """Create Crazy Robotaxi with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS
    )


def create_perf_responsive_app() -> IApplication:
    """Create the performance app with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS
    )


def create_fast_perf_responsive_app() -> IApplication:
    """Create the native-VAE app with responsive early-block model history."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS
    )


def create_optimized_gb300_responsive_app() -> IApplication:
    """Create the responsive app with the GB300-optimized attention policy."""
    return CrazyRobotaxiApplication(
        defaults=OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS
    )


def create_optimized_rtx_pro_6000_responsive_app() -> IApplication:
    """Create the responsive app with the RTX PRO 6000 attention policy."""
    return CrazyRobotaxiApplication(
        defaults=(OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS)
    )


__all__ = [
    "OMNIDREAMS_CRAZY_ROBOTAXI_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_FAST_PERF_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_GB300_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_PERF_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_PERF_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_RESPONSIVE_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_DEFAULTS",
    "OMNIDREAMS_CRAZY_ROBOTAXI_RTX_5090_FAST_DEFAULTS",
    "create_app",
    "create_fast_perf_app",
    "create_fast_perf_responsive_app",
    "create_optimized_gb300_app",
    "create_optimized_gb300_responsive_app",
    "create_optimized_rtx_pro_6000_app",
    "create_optimized_rtx_pro_6000_responsive_app",
    "create_perf_app",
    "create_perf_responsive_app",
    "create_responsive_app",
    "create_rtx_5090_app",
    "create_rtx_5090_fast_app",
]
