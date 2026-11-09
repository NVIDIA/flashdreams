# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint settings of the Cosmos3-Nano-Sim-Transfer release.

Every model setting is read from the export's own metadata, not restated here:
``checkpoint.json`` (falling back to ``transformer/config.json``) carries them
under :data:`EXPORT_METADATA_KEY`. The key names the export flow, not the model;
the release's ``cosmos_dreams_*`` files describe a different checkpoint and are
ignored. Only the history profile that the release documents for long rollouts
is defined here, as a named preset beside the export's own window and sinks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

HF_REPO_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
"""Hugging Face repository of the released checkpoint."""

HF_REVISION = "42da1b57840cec0fd2b642546b72cb95fdc331af"
"""Pinned repository commit; its weights load strictly through ``cosmos_interactive``."""

EXPORT_METADATA_KEY = "cosmos3_nano_sim_bimanual"
"""Key of the export metadata block in ``checkpoint.json`` and ``transformer/config.json``."""

CONTROL_ATTENTION_MODE = "causal_control_with_rgb_history"
"""The only control attention policy this rollout implements."""

CONTROL_HINTS = ("edge", "blur", "depth", "seg")
"""Control video kinds this package can resize (see ``control.resize_control``)."""

SYSTEM_PROMPT_ID = "cosmos3_transfer_v1"
"""The only system prompt this package builds (``prompt.SYSTEM_PROMPT`` is its text)."""

RELEASE_FILES = (
    "checkpoint.json",
    "model.safetensors.index.json",
    "transformer/*",
    "text_tokenizer/*",
)
"""Release files the rollout needs (weights, metadata and tokenizer)."""


@dataclass(frozen=True)
class HistoryProfile:
    """How much generated history each latent attends to.

    ``window_frames`` counts latents in view of a new latent, itself and the
    sinks included (cosmos-framework's ``kv_cache_inference_size``);
    ``sink_frames`` is how many of the first generated latents stay pinned
    (``attention_sink_size``). Memory is bounded by the window whatever the
    clip length.
    """

    window_frames: int
    sink_frames: int

    def __post_init__(self) -> None:
        if self.window_frames < 1:
            raise ValueError(f"window_frames must be >= 1, got {self.window_frames}")
        if not 0 <= self.sink_frames < self.window_frames:
            raise ValueError(
                f"sink_frames must be in [0, window_frames), got {self.sink_frames} "
                f"with window_frames={self.window_frames}"
            )

    @classmethod
    def parse(cls, text: str) -> HistoryProfile:
        """Read ``W/S``: ``W`` latents in view, the first ``S`` pinned (``30/3``).

        Raises:
            ValueError: ``text`` is not two integers separated by ``/``, or the
                window is impossible.
        """
        window, slash, sinks = text.strip().partition("/")
        try:
            if not slash:
                raise ValueError
            return cls(int(window), int(sinks))
        except ValueError as error:
            raise ValueError(
                f"history {text!r} is not a preset {HISTORY_PRESETS} or W/S "
                f"(latents in view / pinned latents, e.g. 30/3): {error}"
            ) from None

    def __str__(self) -> str:
        return f"{self.window_frames}/{self.sink_frames}"


LONG_ROLLOUT_HISTORY = HistoryProfile(window_frames=30, sink_frames=3)
"""The release's long-rollout profile: 30 latents in view, the first 3 pinned."""


def steady_from(history: HistoryProfile, latents: int) -> int:
    """First latent of a run's steady state: a few latents after the window fills."""
    first = history.window_frames + 3 if history.window_frames < latents else 3
    return max(0, min(first, latents - 1))


HISTORY_PRESETS = ("export", "long-rollout")
"""Names accepted by :meth:`TransferModelConfig.history`."""


def check_t_list(t_list: Sequence[float]) -> tuple[float, ...]:
    """A sampler schedule as a tuple of sigmas: at least one, each in (0, 1], strictly decreasing.

    Raises:
        ValueError: ``t_list`` is empty, holds something other than numbers, a
            sigma outside (0, 1] or a sigma not below the one before it.
    """
    try:
        schedule = tuple(float(t) for t in t_list)
    except (TypeError, ValueError):
        raise ValueError(f"t_list {t_list!r} is not a list of numbers") from None
    if not schedule:
        raise ValueError("t_list is empty")
    if any(not 0.0 < t <= 1.0 for t in schedule):
        raise ValueError(f"t_list {list(schedule)}: every sigma must be in (0, 1]")
    if any(later >= earlier for earlier, later in zip(schedule, schedule[1:])):
        raise ValueError(f"t_list {list(schedule)} is not strictly decreasing")
    return schedule


def skip_schedule(t_list: Sequence[float], steps: int) -> tuple[float, ...]:
    """The first sigma of ``t_list`` and its last ``steps - 1``: ``steps`` sampler steps.

    Fewer steps drop the intermediate sigmas right after the first, so the
    last step still starts from the schedule's last sigma. On the release's
    ``(1, 0.9375, 0.8333, 0.625)``, 3 steps give ``(1, 0.8333, 0.625)`` and
    2 steps ``(1, 0.625)``. Truncating the schedule instead (``1, 0.9375``,
    then straight to the clean latent) leaves a color haze in the output.

    Raises:
        ValueError: ``steps`` is below 1 or above the schedule's length, or
            ``t_list`` is not a valid schedule (:func:`check_t_list`).
    """
    schedule = check_t_list(t_list)
    if not 1 <= steps <= len(schedule):
        raise ValueError(
            f"steps must be in [1, {len(schedule)}] for t_list {list(schedule)}, "
            f"got {steps}"
        )
    return schedule[:1] + schedule[len(schedule) - steps + 1 :]


@dataclass(frozen=True)
class TransferModelConfig:
    """Model settings of one exported Transfer checkpoint."""

    checkpoint_id: str
    """Training run and iteration the export was taken from."""
    base_fps: float
    """Frame rate the temporal positions are normalized to."""
    enable_fps_modulation: bool
    """Scale temporal positions by ``base_fps / fps``."""
    temporal_modality_margin: float
    """Gap between the last text position and the first vision position."""
    reset_spatial_ids: bool
    """Spatial positions restart at zero instead of after the text."""
    temporal_compression_factor: int
    """Pixel frames per latent (after the first)."""
    spatial_compression_factor: int
    """Pixels per latent cell along each side."""
    latent_patch_size: int
    """Latent cells per token along each side."""
    t_list: tuple[float, ...]
    """Sigma schedule of the distilled sampler: the export's, or another one
    from :meth:`with_schedule`."""
    num_train_timesteps: float
    """Scale from sigma to the raw timestep the network embeds."""
    sample_type: str
    """``sde`` (re-noise between steps) or ``ode``."""
    chunk_partition: tuple[int, ...]
    """Latents generated per step; Transfer releases use one throughout."""
    window_frames: int
    """Export default history window, in latents."""
    sink_frames: int
    """Export default pinned latents."""
    text_cache_max_len: int
    """Length, in tokens, the prompt cache is padded to for a static-shape
    (compiled or CUDA-graph) run; cosmos-framework's
    ``ar_static_und_cache_max_len``. Not a prompt limit: the eager path takes
    longer prompts."""
    control_attention_mode: str
    """How controls read the history; must be :data:`CONTROL_ATTENTION_MODE`."""
    hints: tuple[str, ...]
    """Control video kinds the checkpoint was trained on."""
    system_prompt_id: str
    """System prompt the checkpoint was trained with; must be :data:`SYSTEM_PROMPT_ID`."""

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> TransferModelConfig:
        """Build from one export metadata block (the value of :data:`EXPORT_METADATA_KEY`).

        Raises:
            KeyError: A required setting is missing.
            ValueError: The block describes something this rollout does not run:
                more than one latent per step, another control attention mode,
                another system prompt, a control kind it cannot resize, or a
                non-control-video checkpoint.
        """
        sampler = metadata["fixed_step_sampler_config"]
        mrope = metadata.get("mrope", {})
        conditioning = metadata.get("conditioning", {})

        def mrope_value(name: str, flat_name: str) -> Any:
            # checkpoint.json nests these under "mrope"; config.json keeps them flat.
            return mrope[name] if name in mrope else metadata[flat_name]

        partition = tuple(
            int(v)
            for v in metadata.get("chunk_partition", (metadata["chunk_size"],) * 2)
        )
        config = cls(
            checkpoint_id=str(metadata.get("checkpoint_id", "")),
            base_fps=float(mrope_value("base_fps", "base_fps")),
            enable_fps_modulation=bool(
                mrope_value("enable_fps_modulation", "enable_fps_modulation")
            ),
            temporal_modality_margin=float(
                mrope_value("temporal_modality_margin", "temporal_modality_margin")
            ),
            reset_spatial_ids=bool(
                mrope_value("reset_spatial_ids", "unified_3d_mrope_reset_spatial_ids")
            ),
            temporal_compression_factor=int(metadata["temporal_compression_factor"]),
            spatial_compression_factor=int(metadata["vae_spatial_compression_factor"]),
            latent_patch_size=int(metadata["latent_patch_size"]),
            t_list=tuple(float(t) for t in sampler["t_list"]),
            num_train_timesteps=float(sampler["num_train_timesteps"]),
            sample_type=str(sampler["sample_type"]),
            chunk_partition=partition,
            window_frames=int(metadata["window_frames"]),
            sink_frames=int(metadata["sink_frames"]),
            text_cache_max_len=int(metadata["text_cache_max_len"]),
            control_attention_mode=str(
                conditioning.get("transfer_control_attention_mode", "")
            ),
            hints=tuple(str(h) for h in conditioning.get("hints", CONTROL_HINTS)),
            system_prompt_id=str(conditioning.get("system_prompt_id", "")),
        )
        if conditioning.get("mode", "control_video") != "control_video":
            raise ValueError(
                f"not a control-video checkpoint: conditioning mode {conditioning['mode']!r}"
            )
        if config.control_attention_mode != CONTROL_ATTENTION_MODE:
            raise ValueError(
                f"control attention mode {config.control_attention_mode!r} is not "
                f"supported; this rollout implements {CONTROL_ATTENTION_MODE!r}"
            )
        if config.system_prompt_id != SYSTEM_PROMPT_ID:
            raise ValueError(
                f"system prompt {config.system_prompt_id!r} is not supported; "
                f"this package builds {SYSTEM_PROMPT_ID!r}"
            )
        unknown = [h for h in config.hints if h not in CONTROL_HINTS]
        if unknown or not config.hints:
            raise ValueError(
                f"control kinds {list(config.hints)} are not supported; "
                f"this package resizes {list(CONTROL_HINTS)}"
            )
        if any(n != 1 for n in config.chunk_partition):
            raise ValueError(
                f"chunk partition {list(config.chunk_partition)} is not supported; "
                "this rollout generates one latent per step"
            )
        HistoryProfile(config.window_frames, config.sink_frames)  # validates
        return config

    @classmethod
    def from_export(cls, export_dir: str | Path) -> TransferModelConfig:
        """Read the settings of the export in ``export_dir`` (a release snapshot root).

        ``checkpoint.json`` is read first; ``transformer/config.json`` is the
        fallback for exports without it.

        Raises:
            FileNotFoundError: Neither file carries an export metadata block.
        """
        root = Path(export_dir)
        for name in ("checkpoint.json", "transformer/config.json"):
            path = root / name
            if path.is_file():
                data = json.loads(path.read_text())
                if EXPORT_METADATA_KEY in data:
                    return cls.from_metadata(data[EXPORT_METADATA_KEY])
        raise FileNotFoundError(
            f"no {EXPORT_METADATA_KEY!r} block in {root}/checkpoint.json or "
            f"{root}/transformer/config.json"
        )

    def with_schedule(
        self, *, steps: int | None = None, t_list: Sequence[float] | None = None
    ) -> TransferModelConfig:
        """These settings with another sampler schedule (fewer denoising passes per latent).

        Args:
            steps: Sampler steps: the schedule with intermediate sigmas dropped
                (:func:`skip_schedule`).
            t_list: The schedule itself: strictly decreasing sigmas in
                (0, 1] that start at this schedule's first sigma, the noise
                level the starting noise is drawn at.

        Returns:
            ``self`` when both are ``None``, else a copy with the new ``t_list``.

        Raises:
            ValueError: Both are given, or the schedule is invalid.
        """
        if steps is not None and t_list is not None:
            raise ValueError("give steps or t_list, not both")
        if steps is not None:
            return replace(self, t_list=skip_schedule(self.t_list, steps))
        if t_list is None:
            return self
        schedule = check_t_list(t_list)
        if schedule[0] != self.t_list[0]:
            raise ValueError(
                f"t_list {list(schedule)} must start at {self.t_list[0]}, the noise "
                "level the starting noise is drawn at"
            )
        return replace(self, t_list=schedule)

    @property
    def export_history(self) -> HistoryProfile:
        """The window and sinks the export declares (stage-3 training)."""
        return HistoryProfile(self.window_frames, self.sink_frames)

    def history(self, preset: str | HistoryProfile = "long-rollout") -> HistoryProfile:
        """Resolve a history: a preset name (``export``, ``long-rollout``) or ``W/S``.

        A :class:`HistoryProfile` is returned as is.

        Raises:
            ValueError: ``preset`` is neither in :data:`HISTORY_PRESETS` nor
                a valid ``W/S``.
        """
        if isinstance(preset, HistoryProfile):
            return preset
        if preset == "export":
            return self.export_history
        if preset == "long-rollout":
            return LONG_ROLLOUT_HISTORY
        return HistoryProfile.parse(preset)


def fetch_release(
    local_dir: str | Path | None = None, *, revision: str = HF_REVISION
) -> Path:
    """Validate a local export, or download the pinned release when none is given.

    Explicit local paths must contain compatible Transfer metadata, a weight
    index and every indexed shard. They are never populated with release weights.

    Raises:
        FileNotFoundError: The local directory, index or a weight shard is missing.
        ValueError: The directory contains native DCP or incompatible export metadata.
    """
    if local_dir is not None:
        root = Path(local_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Transfer export directory does not exist: {root}")
        if any((root / name).is_file() for name in (".metadata", "model/.metadata")):
            raise ValueError(
                f"{root} is a native DCP checkpoint. Export the selected student EMA "
                "to safetensors with compatible Transfer metadata before loading it."
            )
        try:
            TransferModelConfig.from_export(root)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{root} is not a compatible Transfer export: {error}. "
                "Provide exported student weights with their matching Transfer metadata."
            ) from error
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str) or not isinstance(shard, str) or not shard
                for key, shard in weight_map.items()
            )
        ):
            raise ValueError(f"{index_path} must contain a nonempty weight_map")
        for shard in set(weight_map.values()):
            assert isinstance(shard, str)
            if not (root / shard).is_file():
                raise FileNotFoundError(
                    f"Transfer export weight shard is missing: {root / shard}"
                )
        return root
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_REPO_ID,
            revision=revision,
            allow_patterns=list(RELEASE_FILES),
        )
    )
