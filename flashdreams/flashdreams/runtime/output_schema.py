# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared model-output and output-target compatibility contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

RGB_VIDEO = "video/rgb"
"""Decoded RGB video frames."""


@dataclass(frozen=True, kw_only=True, slots=True)
class InferenceOutputSchema:
    """One semantic result type produced by an inference session."""

    modality: str
    """Semantic output modality, such as ``video/rgb``."""

    python_type: type[Any]
    """Runtime type stored in :attr:`StepResult.output`."""

    layouts: frozenset[str] | None = None
    """Tensor layouts the model may emit; ``None`` means not applicable."""

    def __post_init__(self) -> None:
        if not self.modality.strip():
            raise ValueError("InferenceOutputSchema.modality must be non-empty.")
        if not isinstance(self.python_type, type):
            raise TypeError("InferenceOutputSchema.python_type must be a type.")
        if self.layouts is not None:
            normalized = frozenset(layout.strip() for layout in self.layouts)
            if not normalized or "" in normalized:
                raise ValueError(
                    "InferenceOutputSchema.layouts must be ``None`` or contain "
                    "non-empty names."
                )
            object.__setattr__(self, "layouts", normalized)


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputTargetRequirement:
    """Result types one output target can consume."""

    modalities: frozenset[str] | None
    """Accepted modalities; ``None`` accepts every modality."""

    python_type: type[Any] = object
    """Required runtime base type for :attr:`StepResult.output`."""

    layouts: frozenset[str] | None = None
    """Accepted tensor layouts; ``None`` accepts every layout."""

    def __post_init__(self) -> None:
        if self.modalities is not None:
            normalized = frozenset(modality.strip() for modality in self.modalities)
            if not normalized or "" in normalized:
                raise ValueError(
                    "OutputTargetRequirement.modalities must be ``None`` or "
                    "contain non-empty names."
                )
            object.__setattr__(self, "modalities", normalized)
        if not isinstance(self.python_type, type):
            raise TypeError("OutputTargetRequirement.python_type must be a type.")
        if self.layouts is not None:
            normalized_layouts = frozenset(layout.strip() for layout in self.layouts)
            if not normalized_layouts or "" in normalized_layouts:
                raise ValueError(
                    "OutputTargetRequirement.layouts must be ``None`` or contain "
                    "non-empty names."
                )
            object.__setattr__(self, "layouts", normalized_layouts)


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputCompatibility:
    """Compatibility result between one model output and one target."""

    compatible: bool
    """Whether the target can consume the declared model result."""

    reasons: tuple[str, ...] = ()
    """Human-readable incompatibilities; empty when compatible."""


@runtime_checkable
class DeclaresInferenceOutput(Protocol):
    """Model adapter exposing the result type its sessions produce."""

    @property
    def inference_output_schema(self) -> InferenceOutputSchema:
        """Return the session result declaration."""
        ...


@runtime_checkable
class DeclaresOutputRequirement(Protocol):
    """Output target exposing the result types it accepts."""

    @property
    def output_requirement(self) -> OutputTargetRequirement:
        """Return the target's accepted result declaration."""
        ...


def check_output_compatibility(
    *,
    produced: InferenceOutputSchema,
    required: OutputTargetRequirement,
) -> OutputCompatibility:
    """Check whether one output target accepts a model's declared result."""
    reasons: list[str] = []
    if required.modalities is not None and produced.modality not in required.modalities:
        reasons.append(
            f"modality {produced.modality!r} is not accepted; expected one of "
            f"{sorted(required.modalities)!r}"
        )
    if not issubclass(produced.python_type, required.python_type):
        reasons.append(
            f"type {produced.python_type.__name__} is not a subclass of "
            f"{required.python_type.__name__}"
        )
    if required.layouts is not None:
        if produced.layouts is None:
            reasons.append(
                "model output does not declare a tensor layout; expected one of "
                f"{sorted(required.layouts)!r}"
            )
        elif produced.layouts.isdisjoint(required.layouts):
            reasons.append(
                f"layouts {sorted(produced.layouts)!r} do not intersect accepted "
                f"layouts {sorted(required.layouts)!r}"
            )
    return OutputCompatibility(compatible=not reasons, reasons=tuple(reasons))


def require_output_compatibility(
    *,
    produced: InferenceOutputSchema,
    required: OutputTargetRequirement,
) -> None:
    """Raise when an output target cannot consume a model's declared result.

    Raises:
        ValueError: The modality or runtime type is incompatible.
    """
    compatibility = check_output_compatibility(produced=produced, required=required)
    if not compatibility.compatible:
        raise ValueError(
            "Model output is incompatible with the selected output target: "
            + "; ".join(compatibility.reasons)
        )


__all__ = [
    "RGB_VIDEO",
    "DeclaresInferenceOutput",
    "DeclaresOutputRequirement",
    "InferenceOutputSchema",
    "OutputCompatibility",
    "OutputTargetRequirement",
    "check_output_compatibility",
    "require_output_compatibility",
]
