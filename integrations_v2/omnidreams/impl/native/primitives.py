# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared tensor primitives for native pipeline components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


class NativePrepError(ValueError):
    """Raised when a tensor cannot be prepared for native execution."""


@dataclass(frozen=True)
class NativeTensorLayout:
    """Named tensor axes at the Python/native boundary."""

    axes: tuple[str, ...]

    @classmethod
    def parse(cls, axes: str | Sequence[str]) -> "NativeTensorLayout":
        if isinstance(axes, str):
            parsed = tuple(axis for axis in axes.replace(",", " ").split() if axis)
        else:
            parsed = tuple(axes)
        if not parsed:
            raise ValueError("Native tensor layout must contain at least one axis")
        if any(not axis for axis in parsed):
            raise ValueError(f"Native tensor layout contains an empty axis: {parsed!r}")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"Native tensor layout axes must be unique: {parsed!r}")
        return cls(parsed)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def axis_index(self, axis: str) -> int:
        try:
            return self.axes.index(axis)
        except ValueError as exc:
            raise KeyError(f"Axis {axis!r} is not present in layout {self}") from exc

    def __str__(self) -> str:
        return " ".join(self.axes)


@dataclass(frozen=True)
class NativeTensorDescriptor:
    """Observed tensor metadata used for diagnostics and planning."""

    name: str
    layout: NativeTensorLayout
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    is_contiguous: bool


@dataclass(frozen=True)
class NativeTensorSpec:
    """Constraints for a tensor consumed by a native component."""

    name: str
    layout: NativeTensorLayout | str | Sequence[str]
    shape: tuple[int | None, ...] = ()
    dtypes: tuple[torch.dtype, ...] = ()
    device_type: str | None = None
    require_contiguous: bool = True
    axis_divisibility: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        layout = self.resolved_layout
        object.__setattr__(self, "layout", layout)
        if self.shape and len(self.shape) != layout.ndim:
            raise ValueError(
                f"{self.name}: shape constraint rank {len(self.shape)} does not "
                f"match layout rank {layout.ndim}"
            )
        for axis, factor in self.axis_divisibility:
            layout.axis_index(axis)
            if factor < 1:
                raise ValueError(
                    f"{self.name}: divisibility factor for axis {axis!r} must be positive"
                )

    @property
    def resolved_layout(self) -> NativeTensorLayout:
        if isinstance(self.layout, NativeTensorLayout):
            return self.layout
        return NativeTensorLayout.parse(self.layout)


@dataclass(frozen=True)
class NativePreparedTensor:
    """Tensor prepared for a native call plus preparation metadata."""

    tensor: Tensor
    descriptor: NativeTensorDescriptor
    copied: bool


def _describe_tensor(
    tensor: Tensor,
    *,
    name: str,
    layout: NativeTensorLayout | str | Sequence[str],
) -> NativeTensorDescriptor:
    """Return observed metadata for a tensor at a native boundary."""

    resolved_layout = (
        layout
        if isinstance(layout, NativeTensorLayout)
        else NativeTensorLayout.parse(layout)
    )
    return NativeTensorDescriptor(
        name=name,
        layout=resolved_layout,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
        is_contiguous=tensor.is_contiguous(),
    )


def prepare_tensor_for_native(
    tensor: Tensor,
    spec: NativeTensorSpec,
) -> NativePreparedTensor:
    """Validate and make a tensor contiguous when the native spec requires it."""

    descriptor = _describe_tensor(tensor, name=spec.name, layout=spec.layout)
    _validate_descriptor(descriptor, spec, check_contiguous=False)
    prepared = tensor
    copied = False
    if spec.require_contiguous and not tensor.is_contiguous():
        prepared = tensor.contiguous()
        copied = True
        descriptor = _describe_tensor(prepared, name=spec.name, layout=spec.layout)
    _validate_descriptor(descriptor, spec)
    return NativePreparedTensor(
        tensor=prepared,
        descriptor=descriptor,
        copied=copied,
    )


def _validate_descriptor(
    descriptor: NativeTensorDescriptor,
    spec: NativeTensorSpec,
    *,
    check_contiguous: bool = True,
) -> None:
    layout = spec.resolved_layout
    if len(descriptor.shape) != layout.ndim:
        raise NativePrepError(
            f"{spec.name}: expected layout {layout} with rank {layout.ndim}, "
            f"got shape {descriptor.shape}"
        )
    if spec.shape:
        for axis, expected, actual in zip(layout.axes, spec.shape, descriptor.shape):
            if expected is not None and expected != actual:
                raise NativePrepError(
                    f"{spec.name}: expected axis {axis} to be {expected}, got {actual}"
                )
    if spec.dtypes and descriptor.dtype not in spec.dtypes:
        allowed = ", ".join(str(dtype) for dtype in spec.dtypes)
        raise NativePrepError(
            f"{spec.name}: expected dtype in ({allowed}), got {descriptor.dtype}"
        )
    if spec.device_type is not None and descriptor.device.type != spec.device_type:
        raise NativePrepError(
            f"{spec.name}: expected device type {spec.device_type!r}, "
            f"got {descriptor.device.type!r}"
        )
    if check_contiguous and spec.require_contiguous and not descriptor.is_contiguous:
        raise NativePrepError(f"{spec.name}: expected contiguous tensor")
    for axis, factor in spec.axis_divisibility:
        size = descriptor.shape[layout.axis_index(axis)]
        if size % factor != 0:
            raise NativePrepError(
                f"{spec.name}: axis {axis} size {size} must be divisible by {factor}"
            )
