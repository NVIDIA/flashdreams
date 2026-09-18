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

"""Layer-wise CPU parameter offload for sequential inference modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

_PARAMETER_ALIGNMENT_BYTES = 32


@dataclass(frozen=True)
class _ParameterMetadata:
    owner: nn.Module
    name: str
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    offset: int

    @property
    def parameter(self) -> nn.Parameter:
        parameter = self.owner._parameters[self.name]
        assert parameter is not None
        return parameter


@dataclass
class _LayerState:
    parameters: list[_ParameterMetadata]
    host_buffers: dict[torch.dtype, torch.Tensor]
    materialized_device: torch.device | None = None
    slot_index: int | None = None
    ready_event: torch.cuda.Event | None = None


@dataclass
class _DeviceSlot:
    buffers: dict[torch.dtype, torch.Tensor]
    owner_layer: int | None = None
    reusable_event: torch.cuda.Event | None = None


@dataclass
class _DevicePool:
    copy_stream: torch.cuda.Stream
    slots: tuple[_DeviceSlot, _DeviceSlot]
    next_slot: int = 0


class LayerwiseOffloader:
    """Keep sequential layer parameters on CPU and materialize them just in time.

    The target layers must still be on CPU when this object is constructed. Their
    parameters are packed into pinned host buffers and rebound to empty placeholders,
    so a later parent ``module.to("cuda")`` only moves the placeholders. During an
    execution context, the current layer is restored and the next layer is prefetched
    on a dedicated CUDA stream. Parameters are immutable CPU masters: finishing a
    layer drops its GPU views instead of copying them back to the host.

    This helper is intended for single-threaded inference. It does not support
    training, parameter mutation, DTensor parameters, or dtype conversion after
    construction. CUDA work that consumes materialized parameters must be enqueued
    on the current stream inside the context; callers are responsible for joining
    any side streams before leaving it. Buffers intentionally remain resident on
    the execution device.
    """

    def __init__(
        self,
        layers: Iterable[nn.Module],
        *,
        pin_memory: bool = True,
    ) -> None:
        resolved_layers = tuple(layers)
        if not resolved_layers:
            raise ValueError("layer-wise offload requires at least one layer")

        self._layers = resolved_layers
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._placeholders: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._device_pools: dict[torch.device, _DevicePool] = {}
        self._states: list[_LayerState] = []

        seen_parameters: dict[int, tuple[int, str]] = {}
        seen_storages: dict[int, tuple[int, str]] = {}
        layer_parameters: list[list[tuple[str, nn.Parameter]]] = []
        for layer_index, layer in enumerate(self._layers):
            named_parameters = list(
                layer.named_parameters(recurse=True, remove_duplicate=False)
            )
            layer_parameters.append(named_parameters)
            for name, parameter in named_parameters:
                if hasattr(parameter, "device_mesh"):
                    raise ValueError(
                        "layer-wise offload does not support DTensor parameters"
                    )
                previous = seen_parameters.get(id(parameter))
                if previous is not None:
                    previous_layer, previous_name = previous
                    raise ValueError(
                        "layer-wise offload does not support tied parameters: "
                        f"layer {previous_layer} {previous_name!r} and layer "
                        f"{layer_index} {name!r} are the same Parameter object"
                    )
                seen_parameters[id(parameter)] = (layer_index, name)
                storage_pointer = parameter.untyped_storage().data_ptr()
                previous_storage = seen_storages.get(storage_pointer)
                if previous_storage is not None:
                    previous_layer, previous_name = previous_storage
                    raise ValueError(
                        "layer-wise offload does not support parameter storage "
                        f"aliases: layer {previous_layer} {previous_name!r} and "
                        f"layer {layer_index} {name!r} share one allocation"
                    )
                seen_storages[storage_pointer] = (layer_index, name)
        try:
            for layer, named_parameters in zip(
                self._layers, layer_parameters, strict=True
            ):
                named_parameters = self._normalize_inference_parameters(
                    layer, named_parameters
                )
                state = self._pack_layer(layer, named_parameters)
                self._states.append(state)
                self._clear_parameter_views(state)
        except BaseException:
            for state in self._states:
                self._restore_parameter_copies(state)
            raise

        for layer in self._layers:
            layer.register_state_dict_pre_hook(self._reject_serialization)
            layer.register_load_state_dict_pre_hook(self._reject_serialization)

    @property
    def num_layers(self) -> int:
        """Number of managed sequential layers."""
        return len(self._states)

    @property
    def parameter_bytes(self) -> int:
        """Logical bytes of parameters managed by this offloader."""
        return sum(
            int(torch.Size(metadata.shape).numel())
            * torch.empty((), dtype=metadata.dtype).element_size()
            for state in self._states
            for metadata in state.parameters
        )

    @property
    def host_buffer_bytes(self) -> int:
        """Physical bytes allocated for aligned host staging buffers."""
        return sum(
            buffer.numel() * buffer.element_size()
            for state in self._states
            for buffer in state.host_buffers.values()
        )

    @property
    def device_slot_bytes(self) -> int:
        """GPU bytes reserved by the fixed two-slot prefetch window."""
        maxima: dict[torch.dtype, int] = {}
        for state in self._states:
            for dtype, buffer in state.host_buffers.items():
                maxima[dtype] = max(maxima.get(dtype, 0), buffer.numel())
        return 2 * sum(
            numel * torch.empty((), dtype=dtype).element_size()
            for dtype, numel in maxima.items()
        )

    @property
    def resident_layer_indices(self) -> tuple[int, ...]:
        """Layer indices whose real parameters are currently materialized."""
        return tuple(
            index
            for index, state in enumerate(self._states)
            if state.materialized_device is not None
        )

    @contextmanager
    def materialize(self, layer_index: int) -> Iterator[None]:
        """Materialize one layer, prefetch its successor, then evict it."""
        self._validate_layer_index(layer_index)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "layer-wise offload only supports inference; run under "
                "torch.no_grad() or torch.inference_mode()"
            )
        try:
            if self._states[layer_index].materialized_device is None:
                # A non-sequential caller may leave an unrelated speculative layer.
                # Clear it so the fixed two-slot window always has room for current
                # plus next.
                self.release_all()
            device = self._materialize(layer_index)
            self._wait_until_ready(layer_index, device)
            self._materialize((layer_index + 1) % self.num_layers, device=device)
            yield
        except BaseException:
            self.release_all()
            raise
        else:
            self._release(layer_index)

    def release_all(self) -> None:
        """Evict every materialized layer, including speculative prefetches."""
        for layer_index in range(self.num_layers):
            self._release(layer_index)

    @staticmethod
    def _reject_serialization(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "a layer-wise offloaded inference module cannot be serialized or "
            "reloaded; use the original checkpoint"
        )

    @staticmethod
    @torch.inference_mode(False)
    def _normalize_inference_parameters(
        layer: nn.Module,
        named_parameters: list[tuple[str, nn.Parameter]],
    ) -> list[tuple[str, nn.Parameter]]:
        normalized: list[tuple[str, nn.Parameter]] = []
        with torch.no_grad():
            for qualified_name, parameter in named_parameters:
                if parameter.is_inference():
                    owner, local_name = _resolve_parameter_owner(layer, qualified_name)
                    parameter = nn.Parameter(
                        parameter.detach().clone(),
                        requires_grad=parameter.requires_grad,
                    )
                    owner._parameters[local_name] = parameter
                normalized.append((qualified_name, parameter))
        return normalized

    @torch.inference_mode(False)
    def _pack_layer(
        self,
        layer: nn.Module,
        named_parameters: list[tuple[str, nn.Parameter]],
    ) -> _LayerState:
        if not named_parameters:
            raise ValueError(f"managed layer {type(layer).__name__} has no parameters")

        grouped: dict[torch.dtype, list[tuple[str, nn.Parameter]]] = {}
        for name, parameter in named_parameters:
            if parameter.device.type != "cpu":
                raise ValueError(
                    "layer-wise offload must be enabled before moving the model "
                    f"to an accelerator; {name!r} is on {parameter.device}"
                )
            if parameter.layout is not torch.strided:
                raise ValueError(
                    f"layer-wise offload only supports strided parameters; {name!r} "
                    f"uses {parameter.layout}"
                )
            if hasattr(parameter, "device_mesh"):
                raise ValueError(
                    "layer-wise offload does not support DTensor parameters"
                )
            if parameter.numel() == 0:
                raise ValueError(
                    f"layer-wise offload does not support empty parameter {name!r}"
                )
            if torch._debug_has_internal_overlap(parameter) != 0:
                raise ValueError(
                    "layer-wise offload does not support overlapping parameter "
                    f"layout for {name!r}"
                )
            grouped.setdefault(parameter.dtype, []).append((name, parameter))

        host_buffers: dict[torch.dtype, torch.Tensor] = {}
        metadata: list[_ParameterMetadata] = []

        for dtype, entries in grouped.items():
            alignment_numel = max(
                1,
                _PARAMETER_ALIGNMENT_BYTES
                // torch.empty((), dtype=dtype).element_size(),
            )
            offset = 0
            layouts: list[tuple[str, nn.Parameter, nn.Module, str, int]] = []
            for qualified_name, parameter in entries:
                offset = _align(offset, alignment_numel)
                storage_numel = 1 + sum(
                    (size - 1) * stride
                    for size, stride in zip(parameter.shape, parameter.stride())
                )
                if storage_numel < parameter.numel() or any(
                    stride < 0 for stride in parameter.stride()
                ):
                    raise ValueError(
                        "layer-wise offload does not support overlapping or negative "
                        f"strides for parameter {qualified_name!r}"
                    )
                owner, local_name = _resolve_parameter_owner(layer, qualified_name)
                layouts.append(
                    (
                        qualified_name,
                        parameter,
                        owner,
                        local_name,
                        offset,
                    )
                )
                offset += storage_numel

            host_buffer = torch.empty(
                offset,
                dtype=dtype,
                device="cpu",
                pin_memory=self._pin_memory,
            )
            for (
                qualified_name,
                parameter,
                owner,
                local_name,
                parameter_offset,
            ) in layouts:
                host_view = torch.as_strided(
                    host_buffer,
                    size=parameter.shape,
                    stride=parameter.stride(),
                    storage_offset=parameter_offset,
                )
                try:
                    host_view.copy_(parameter.detach())
                except RuntimeError as error:
                    raise ValueError(
                        "layer-wise offload cannot preserve the stride layout of "
                        f"parameter {qualified_name!r}"
                    ) from error
                metadata.append(
                    _ParameterMetadata(
                        owner=owner,
                        name=local_name,
                        shape=parameter.shape,
                        stride=parameter.stride(),
                        dtype=dtype,
                        offset=parameter_offset,
                    )
                )
            host_buffers[dtype] = host_buffer

        return _LayerState(parameters=metadata, host_buffers=host_buffers)

    def _clear_parameter_views(self, state: _LayerState) -> None:
        with torch.inference_mode(False), torch.no_grad():
            for metadata in state.parameters:
                parameter = metadata.parameter
                parameter.data = self._placeholder(parameter.device, parameter.dtype)

    @staticmethod
    @torch.inference_mode(False)
    def _restore_parameter_copies(state: _LayerState) -> None:
        with torch.no_grad():
            for metadata in state.parameters:
                packed_view = torch.as_strided(
                    state.host_buffers[metadata.dtype],
                    size=metadata.shape,
                    stride=metadata.stride,
                    storage_offset=metadata.offset,
                )
                restored = torch.empty_strided(
                    metadata.shape,
                    metadata.stride,
                    dtype=metadata.dtype,
                    device="cpu",
                )
                restored.copy_(packed_view)
                metadata.parameter.data = restored

    @torch.compiler.disable
    def _materialize(
        self,
        layer_index: int,
        *,
        device: torch.device | None = None,
    ) -> torch.device:
        state = self._states[layer_index]
        if state.materialized_device is not None:
            if device is not None and state.materialized_device != device:
                raise ValueError(
                    "an offloaded layer cannot be materialized on multiple devices"
                )
            return state.materialized_device

        layer_device = self._layer_device(state) if device is None else device
        self._validate_parameter_targets(state, layer_device)
        if layer_device.type == "cpu":
            self._bind_parameter_views(state, state.host_buffers)
            state.materialized_device = layer_device
            return layer_device
        if layer_device.type != "cuda":
            raise ValueError(
                f"layer-wise offload only supports CPU and CUDA execution, got {layer_device}"
            )

        current_stream = torch.cuda.current_stream(layer_device)
        pool = self._device_pool(layer_device)
        slot_index = self._reserve_slot(pool, layer_index)
        slot = pool.slots[slot_index]
        pool.copy_stream.wait_stream(current_stream)
        if slot.reusable_event is not None:
            pool.copy_stream.wait_event(slot.reusable_event)

        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.cuda.stream(pool.copy_stream),
        ):
            for dtype, host_buffer in state.host_buffers.items():
                slot.buffers[dtype][: host_buffer.numel()].copy_(
                    host_buffer,
                    non_blocking=self._pin_memory,
                )
            self._bind_parameter_views(state, slot.buffers)

        ready_event = torch.cuda.Event()
        ready_event.record(pool.copy_stream)
        state.materialized_device = layer_device
        state.slot_index = slot_index
        state.ready_event = ready_event
        return layer_device

    @torch.compiler.disable
    def _wait_until_ready(self, layer_index: int, device: torch.device) -> None:
        if device.type != "cuda":
            return
        state = self._states[layer_index]
        assert state.slot_index is not None
        pool = self._device_pools[device]
        slot = pool.slots[state.slot_index]
        current_stream = torch.cuda.current_stream(device)
        if state.ready_event is not None:
            current_stream.wait_event(state.ready_event)
        for buffer in slot.buffers.values():
            buffer.record_stream(current_stream)

    @torch.compiler.disable
    def _release(self, layer_index: int) -> None:
        state = self._states[layer_index]
        if state.materialized_device is None:
            return

        device = state.materialized_device
        if device.type == "cuda":
            assert state.slot_index is not None
            pool = self._device_pools[device]
            slot = pool.slots[state.slot_index]
            current_stream = torch.cuda.current_stream(device)
            if state.ready_event is not None:
                current_stream.wait_event(state.ready_event)
            for buffer in slot.buffers.values():
                buffer.record_stream(current_stream)
            reusable_event = torch.cuda.Event()
            reusable_event.record(current_stream)
            slot.reusable_event = reusable_event
            slot.owner_layer = None

        with torch.inference_mode(False), torch.no_grad():
            for parameter_metadata in state.parameters:
                parameter = parameter_metadata.parameter
                parameter.data = self._placeholder(
                    parameter.device, parameter_metadata.dtype
                )
        state.materialized_device = None
        state.slot_index = None
        state.ready_event = None

    @torch.inference_mode(False)
    def _device_pool(self, device: torch.device) -> _DevicePool:
        pool = self._device_pools.get(device)
        if pool is not None:
            return pool

        maxima: dict[torch.dtype, int] = {}
        for state in self._states:
            for dtype, host_buffer in state.host_buffers.items():
                maxima[dtype] = max(maxima.get(dtype, 0), host_buffer.numel())
        slots = tuple(
            _DeviceSlot(
                buffers={
                    dtype: torch.empty(numel, dtype=dtype, device=device)
                    for dtype, numel in maxima.items()
                }
            )
            for _ in range(2)
        )
        pool = _DevicePool(
            copy_stream=torch.cuda.Stream(device=device),
            slots=(slots[0], slots[1]),
        )
        self._device_pools[device] = pool
        return pool

    @staticmethod
    def _reserve_slot(pool: _DevicePool, layer_index: int) -> int:
        for offset in range(2):
            slot_index = (pool.next_slot + offset) % 2
            slot = pool.slots[slot_index]
            if slot.owner_layer is None:
                slot.owner_layer = layer_index
                pool.next_slot = (slot_index + 1) % 2
                return slot_index
        raise RuntimeError(
            "layer-wise offload exhausted its fixed current/next device slots"
        )

    def _bind_parameter_views(
        self,
        state: _LayerState,
        buffers: dict[torch.dtype, torch.Tensor],
    ) -> None:
        with torch.inference_mode(False), torch.no_grad():
            for metadata in state.parameters:
                buffer = buffers[metadata.dtype]
                view = torch.as_strided(
                    buffer,
                    size=metadata.shape,
                    stride=metadata.stride,
                    storage_offset=metadata.offset,
                )
                metadata.parameter.data = view

    def _validate_parameter_targets(
        self,
        state: _LayerState,
        device: torch.device,
    ) -> None:
        for metadata in state.parameters:
            parameter = metadata.parameter
            if parameter.device != device:
                raise ValueError(
                    "all offloaded parameters must be moved to one execution device; "
                    f"expected {device}, got {parameter.device}"
                )
            if parameter.dtype != metadata.dtype:
                raise ValueError(
                    "dtype conversion after enabling layer-wise offload is not "
                    f"supported; expected {metadata.dtype}, got {parameter.dtype}"
                )

    @staticmethod
    def _layer_device(state: _LayerState) -> torch.device:
        return state.parameters[0].parameter.device

    @torch.inference_mode(False)
    def _placeholder(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device, dtype)
        placeholder = self._placeholders.get(key)
        if placeholder is None:
            placeholder = torch.empty(0, device=device, dtype=dtype)
            self._placeholders[key] = placeholder
        return placeholder

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer index {layer_index} is outside [0, {self.num_layers})"
            )


def _resolve_parameter_owner(
    layer: nn.Module,
    qualified_name: str,
) -> tuple[nn.Module, str]:
    parent_name, separator, local_name = qualified_name.rpartition(".")
    owner = layer.get_submodule(parent_name) if separator else layer
    return owner, local_name


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
