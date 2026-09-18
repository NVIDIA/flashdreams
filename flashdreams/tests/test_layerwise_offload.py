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

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from flashdreams.infra.acceleration.layerwise_offload import LayerwiseOffloader


@pytest.mark.ci_cpu
@torch.no_grad()
def test_layerwise_offloader_matches_resident_sequential_model() -> None:
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
    reference = copy.deepcopy(layers)
    parameter_ids = [
        id(parameter) for layer in layers for parameter in layer.parameters()
    ]
    expected_bytes = sum(
        parameter.numel() * parameter.element_size()
        for layer in layers
        for parameter in layer.parameters()
    )

    offloader = LayerwiseOffloader(layers, pin_memory=False)

    assert offloader.parameter_bytes == expected_bytes
    assert offloader.host_buffer_bytes >= expected_bytes
    assert offloader.resident_layer_indices == ()
    assert all(
        parameter.numel() == 0 for layer in layers for parameter in layer.parameters()
    )

    for _ in range(2):
        value = torch.randn(4, 8)
        expected = value
        actual = value
        for layer in reference:
            expected = layer(expected)
        for layer_index, layer in enumerate(layers):
            with offloader.materialize(layer_index):
                actual = layer(actual)
        torch.testing.assert_close(actual, expected)

    assert parameter_ids == [
        id(parameter) for layer in layers for parameter in layer.parameters()
    ]
    # The final layer prefetches layer zero for the next sequential invocation.
    assert offloader.resident_layer_indices == (0,)


@pytest.mark.ci_cpu
@torch.no_grad()
def test_layerwise_offloader_preserves_strides_and_keeps_buffers_resident() -> None:
    layers = [_StridedLayer(), _StridedLayer()]
    original_weights = [layer.weight.detach().clone() for layer in layers]
    original_strides = [layer.weight.stride() for layer in layers]
    original_buffers = [layer.scale for layer in layers]

    offloader = LayerwiseOffloader(layers, pin_memory=False)

    assert [layer.scale for layer in layers] == original_buffers
    with offloader.materialize(1):
        assert layers[1].weight.stride() == original_strides[1]
        assert layers[1].weight.data_ptr() % 32 == 0
        torch.testing.assert_close(layers[1].weight, original_weights[1])
    assert layers[1].weight.numel() == 0


@pytest.mark.ci_cpu
@torch.no_grad()
def test_layerwise_offloader_supports_out_of_order_access() -> None:
    layers = [nn.Linear(2, 2, bias=False) for _ in range(3)]
    weights = [layer.weight.detach().clone() for layer in layers]
    offloader = LayerwiseOffloader(layers, pin_memory=False)

    for layer_index in (2, 0, 1):
        with offloader.materialize(layer_index):
            torch.testing.assert_close(layers[layer_index].weight, weights[layer_index])


@pytest.mark.ci_cpu
@torch.no_grad()
def test_layerwise_offloader_releases_prefetches_after_error() -> None:
    layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    offloader = LayerwiseOffloader(layers, pin_memory=False)

    with pytest.raises(RuntimeError, match="expected failure"):
        with offloader.materialize(1):
            raise RuntimeError("expected failure")

    assert offloader.resident_layer_indices == ()
    assert all(
        parameter.numel() == 0 for layer in layers for parameter in layer.parameters()
    )


@pytest.mark.ci_cpu
def test_layerwise_offloader_rejects_tied_parameters() -> None:
    shared = nn.Parameter(torch.ones(2, 2))
    first = nn.Linear(2, 2, bias=False)
    second = nn.Linear(2, 2, bias=False)
    first.weight = shared
    second.weight = shared

    with pytest.raises(ValueError, match="tied parameters"):
        LayerwiseOffloader(nn.ModuleList([first, second]), pin_memory=False)


@pytest.mark.ci_cpu
def test_layerwise_offloader_restores_layers_after_constructor_failure() -> None:
    valid_layer = nn.Linear(2, 2)
    value = torch.randn(2, 2)
    expected = valid_layer(value)
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in valid_layer.state_dict().items()
    }

    with pytest.raises(ValueError, match="has no parameters"):
        LayerwiseOffloader([valid_layer, nn.Identity()], pin_memory=False)

    restored_state = valid_layer.state_dict()
    assert restored_state.keys() == original_state.keys()
    for name, expected_parameter in original_state.items():
        torch.testing.assert_close(restored_state[name], expected_parameter)
    torch.testing.assert_close(valid_layer(value), expected)

    retry = LayerwiseOffloader(
        [valid_layer, nn.Linear(2, 2)],
        pin_memory=False,
    )
    assert retry.num_layers == 2


@pytest.mark.ci_cpu
def test_layerwise_offloader_rejects_parameter_storage_aliases() -> None:
    storage = torch.arange(8.0)
    first = nn.Linear(2, 2, bias=False)
    second = nn.Linear(2, 2, bias=False)
    first.weight = nn.Parameter(storage[:4].view(2, 2))
    second.weight = nn.Parameter(storage[4:].view(2, 2))

    with pytest.raises(ValueError, match="storage aliases"):
        LayerwiseOffloader(nn.ModuleList([first, second]), pin_memory=False)


@pytest.mark.ci_cpu
def test_layerwise_offloader_rejects_late_dtype_conversion() -> None:
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    offloader = LayerwiseOffloader(layers, pin_memory=False)
    layers.double()

    with torch.inference_mode(), pytest.raises(ValueError, match="dtype conversion"):
        with offloader.materialize(0):
            pass


@pytest.mark.ci_cpu
def test_layerwise_offloader_rejects_grad_enabled_execution() -> None:
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    offloader = LayerwiseOffloader(layers, pin_memory=False)

    with pytest.raises(RuntimeError, match="only supports inference"):
        with offloader.materialize(0):
            pass


@pytest.mark.ci_cpu
def test_layerwise_offloader_rejects_state_dict_operations() -> None:
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    LayerwiseOffloader(layers, pin_memory=False)

    with pytest.raises(RuntimeError, match="cannot be serialized"):
        layers.state_dict()
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        layers.load_state_dict({})


@pytest.mark.ci_cpu
@torch.inference_mode()
def test_layerwise_offloader_handles_construction_inside_inference_mode() -> None:
    layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    value = torch.randn(2, 4)
    expected = layers[1](layers[0](value))

    offloader = LayerwiseOffloader(layers, pin_memory=False)
    actual = value
    for layer_index, layer in enumerate(layers):
        with offloader.materialize(layer_index):
            actual = layer(actual)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.ci_gpu
@torch.inference_mode()
def test_layerwise_offloader_cuda_prefetch_deep_queue_parity() -> None:
    """Exercise ready/reuse events without synchronizing between layer calls."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(6)])
    reference = copy.deepcopy(layers).to(device)
    offloader = LayerwiseOffloader(layers)
    layers.to(device)
    value = torch.randn(8, 128, device=device)

    expected = value
    for layer in reference:
        expected = layer(expected)

    actual_outputs: list[torch.Tensor] = []
    for _ in range(100):
        actual = value
        for layer_index, layer in enumerate(layers):
            with offloader.materialize(layer_index):
                actual = layer(actual)
        actual_outputs.append(actual)

    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        torch.stack(actual_outputs),
        expected.unsqueeze(0).expand(len(actual_outputs), *expected.shape),
    )
    assert offloader.resident_layer_indices == (0,)


class _StridedLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12.0).reshape(3, 4).t())
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight
