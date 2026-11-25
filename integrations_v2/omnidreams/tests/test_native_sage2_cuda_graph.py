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

"""Probe changed-input CUDA Graph replay with real OmniDreams Native Sage2.

Run this manual regression probe explicitly because building the native
extension is expensive::

    OMNIDREAMS_RUN_NATIVE_SAGE2_CUDAGRAPH_PROBE=1 \
    uv run --group test pytest -p no:manual_marker -m manual \
        integrations_v2/omnidreams/tests/test_native_sage2_cuda_graph.py -v

The expected result while Native Sage2 remains graph-incompatible is one
strict xfail. ``OMNIDREAMS_SINGLEVIEW_NATIVE_TEST_EXTENSION_PATH`` may point at
a previously built extension for a faster local run.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omnidreams.impl.native import omnidreams_singleview as native
from torch import Tensor

from flashdreams.infra.cuda_graph import CUDAGraphWrapper

pytestmark = pytest.mark.manual

_RUN_PROBE_ENV = "OMNIDREAMS_RUN_NATIVE_SAGE2_CUDAGRAPH_PROBE"
_PREBUILT_EXTENSION_ENV = "OMNIDREAMS_SINGLEVIEW_NATIVE_TEST_EXTENSION_PATH"
_CHUNK_TOKENS = 128
_HEADS = 4
_HEAD_DIM = 128


def _load_prebuilt_extension(path: Path) -> ModuleType:
    """Load an explicitly selected PyTorch extension by its compiled name."""
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native extension from {path}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


@pytest.fixture(scope="module")
def native_sage2_extension() -> ModuleType:
    """Load a real Sage2-enabled extension or skip unsupported environments."""
    if os.environ.get(_RUN_PROBE_ENV) != "1":
        pytest.skip(f"Set {_RUN_PROBE_ENV}=1 to run the native CUDA Graph probe.")
    if not torch.cuda.is_available():
        pytest.skip("Native Sage2 CUDA Graph parity requires CUDA.")

    prebuilt_path = os.environ.get(_PREBUILT_EXTENSION_ENV)
    if prebuilt_path:
        extension = _load_prebuilt_extension(Path(prebuilt_path).expanduser())
    else:
        extension = native.load_extension()
        if extension is None:
            pytest.skip(
                f"Native extension is unavailable: {native.extension_load_error()}"
            )

    required_symbols = (
        "sage2_is_built",
        "sage2_is_runtime_supported",
        "sage2_test_attention",
    )
    if any(not callable(getattr(extension, name, None)) for name in required_symbols):
        pytest.skip("The native extension does not expose its Sage2 test interface.")
    if not bool(extension.sage2_is_built()):
        pytest.skip("The native extension was built with Sage2 stubs.")
    device_index = torch.cuda.current_device()
    if not bool(extension.sage2_is_runtime_supported(device_index)):
        pytest.skip(f"Native Sage2 is unsupported on CUDA device {device_index}.")
    return extension


def _random_tensor(generator: torch.Generator, *, sequence: int) -> Tensor:
    """Create one deterministic contiguous CUDA tensor for the probe."""
    return torch.randn(
        (1, sequence, _HEADS, _HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()


def _run_step(
    step: Callable[[Tensor, Tensor, Tensor], Tensor],
    inputs: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Run one attention step and clone its output after all streams finish."""
    output = step(*inputs)
    torch.cuda.synchronize()
    return output.clone()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Native Sage2 raw CUDA kernels launch on the legacy default stream, so "
        "CUDAGraphWrapper capture omits the attention work and replay stays stale."
    ),
)
def test_native_sage2_cudagraph_replay_matches_eager(
    native_sage2_extension: ModuleType,
) -> None:
    """Require capture and changed-input replay to match eager Native Sage2."""
    generator = torch.Generator(device="cuda").manual_seed(20260921)
    key_prefix = _random_tensor(generator, sequence=_CHUNK_TOKENS)
    value_prefix = _random_tensor(generator, sequence=_CHUNK_TOKENS)
    inputs = tuple(
        (
            _random_tensor(generator, sequence=_CHUNK_TOKENS),
            torch.cat(
                (key_prefix, _random_tensor(generator, sequence=_CHUNK_TOKENS)),
                dim=1,
            ),
            torch.cat(
                (value_prefix, _random_tensor(generator, sequence=_CHUNK_TOKENS)),
                dim=1,
            ),
        )
        for _ in range(3)
    )

    def attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return native_sage2_extension.sage2_test_attention(
            query,
            key,
            value,
            False,
        )

    eager_outputs = tuple(_run_step(attention, step_inputs) for step_inputs in inputs)
    wrapper = CUDAGraphWrapper(attention, warmup_iters=1)
    graph_outputs = tuple(_run_step(wrapper, step_inputs) for step_inputs in inputs)

    # CUDAGraphWrapper captures the current stream. Vendored Sage2 launches in
    # fused.cu and qk_int_sv_f8_cuda_sm90.cu omit the stream launch argument,
    # so attention currently escapes capture on the legacy default stream.
    if wrapper._graph is None:
        raise RuntimeError("The probe did not capture a CUDA graph.")
    if not torch.equal(graph_outputs[0], eager_outputs[0]):
        raise RuntimeError("The eager graph-warmup call did not match Sage2.")
    if torch.equal(eager_outputs[1], eager_outputs[2]):
        raise RuntimeError("Changed inputs did not change the eager reference.")

    output_matches = tuple(
        torch.equal(actual, expected)
        for actual, expected in zip(graph_outputs, eager_outputs, strict=True)
    )
    replay_tracks_changed_input = not torch.equal(
        graph_outputs[1],
        graph_outputs[2],
    )
    assert all(output_matches) and replay_tracks_changed_input, (
        f"eager parity by step: {output_matches}; "
        f"changed-input replay updated output: {replay_tracks_changed_input}"
    )
