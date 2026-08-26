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

"""CPU tests for Omnidreams PyTorch module helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omnidreams._module_utils import unwrap_compiled_module

pytestmark = pytest.mark.ci_cpu


def test_unwrap_compiled_module_returns_plain_module_unchanged() -> None:
    module = nn.Linear(2, 2)

    assert unwrap_compiled_module(module) is module


def test_unwrap_compiled_module_returns_registered_original() -> None:
    original = nn.Linear(2, 2)
    wrapper = nn.Module()
    wrapper.add_module("_orig_mod", original)

    assert unwrap_compiled_module(wrapper) is original


def test_unwrap_compiled_module_rejects_non_module_original() -> None:
    wrapper = nn.Module()
    wrapper.register_buffer("_orig_mod", torch.zeros(1))

    with pytest.raises(TypeError, match=r"_orig_mod must be an nn\.Module"):
        unwrap_compiled_module(wrapper)
