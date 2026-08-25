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

"""CPU tests for the ``use_kv_cache=True`` parity helper's subclass factory.

The helper script at ``parity_check/run_vendor_use_kv_cache.py`` runs
under GPU (it delegates to vendor's ``wan/generate.py`` via
``runpy``), but the ``__setattr__`` coercion that forces
``use_kv_cache=True`` is a pure-Python class transformation and
testable on CPU through a tiny ``WanPipeline`` stand-in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


_HELPER_PATH = (
    Path(__file__).parent / "parity_check" / "run_vendor_use_kv_cache.py"
).resolve()


def _load_helper_module():
    """Import the helper script without executing its ``__main__`` block.

    Uses :mod:`importlib.util` so the top-level definitions
    (``make_use_kv_cache_true_subclass``) become importable without
    triggering the GPU-only ``_patch_and_run``. Registered under a
    distinct name so it doesn't shadow any sibling module.
    """
    spec = importlib.util.spec_from_file_location(
        "hy_worldplay_use_kv_cache_helper", _HELPER_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"Failed to build module spec for helper at {_HELPER_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_subclass_coerces_use_kv_cache_to_true() -> None:
    """Subclass factory coerces ``self.use_kv_cache = False`` assignments back to ``True``.

    Mirrors the path vendor takes inside
    ``pipeline_wan_w_mem_relative_rope.WanPipeline.predict``: the
    in-method ``self.use_kv_cache = False`` is silently coerced so
    vendor's predict body takes the cache-prefill branch.
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        """Stand-in for vendor's :class:`WanPipeline`."""

        def __init__(self) -> None:
            # Vendor initialises use_kv_cache=True; the False
            # reassignment happens mid-predict.
            self.use_kv_cache = True

        def predict(self) -> None:
            self.use_kv_cache = False

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    instance = Patched()
    assert instance.use_kv_cache is True
    instance.predict()
    assert instance.use_kv_cache is True, (
        "Expected the False assignment inside predict() to be coerced "
        "back to True by the subclass __setattr__"
    )


def test_make_subclass_preserves_other_attributes() -> None:
    """Only ``use_kv_cache`` is coerced; other attributes pass through unchanged."""
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    instance = Patched()
    instance.some_other_attr = "hello"
    instance.use_kv_cache = False
    instance.numeric_attr = 42
    instance.list_attr = [1, 2, 3]
    assert instance.some_other_attr == "hello"
    assert instance.use_kv_cache is True
    assert instance.numeric_attr == 42
    assert instance.list_attr == [1, 2, 3]


def test_make_subclass_idempotent() -> None:
    """Applying the transform twice still coerces the assignment.

    Double-wrapping produces a deeper subclass tree, but every level's
    ``__setattr__`` routes through ``super().__setattr__`` so the
    outermost layer's coercion still fires.
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    OncePatched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    TwicePatched = helper.make_use_kv_cache_true_subclass(OncePatched)
    instance = TwicePatched()
    instance.use_kv_cache = False
    assert instance.use_kv_cache is True


def test_make_subclass_sets_descriptive_name() -> None:
    """Generated subclass ``__name__`` includes both the base name and the ``UseKvCacheTrue`` tag.

    Makes the override obvious in tracebacks / ``repr`` rather than
    showing a bare ``_UseKvCacheTrue``.
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    assert "FakeWanPipeline" in Patched.__name__
    assert "UseKvCacheTrue" in Patched.__name__
    assert Patched.__qualname__ == Patched.__name__
