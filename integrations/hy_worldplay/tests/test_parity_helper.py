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

"""CPU tests for the phase 2b.6 ``use_kv_cache=True`` parity helper.

The helper at
``integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py``
is GPU-only at runtime (it delegates to vendor's
``HY-WorldPlay/wan/generate.py`` via :func:`runpy.run_path`), but the
``__setattr__`` coercion that forces ``use_kv_cache=True`` on the
vendor ``WanPipeline`` is a pure-Python class transformation and
fully testable on CPU. We exercise the
:func:`make_use_kv_cache_true_subclass` factory through a tiny
:class:`WanPipeline` stand-in -- the real vendor class is not
imported here (it would pull the entire HY-WorldPlay tree + heavy
deps which only live in the parity sub-venv).
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

    Uses :mod:`importlib.util` so the helper's top-level definitions
    (``make_use_kv_cache_true_subclass``) become importable without
    triggering ``_patch_and_run`` (which requires the HY-WorldPlay
    vendor tree + GPU). The helper module is registered under a
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
    """The subclass factory intercepts ``use_kv_cache=False`` assignments.

    Mirrors the path vendor takes inside
    ``pipeline_wan_w_mem_relative_rope.WanPipeline.predict`` (line 707:
    ``self.use_kv_cache = False``): after the helper's transform, that
    same assignment is silently coerced to ``True`` so vendor's predict
    body takes the cache-prefill branch.
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        """Stand-in for vendor's :class:`WanPipeline`."""

        def __init__(self) -> None:
            # WanPipeline initialises use_kv_cache=True in __init__; the
            # False reassignment happens mid-predict (line 707).
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
    """Applying the transform twice doesn't break the coercion chain.

    Double-wrapping produces a deeper subclass tree but every level's
    ``__setattr__`` still routes through ``super().__setattr__``, so
    the outermost layer's coercion still fires. Guards against future
    refactors that might accidentally short-circuit the recursion.
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
    """The generated class has a discoverable ``__name__`` for debugging.

    When we patch ``_vendor_pipe_mod.WanPipeline = make_use_kv_cache_true_subclass(...)``
    and later traceback / repr inspection shows the class name, we
    want the override to be obvious (not just ``_UseKvCacheTrue``).
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    assert "FakeWanPipeline" in Patched.__name__
    assert "UseKvCacheTrue" in Patched.__name__
    assert Patched.__qualname__ == Patched.__name__
