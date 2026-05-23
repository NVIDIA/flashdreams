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

"""Run vendor's ``wan/generate.py`` with ``WanPipeline.use_kv_cache`` forced True.

Phase 2b.6 close path (Option C in the phase-2b design spec).

Mirrors the invocation in :file:`run.sh`'s ``[run]`` block
(``torchrun wan/generate.py ...``) but with a runtime monkey-patch
that intercepts :class:`WanPipeline`'s ``__setattr__`` to coerce
``use_kv_cache=True``. This lets the parity diff compare the native
HY runner against vendor's cache-prefill code path (the
``use_kv_cache=True`` branch in vendor's pipeline) instead of the
default single-forward-pass branch.

The default ``self.use_kv_cache = False`` assignment lives *inside*
:meth:`WanPipeline.predict` at line 707 of
``pipeline_wan_w_mem_relative_rope.py`` (mid-execution), so setting
the attribute before calling ``predict()`` won't help -- the
assignment will simply overwrite our value. Instead, we subclass
:class:`WanPipeline` with a ``__setattr__`` that maps any
``use_kv_cache`` assignment to ``True``; replace the
:class:`WanPipeline` reference in the module's namespace **before**
:file:`generate.py`'s ``from ... import WanPipeline`` resolves;
then use :func:`runpy.run_path` to execute :file:`generate.py` with
the patched class in place.

This helper is dispatched by :file:`run.sh` when ``USE_KV_CACHE_TRUE=1``
is set in the environment. CPU testability of the subclass factory
is covered by ``integrations/hy_worldplay/tests/test_parity_helper.py``
(no GPU / no vendor tree required for those tests).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Type, TypeVar

_T = TypeVar("_T")

_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR = _SCRIPT_DIR / "HY-WorldPlay"


def make_use_kv_cache_true_subclass(base: Type[_T]) -> Type[_T]:
    """Return a subclass of ``base`` that coerces ``use_kv_cache`` to True.

    Vendor's :meth:`WanPipeline.predict` hardcodes
    ``self.use_kv_cache = False`` mid-method (line 707 of
    ``pipeline_wan_w_mem_relative_rope.py``); we intercept that
    assignment so the predict body takes the ``use_kv_cache=True``
    branch (cache-prefill + chunk-1-only forward), which is the same
    architecture the native HY-WorldPlay runner already uses.

    Idempotent: applying the transform twice produces a deeper subclass
    chain but every level still routes through ``super().__setattr__``
    so the outermost layer's coercion remains effective.
    """

    class _UseKvCacheTrue(base):  # type: ignore[valid-type, misc]
        def __setattr__(self, name: str, value: object) -> None:
            if name == "use_kv_cache":
                value = True
            super().__setattr__(name, value)

    _UseKvCacheTrue.__name__ = f"_UseKvCacheTrue_{base.__name__}"
    _UseKvCacheTrue.__qualname__ = _UseKvCacheTrue.__name__
    return _UseKvCacheTrue


def _patch_and_run() -> None:
    """Patch the WanPipeline module binding, then delegate to vendor's generate.py.

    Mirrors :file:`run.sh`'s ``PYTHONPATH`` injection
    (``${REPO_DIR}:${REPO_DIR}/wan``) so vendor's import graph
    resolves; imports
    :mod:`wan.inference.pipeline_wan_w_mem_relative_rope` so we can
    rebind its ``WanPipeline`` symbol to the patched subclass; then
    invokes :func:`runpy.run_path` on :file:`wan/generate.py` with
    ``run_name="__main__"`` to execute the script's ``if __name__ ==
    "__main__":`` block. Vendor's :func:`argparse` parser reads
    :attr:`sys.argv` directly, so the CLI surface passes through
    unchanged.
    """
    if not _REPO_DIR.exists():
        raise FileNotFoundError(
            f"Vendor HY-WorldPlay tree not found at {_REPO_DIR}. "
            f"Run `bash {_SCRIPT_DIR / 'run.sh'}` once with the "
            f"default settings to clone + checkout the pinned commit."
        )
    sys.path.insert(0, str(_REPO_DIR))
    sys.path.insert(0, str(_REPO_DIR / "wan"))

    from wan.inference import (  # noqa: E402 (deferred import after sys.path setup)
        pipeline_wan_w_mem_relative_rope as _vendor_pipe_mod,
    )

    _vendor_pipe_mod.WanPipeline = make_use_kv_cache_true_subclass(
        _vendor_pipe_mod.WanPipeline
    )

    # Phase 2b.6.2 attention-impl probe. When ``HY_VENDOR_SDPA=1`` is
    # set the sdpa_patch swaps vendor's ``sageattn`` import for
    # ``F.scaled_dot_product_attention`` so vendor + native both
    # exercise the same attention kernel. Useful for isolating
    # numerical drift caused by sageattn's INT8 / FP8 path vs cudnn's
    # bf16 path -- once the structural bugs are closed, sageattn vs
    # cudnn is the dominant residual divergence source. No-op when
    # ``HY_VENDOR_SDPA`` is unset (default behaviour preserved).
    from sdpa_patch import install_sdpa_patch  # noqa: E402

    install_sdpa_patch()

    runpy.run_path(
        str(_REPO_DIR / "wan" / "generate.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    _patch_and_run()
