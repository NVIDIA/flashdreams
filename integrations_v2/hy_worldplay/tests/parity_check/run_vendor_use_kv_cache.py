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

"""Run vendor's ``wan/generate.py`` with ``WanPipeline.use_kv_cache`` forced to ``True``.

Vendor's :meth:`WanPipeline.predict` hardcodes
``self.use_kv_cache = False`` mid-method, so the only way to flip the
branch is to subclass :class:`WanPipeline` with a ``__setattr__``
that coerces the assignment, rebind the symbol in the vendor module
before ``generate.py`` imports it, then dispatch via
:func:`runpy.run_path`. The CPU-testable subclass factory is covered
by ``tests/test_parity_helper.py``.
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
    """Return a subclass of ``base`` whose ``__setattr__`` coerces ``use_kv_cache`` to ``True``."""

    class _UseKvCacheTrue(base):  # type: ignore[valid-type, misc]
        def __setattr__(self, name: str, value: object) -> None:
            if name == "use_kv_cache":
                value = True
            super().__setattr__(name, value)

    _UseKvCacheTrue.__name__ = f"_UseKvCacheTrue_{base.__name__}"
    _UseKvCacheTrue.__qualname__ = _UseKvCacheTrue.__name__
    return _UseKvCacheTrue


def _patch_and_run() -> None:
    """Rebind ``WanPipeline`` in the vendor module, then dispatch ``wan/generate.py``."""
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

    # Optional diagnostic patches (gated by env vars; no-ops by default).
    from sdpa_patch import install_sdpa_patch  # noqa: E402

    install_sdpa_patch()

    from vae_mean_patch import install_vae_mean_patch  # noqa: E402

    install_vae_mean_patch()

    from vendor_profile_patch import install_vendor_profile_patch  # noqa: E402

    install_vendor_profile_patch()

    runpy.run_path(
        str(_REPO_DIR / "wan" / "generate.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    _patch_and_run()
