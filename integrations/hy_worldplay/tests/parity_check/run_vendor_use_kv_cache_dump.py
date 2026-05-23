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

"""Run vendor's ``wan/generate.py`` with ``use_kv_cache=True`` AND tensor dumps enabled.

Combines :mod:`run_vendor_use_kv_cache` (coerces ``use_kv_cache=True``)
with :mod:`dump_patch` (monkey-patches attention + transformer to
write JSONL records mirroring :mod:`hy_worldplay._debug_dump`).
Activate dumps with ``HY_DEBUG_DUMP=/path/to/vendor_dump.jsonl``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from run_vendor_use_kv_cache import make_use_kv_cache_true_subclass  # noqa: E402

_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR = _SCRIPT_DIR / "HY-WorldPlay"


def _patch_and_run() -> None:
    if not _REPO_DIR.exists():
        raise FileNotFoundError(
            f"Vendor HY-WorldPlay tree not found at {_REPO_DIR}. "
            f"Run `bash {_SCRIPT_DIR / 'run.sh'}` once with the "
            f"default settings to clone + checkout the pinned commit."
        )
    sys.path.insert(0, str(_REPO_DIR))
    sys.path.insert(0, str(_REPO_DIR / "wan"))
    sys.path.insert(0, str(_SCRIPT_DIR))

    from wan.inference import (  # noqa: E402
        pipeline_wan_w_mem_relative_rope as _vendor_pipe_mod,
    )

    _vendor_pipe_mod.WanPipeline = make_use_kv_cache_true_subclass(
        _vendor_pipe_mod.WanPipeline
    )

    import dump_patch  # noqa: E402

    dump_patch.install_patches()

    from vae_mean_patch import install_vae_mean_patch  # noqa: E402

    install_vae_mean_patch()

    runpy.run_path(
        str(_REPO_DIR / "wan" / "generate.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    _patch_and_run()
