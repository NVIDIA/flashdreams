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

"""Audit upstream's ``Wan2.2_VAE.pth`` keys against our internal residual-VAE shape.

Run on a GPU box (or any box with the parity sub-venv synced) to
decide whether ``Wan2.2_VAE.pth`` can replace the diffusers
safetensors path + the
:func:`wan22_ti2v_5b_vae_state_dict_transform` remap.

Outputs:
    * ``./Wan2.2_VAE.pth.keys.txt`` -- one ``<key> <shape>`` line per
      tensor in the upstream .pth.
    * ``./flashdreams_vae.keys.txt`` -- the same for our internal
      :class:`flashdreams.recipes.wan.autoencoder.vae.WanVAE` instance.
    * Console diff summary: how many keys match by name, how many need
      a remap, whether shapes line up.

Decision rule:
    * If >= 95%% of keys match by name AND every matched shape lines
      up -> drop the state_dict_transform entirely.
    * If keys diverge but a regex-renaming pass aligns them -> write a
      simpler new transform than the diffusers one.
    * Otherwise -> keep the diffusers path; document as not a win.

Usage:
    cd integrations/hy_worldplay/tests/parity_check
    uv run python audit_vae_pth.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch

PTH_URL = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
"""Upstream's single-file ``.pth`` Ruilong's comment 1 suggests we switch to."""

PTH_PATH = Path("./Wan2.2_VAE.pth")
"""Cache path for the downloaded checkpoint."""


def _ensure_downloaded() -> None:
    """Download ``Wan2.2_VAE.pth`` if not already cached locally."""
    if PTH_PATH.exists():
        size_gb = PTH_PATH.stat().st_size / (1024**3)
        print(f"[audit] using cached {PTH_PATH} ({size_gb:.2f} GiB)")
        return
    print(f"[audit] downloading {PTH_URL} -> {PTH_PATH} (~5 GiB, slow)")
    urlretrieve(PTH_URL, PTH_PATH)


def _dump_pth_keys() -> dict[str, tuple[int, ...]]:
    """Load the .pth (meta-tensors only, no weight transfer) and return ``{key: shape}``."""
    print(f"[audit] loading {PTH_PATH} meta-keys ...")
    sd = torch.load(
        PTH_PATH,
        map_location="meta",
        weights_only=True,
    )
    # Handle nested wrappers (some upstream pth files store under "model" / "state_dict").
    while isinstance(sd, dict) and "state_dict" in sd and len(sd) <= 4:
        sd = sd["state_dict"]
    return {k: tuple(v.shape) for k, v in sd.items()}


def main() -> None:
    """Download and dump the .pth's keys + shapes; eyeball-compare to our model."""
    _ensure_downloaded()
    pth_keys = _dump_pth_keys()

    out_path = Path("Wan2.2_VAE.pth.keys.txt")
    out_path.write_text("\n".join(f"{k} {pth_keys[k]}" for k in sorted(pth_keys)))
    print(f"[audit] wrote {out_path} ({len(pth_keys)} keys)")

    # Quick top-level layout summary so we can decide simplification potential.
    print()
    print("===== top-level prefixes =====")
    prefixes: dict[str, int] = {}
    for k in pth_keys:
        head = k.split(".", 1)[0]
        prefixes[head] = prefixes.get(head, 0) + 1
    for head, count in sorted(prefixes.items()):
        print(f"  {head:30s} {count}")

    print()
    print("===== first 30 keys =====")
    for k in sorted(pth_keys)[:30]:
        print(f"  {k:60s} {pth_keys[k]}")


if __name__ == "__main__":
    sys.exit(main())
