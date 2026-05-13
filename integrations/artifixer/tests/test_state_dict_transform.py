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

"""Key-naming tests for the merged-DMD state_dict transform.

CPU-only; uses a ``param_audit.json`` listing every key in the merged
ArtiFixer DMD safetensors to assert that
:func:`artifixer_dmd_state_dict_transform` rewrites the merged
diffusers-format keys into names that we expect to find on the
flashdreams ``ArtifixerDiTNetwork.state_dict()``. Provide the audit
JSON path via ``ARTIFIXER_PARAM_AUDIT_PATH``; the test is skipped if
unset.

These tests do *not* load the actual safetensors. They synthesize the
expected key set from the audit JSON and run it through the transform.
A separate GPU-side test (later phase) will load the real safetensors
into an instantiated network with ``load_state_dict(strict=True)`` and
verify zero missing / unexpected keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from artifixer.checkpoint import (
    DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING,
    artifixer_dmd_state_dict_transform,
)


def _audit_path() -> Path | None:
    raw = os.environ.get("ARTIFIXER_PARAM_AUDIT_PATH")
    return Path(raw) if raw else None


def _load_audit() -> dict[str, dict]:
    audit_path = _audit_path()
    if audit_path is None:
        pytest.skip(
            "ARTIFIXER_PARAM_AUDIT_PATH is unset. Point it at the "
            "param_audit.json listing every key in the merged "
            "ArtiFixer DMD safetensors to exercise the full-merged-"
            "checkpoint path."
        )
    if not audit_path.exists():
        pytest.skip(
            f"param audit JSON not found at {audit_path}; generate it "
            f"from the merged ArtiFixer DMD safetensors first."
        )
    return json.loads(audit_path.read_text())


def _synthetic_state_dict(audit: dict[str, dict]) -> dict[str, torch.Tensor]:
    """Build a dummy state_dict with one tensor per key in the audit."""
    state_dict: dict[str, torch.Tensor] = {}
    for category in ("base_shared", "artifixer_only"):
        for key, shape in audit[category].items():
            state_dict[key] = torch.zeros(shape)
    return state_dict


def test_transform_renames_known_diffusers_keys() -> None:
    """Spot-check a few diffusers -> WanDiTNetwork renames the mapping does."""
    sd = {
        "blocks.0.attn1.to_q.weight": torch.zeros(1536, 1536),
        "blocks.0.attn2.to_q.weight": torch.zeros(1536, 1536),
        "blocks.0.attn2.add_k_proj.weight": torch.zeros(1536, 1536),
        "blocks.0.attn2.add_v_proj.bias": torch.zeros(1536),
        "blocks.0.attn2.norm_added_k.weight": torch.zeros(1536),
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(8960, 1536),
        "blocks.0.ffn.net.2.weight": torch.zeros(1536, 8960),
        "blocks.0.norm2.weight": torch.zeros(1536),
        "scale_shift_table": torch.zeros(2, 1536),
        "proj_out.weight": torch.zeros(64, 1536),
    }
    out = artifixer_dmd_state_dict_transform(sd)

    expected_renames = {
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.add_k_proj.weight",
        "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.add_v_proj.bias",
        "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_added_k.weight",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.weight": "head.head.weight",
    }
    for src, dst in expected_renames.items():
        assert src not in out, f"diffusers key {src!r} should have been renamed"
        assert dst in out, f"expected remapped key {dst!r} missing from transform output"


def test_transform_preserves_opacity_and_camera_keys_unchanged() -> None:
    """ArtiFixer-only keys without ``attn2`` prefix pass through unchanged.

    ``opacity_embedding`` and ``camera_embedding`` are registered directly
    on :class:`ArtifixerBlock` (no nested wrapper), so their reference
    names match the flashdreams names verbatim.
    """
    sd = {
        "blocks.0.opacity_embedding.weight": torch.zeros(1536, 1024),
        "blocks.0.opacity_embedding.bias": torch.zeros(1536),
        "blocks.5.camera_embedding.weight": torch.zeros(1536, 1536),
    }
    out = artifixer_dmd_state_dict_transform(sd)
    assert set(out) == set(sd)


def test_transform_on_param_audit_yields_unique_keys() -> None:
    """Apply the transform to the entire merged-checkpoint key set and
    assert no two source keys collapse to the same target key.
    """
    audit = _load_audit()
    src_sd = _synthetic_state_dict(audit)
    out = artifixer_dmd_state_dict_transform(src_sd)
    assert len(out) == len(src_sd), (
        f"transform collapsed {len(src_sd) - len(out)} keys to duplicates"
    )


def test_transform_on_param_audit_matches_expected_block_layout() -> None:
    """End-to-end shape check on the merged-DMD audit.

    Required post-transform invariants per block ``X`` (30 blocks total):

      - self_attn.{q,k,v,o,norm_q,norm_k}
      - cross_attn.{q,k,v,o,norm_q,norm_k}
      - cross_attn.{add_k_proj,add_v_proj,norm_added_k} (ArtiFixer-only)
      - opacity_embedding, camera_embedding (ArtiFixer-only)
      - norm3, ffn.0, ffn.2, modulation
    """
    audit = _load_audit()
    sd = _synthetic_state_dict(audit)
    out = artifixer_dmd_state_dict_transform(sd)

    expected_per_block = [
        "self_attn.q.weight",
        "self_attn.k.weight",
        "self_attn.v.weight",
        "self_attn.o.weight",
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "cross_attn.q.weight",
        "cross_attn.k.weight",
        "cross_attn.v.weight",
        "cross_attn.o.weight",
        "cross_attn.norm_q.weight",
        "cross_attn.norm_k.weight",
        "cross_attn.add_k_proj.weight",
        "cross_attn.add_v_proj.weight",
        "cross_attn.norm_added_k.weight",
        "opacity_embedding.weight",
        "camera_embedding.weight",
        "norm3.weight",
        "ffn.0.weight",
        "ffn.2.weight",
        "modulation",
    ]
    num_blocks = max(
        int(k.split(".", 2)[1]) for k in out if k.startswith("blocks.")
    ) + 1
    assert num_blocks == 30, f"expected 30 transformer blocks, got {num_blocks}"
    for block_idx in range(num_blocks):
        for suffix in expected_per_block:
            key = f"blocks.{block_idx}.{suffix}"
            assert key in out, f"missing post-transform key {key!r}"


def test_transform_resolves_globals() -> None:
    """Network-level keys land on the expected ``head.*`` / embedding names."""
    audit = _load_audit()
    sd = _synthetic_state_dict(audit)
    out = artifixer_dmd_state_dict_transform(sd)
    expected_globals = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "text_embedding.2.weight",
        "time_embedding.0.weight",
        "time_embedding.2.weight",
        "time_projection.1.weight",
        "head.modulation",
        "head.head.weight",
        "head.head.bias",
    ]
    for key in expected_globals:
        assert key in out, f"missing post-transform global {key!r}"


def test_diffusers_mapping_has_no_overlap_in_capture_groups() -> None:
    """Every regex in the mapping uses ``\\1`` / ``\\2`` consistently."""
    for old, new in DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING.items():
        # All groups referenced in ``new`` must exist in ``old``.
        for ref in (1, 2, 3):
            if f"\\{ref}" in new:
                # Count ``(`` not preceded by ``?:``.
                groups = old.count("(") - old.count("(?:")
                assert ref <= groups, (
                    f"pattern {old!r} -> {new!r} references group {ref} "
                    f"but pattern only has {groups} capture group(s)"
                )
