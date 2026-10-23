# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map exported Cosmos3 checkpoint keys to FlashDreams parameter names.

Understanding and generation layer weights map to ``layers.{i}.und`` and
``layers.{i}.gen``. Video, vision and LiDAR adapters retain their tensor values.
Unknown source keys are rejected so strict loading can detect incompatible
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypeVar

from cosmos_interactive.diffusers_checkpoint import remap_diffusers_key

_NET_PREFIX = "model.net."
_LM_PREFIX = "language_model."
_LAYERS_PREFIX = "model.layers."

# ``_moe_gen`` either suffixes a leaf module (``q_proj_moe_gen``) or a parent
# module (``mlp_moe_gen.gate_proj``). Strip both forms to recover the und leaf.
_MOE_GEN_RE = re.compile(r"_moe_gen(?=\.|$)")

T = TypeVar("T")


def remap_key(key: str) -> str | None:
    """Rename one exported checkpoint key to its FlashDreams target name.

    Returns ``None`` for keys that are intentionally not part of the network
    weights (there are none expected today; ``None`` surfaces surprises).
    """
    if not key.startswith(_NET_PREFIX):
        return remap_diffusers_key(key)
    rest = key[len(_NET_PREFIX) :]

    # VFM adapters + action-conditioning pathway live directly under ``model.net``.
    if rest.startswith(
        (
            "vae2llm.",
            "llm2vae.",
            "time_embedder.",
            "action2llm.",
            "llm2action.",
            "lidar2llm.",
            "llm2lidar.",
            "rig_view_embed.",
        )
    ):
        return rest
    if rest == "action_modality_embed":  # bare nn.Parameter (camera checkpoint)
        return rest

    if not rest.startswith(_LM_PREFIX):
        return None
    rest = rest[len(_LM_PREFIX) :]  # after ``language_model.``

    # Vision (ViT) tower.
    if rest.startswith("visual."):
        return rest

    # Top-level LM head (sits at ``language_model.lm_head`` in the export).
    if rest.startswith("lm_head."):
        return rest

    # Transformer body: ``model.embed_tokens`` / ``model.norm`` / ``model.layers.*``.
    if rest.startswith("model."):
        body = rest[len("model.") :]
        if body.startswith("layers."):
            return _remap_layer(body)
        # embed_tokens.weight / norm.weight
        return body

    return None


def _remap_layer(body: str) -> str:
    """Map ``layers.{i}.<leaf>`` → ``layers.{i}.{und|gen}.<leaf>``."""
    m = re.match(r"^layers\.(\d+)\.(.+)$", body)
    if m is None:
        return body
    layer_idx, leaf = m.group(1), m.group(2)
    if _MOE_GEN_RE.search(leaf):
        pathway = "gen"
        leaf = _MOE_GEN_RE.sub("", leaf)
    else:
        pathway = "und"
    return f"layers.{layer_idx}.{pathway}.{leaf}"


def state_dict_transform(raw: dict[str, T]) -> dict[str, T]:
    """Rename every key in ``raw`` to the FlashDreams target name (tensors ride along).

    Raises ``KeyError`` if any source key has no mapping, so an unexpected key
    fails loudly rather than being silently dropped.
    """
    out: dict[str, T] = {}
    unmapped: list[str] = []
    for key, value in raw.items():
        target = remap_key(key)
        if target is None:
            unmapped.append(key)
            continue
        out[target] = value
    if unmapped:
        raise KeyError(
            f"{len(unmapped)} source keys have no target mapping, e.g. {unmapped[:5]}"
        )
    return out


def load_weight_keys(index_path: Path) -> list[str]:
    """Return every weight key from a safetensors index (no tensors loaded)."""
    data = json.loads(index_path.read_text())
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no 'weight_map' object")
    return sorted(weight_map)


def check_source_coverage(keys: list[str]) -> dict[str, object]:
    """Validate the remap covers all source keys with symmetric und/gen sets."""
    unmapped: list[str] = []
    targets: list[str] = []
    und_leaves: dict[int, set[str]] = {}
    gen_leaves: dict[int, set[str]] = {}

    for key in keys:
        target = remap_key(key)
        if target is None:
            unmapped.append(key)
            continue
        targets.append(target)
        layer_match = re.match(r"^layers\.(\d+)\.(und|gen)\.(.+)$", target)
        if layer_match:
            idx = int(layer_match.group(1))
            leaf = layer_match.group(3)
            bucket = und_leaves if layer_match.group(2) == "und" else gen_leaves
            bucket.setdefault(idx, set()).add(leaf)

    asymmetric = [
        idx for idx in und_leaves if und_leaves.get(idx) != gen_leaves.get(idx)
    ]
    duplicate_targets = len(targets) != len(set(targets))

    return {
        "num_source_keys": len(keys),
        "num_mapped": len(targets),
        "num_unmapped": len(unmapped),
        "unmapped_examples": unmapped[:10],
        "num_layers_und": len(und_leaves),
        "num_layers_gen": len(gen_leaves),
        "asymmetric_layers": asymmetric,
        "has_duplicate_targets": duplicate_targets,
        "ok": not unmapped and not asymmetric and not duplicate_targets,
    }


def _format(report: dict[str, object]) -> str:
    lines = [
        "Cosmos3 → FlashDreams checkpoint remap (source coverage)",
        f"  source keys       : {report['num_source_keys']}",
        f"  mapped            : {report['num_mapped']}",
        f"  unmapped          : {report['num_unmapped']} {report['unmapped_examples'] or ''}",
        f"  layers (und/gen)  : {report['num_layers_und']} / {report['num_layers_gen']}",
        f"  asymmetric layers : {report['asymmetric_layers']}",
        f"  duplicate targets : {report['has_duplicate_targets']}",
    ]
    lines.append(
        "  COVERAGE OK: every source key maps 1:1, symmetric MoT pathways"
        if report["ok"]
        else "  PROBLEMS FOUND (see above)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index_path",
        type=Path,
        help="Path to the exported checkpoint's model.safetensors.index.json",
    )
    args = parser.parse_args()
    report = check_source_coverage(load_weight_keys(args.index_path))
    print(_format(report))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
