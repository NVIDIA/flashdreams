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

"""Musubi LoRA conversion and merging for the native H3 transformer."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import torch

CONVERSION_VERSION = "5"
MUSUBI_KEY = re.compile(
    r"^lora_unet_blocks_(?P<block>\d+)_"
    r"(?P<module>attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)"
    r"\.(?P<part>alpha|lora_down\.weight|lora_up\.weight)$"
)
DIRECT_TARGETS = {
    "attn_out_proj": "attn.to_out.0",
    "mlp_fc1": "ff.net.0.proj",
    "mlp_fc2": "ff.net.2",
}


def _lora_cache_root() -> Path:
    configured = os.environ.get("MINIMAX_H3_LORA_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(
            os.environ.get("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
        ).expanduser()
        / "minimax_h3"
        / "lora"
    )


def resolve_lora(source: str, weight_name: str | None = None) -> tuple[Path, str]:
    """Resolve a local Musubi adapter or download one Hub file."""
    local = Path(source).expanduser()
    if local.is_file():
        return local.resolve(), "local"
    if local.exists():
        raise ValueError(f"LoRA source is not a file: {local}")

    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(source)
    if weight_name is None:
        candidates = sorted(
            sibling.rfilename
            for sibling in info.siblings or []
            if sibling.rfilename.endswith(".safetensors")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"LoRA repository {source!r} contains {len(candidates)} safetensors "
                "files; pass --lora-weight-name explicitly"
            )
        weight_name = candidates[0]
    path = hf_hub_download(source, filename=weight_name, revision=info.sha)
    return Path(path), info.sha or "unversioned"


def _converted_path(source: Path, revision: str) -> Path:
    identity = (
        f"{source.resolve()}:{source.stat().st_size}:"
        f"{source.stat().st_mtime_ns}:{revision}:{CONVERSION_VERSION}"
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return _lora_cache_root() / f"{source.stem}-{digest}.flashdreams.safetensors"


def _source_groups(handle: Any) -> dict[tuple[int, str], dict[str, str]]:
    groups: dict[tuple[int, str], dict[str, str]] = {}
    unknown: list[str] = []
    source_keys = handle.keys()
    for key in source_keys:
        match = MUSUBI_KEY.fullmatch(key)
        if match is None:
            unknown.append(key)
            continue
        group = (int(match["block"]), match["module"])
        groups.setdefault(group, {})[match["part"]] = key
    if unknown:
        raise ValueError(f"unsupported MiniMax H3 LoRA tensors: {unknown[:3]}")
    modules = {"attn_qkv_proj", *DIRECT_TARGETS}
    expected = {(block, module) for block in range(50) for module in modules}
    if set(groups) != expected:
        missing = sorted(expected - set(groups))
        extra = sorted(set(groups) - expected)
        raise ValueError(
            "LoRA does not cover the expected 50 H3 blocks "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )
    for group, parts in groups.items():
        if set(parts) != {"alpha", "lora_down.weight", "lora_up.weight"}:
            raise ValueError(f"incomplete LoRA tensor group {group}: {sorted(parts)}")
    return groups


def convert_musubi_lora(source: Path, output: Path) -> Path:
    """Translate Musubi adapter names to the native H3 modules."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    converted: dict[str, torch.Tensor] = {}
    output_metadata = {
        "format": "pt",
        "source_format": "musubi-minimax-h3",
        "source_file": source.name,
        "conversion": "split fused qkv; fold alpha/rank into lora_B",
        "conversion_version": CONVERSION_VERSION,
    }
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        architecture = metadata.get("modelspec.architecture")
        if architecture not in {None, "MiniMax-H3/lora"}:
            raise ValueError(
                f"expected a MiniMax-H3 LoRA, found architecture {architecture!r}"
            )
        if training_mode := metadata.get("ss_h3_training_mode"):
            output_metadata["training_mode"] = training_mode
        groups = _source_groups(handle)
        for (block, module), parts in sorted(groups.items()):
            down = handle.get_tensor(parts["lora_down.weight"])
            up = handle.get_tensor(parts["lora_up.weight"])
            alpha = float(handle.get_tensor(parts["alpha"]).item())
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(
                    f"invalid LoRA shapes for block {block} {module}: "
                    f"{down.shape}, {up.shape}"
                )
            scaled_up = up * (alpha / down.shape[0])
            block_prefix = f"transformer.transformer_blocks.{block}"
            if module == "attn_qkv_proj":
                if scaled_up.shape[0] % 3:
                    raise ValueError(
                        f"QKV LoRA output is not divisible by three in block {block}: "
                        f"{scaled_up.shape}"
                    )
                for target, target_up in zip(
                    ("attn.to_q", "attn.to_k", "attn.to_v"),
                    scaled_up.chunk(3, dim=0),
                    strict=True,
                ):
                    prefix = f"{block_prefix}.{target}"
                    converted[f"{prefix}.lora_A.weight"] = down.clone()
                    converted[f"{prefix}.lora_B.weight"] = target_up
            else:
                prefix = f"{block_prefix}.{DIRECT_TARGETS[module]}"
                converted[f"{prefix}.lora_A.weight"] = down
                converted[f"{prefix}.lora_B.weight"] = scaled_up

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    save_file(converted, str(temporary), metadata=output_metadata)
    os.replace(temporary, output)
    return output


def prepare_lora(source: str, weight_name: str | None = None) -> Path:
    """Return a converted, cached native adapter path."""
    resolved, revision = resolve_lora(source, weight_name)
    output = _converted_path(resolved, revision)
    if not output.is_file():
        convert_musubi_lora(resolved, output)
    return output


def load_lora(
    transformer: Any,
    source: str,
    scale: float,
    weight_name: str | None = None,
) -> Path:
    """Merge a converted Musubi adapter into the native BF16 transformer."""
    if not 0 <= scale <= 4:
        raise ValueError("LoRA scale must be between 0 and 4")
    converted = prepare_lora(source, weight_name)
    from safetensors import safe_open

    with safe_open(converted, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        expected_b = {
            key.removesuffix(".lora_A.weight") + ".lora_B.weight" for key in a_keys
        }
        if expected_b != {key for key in keys if key.endswith(".lora_B.weight")}:
            raise ValueError(f"incomplete converted LoRA pairs in {converted}")
        with torch.no_grad():
            for a_key in a_keys:
                prefix = a_key.removeprefix("transformer.").removesuffix(
                    ".lora_A.weight"
                )
                a = handle.get_tensor(a_key)
                b = handle.get_tensor(
                    a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
                )
                module = transformer.get_submodule(prefix)
                if not hasattr(module, "weight"):
                    raise ValueError(f"LoRA target has no weight: {prefix}")
                down = a.to(device=module.weight.device, dtype=module.weight.dtype)
                up = b.to(device=module.weight.device, dtype=module.weight.dtype)
                module.weight.addmm_(up, down, beta=1.0, alpha=scale)
    return converted
