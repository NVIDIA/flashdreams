# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local preparation of native Framework student exports for Transfer inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from cosmos_interactive.checkpoint_remap import check_source_coverage

from ..config import CONTROL_ATTENTION_MODE, CONTROL_HINTS, EXPORT_METADATA_KEY
from ..config import SYSTEM_PROMPT_ID, TransferModelConfig, fetch_release

FRAMEWORK_REVISION = "5950e7b4ea3677b99a2de279a7ed7058b5446e42"
"""Framework revision whose resolved native config and export format are supported."""


def transfer_metadata(
    payload: dict[str, Any],
    *,
    checkpoint_id: str,
    hints: list[str],
    emphasize_control_in_prompt: bool = False,
) -> dict[str, Any]:
    """Translate a resolved native model config without filling missing settings.

    ``hints`` and the prompt option are supplied from the trained model's
    reference request; native exports omit the training loader's modality policy.
    """
    config = payload["model"]["config"]
    expected = {
        "video_temporal_causal": True,
        "vision_gen": True,
        "action_gen": False,
        "sound_gen": False,
        "precision": "bfloat16",
        "teacher_forcing_frames_per_chunk": 1,
        "teacher_forcing_kv_implementation": "singleview_threeway_kv",
        "kv_cache_dtype": None,
    }
    replay = config["teacher_forcing_replay_policy"]
    replay_expected = {
        "control_visibility": "causal",
        "controls_read_strict_past_clean_rgb": True,
        "clean_pass_causality": "frame",
        "decomposed_temporal_window_seconds": None,
    }
    for source, required in ((config, expected), (replay, replay_expected)):
        for key, value in required.items():
            if key not in source or source[key] != value:
                raise ValueError(
                    f"Unsupported native setting {key}: expected {value!r}"
                )
    if not checkpoint_id.strip():
        raise ValueError(
            "checkpoint_id must identify the selected training run and iteration"
        )
    if (
        not hints
        or len(set(hints)) != len(hints)
        or any(h not in CONTROL_HINTS for h in hints)
    ):
        raise ValueError(
            f"hints must name distinct trained controls from {CONTROL_HINTS}"
        )
    tokenizer = config["tokenizer"]
    expert = config["diffusion_expert_config"]
    vae_target = tokenizer.get("_target", tokenizer.get("_target_"))
    if (
        vae_target
        not in (
            "wan2pt2_vae_interface",
            "projects.cosmos3.cosmos3.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface",
        )
        or tokenizer["temporal_compression_factor"] != 4
        or tokenizer["spatial_compression_factor"] != 16
        or tokenizer["causal"] is not True
        or expert["patch_spatial"] != 2
        or expert["vision_temporal_position_mode"] != "latent_index"
    ):
        raise ValueError(
            "Native export must use causal Wan 2.2 geometry and latent-index positions"
        )
    metadata = {
        "checkpoint_id": checkpoint_id,
        "chunk_size": config["teacher_forcing_frames_per_chunk"],
        "chunk_partition": [config["teacher_forcing_frames_per_chunk"]] * 2,
        "conditioning": {
            "mode": "control_video",
            "hints": hints,
            "system_prompt_id": SYSTEM_PROMPT_ID,
            "transfer_control_attention_mode": CONTROL_ATTENTION_MODE,
            "share_vision_temporal_positions": True,
            "emphasize_control_in_prompt": emphasize_control_in_prompt,
        },
        "fixed_step_sampler_config": {
            "t_list": config["fixed_step_sampler_config"]["t_list"],
            "sample_type": config["fixed_step_sampler_config"]["sample_type"],
            "num_train_timesteps": config["rectified_flow_inference_config"][
                "num_train_timesteps"
            ],
        },
        "latent_patch_size": expert["patch_spatial"],
        "mrope": {
            "base_fps": expert["base_fps"],
            "enable_fps_modulation": expert["enable_fps_modulation"],
            "reset_spatial_ids": expert["unified_3d_mrope_reset_spatial_ids"],
            "temporal_modality_margin": expert[
                "unified_3d_mrope_temporal_modality_margin"
            ],
        },
        "temporal_compression_factor": tokenizer["temporal_compression_factor"],
        "vae_spatial_compression_factor": tokenizer["spatial_compression_factor"],
        "text_cache_max_len": config["ar_static_und_cache_max_len"],
        "window_frames": config["kv_cache_inference_size"],
        "sink_frames": config["attention_sink_size"],
    }
    TransferModelConfig.from_metadata(metadata).with_schedule(
        t_list=metadata["fixed_step_sampler_config"]["t_list"]
    )
    return metadata


def prepare_export(
    source: Path,
    output: Path,
    *,
    tokenizer: Path,
    checkpoint_id: str,
    hints: list[str],
    emphasize_control_in_prompt: bool = False,
) -> Path:
    """Create a fresh compatibility directory pointing to unchanged local weights.

    This checks metadata and key coverage; GPU parity remains a separate check.
    Keep the source export and tokenizer in place while using the resulting links.
    """
    source, tokenizer = source.resolve(), tokenizer.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if any((source / name).is_file() for name in (".metadata", "model/.metadata")):
        raise ValueError(
            "Export native DCP student EMA with Framework export_model first"
        )
    provenance = json.loads((source / "checkpoint.json").read_text())
    if (
        provenance.get("source_weights") != "ema"
        and provenance.get("use_ema_weights") is not True
    ):
        raise ValueError("Native export must declare selected student EMA weights")
    config_bytes = (source / "config.json").read_bytes()
    metadata = transfer_metadata(
        json.loads(config_bytes),
        checkpoint_id=checkpoint_id,
        hints=hints,
        emphasize_control_in_prompt=emphasize_control_in_prompt,
    )
    index_bytes = (source / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Native export requires a nonempty safetensors weight_map")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("Native weight_map keys and shard names must be strings")
    keys: list[str] = []
    for key in weight_map:
        assert isinstance(key, str)
        keys.append(key)
    coverage = check_source_coverage(keys)
    if not coverage["ok"]:
        raise ValueError(f"Native checkpoint key remapping failed: {coverage}")
    shards = set(weight_map.values())
    for shard in shards:
        assert isinstance(shard, str)
        path = Path(shard)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Weight shard must stay within the source export: {shard}"
            )
        # Metadata is written after the shard links. A shard named
        # checkpoint.json/preparation.json would make those writes follow a
        # link into the original export, violating source preservation.
        if path.suffix != ".safetensors":
            raise ValueError(f"Weight shard must be a .safetensors file: {shard}")
        if not (source / path).is_file():
            raise FileNotFoundError(source / path)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (tokenizer / filename).is_file():
            raise FileNotFoundError(tokenizer / filename)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "model.safetensors.index.json").write_bytes(index_bytes)
        for shard in shards:
            assert isinstance(shard, str)
            destination = staging / shard
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source / shard)
        (staging / "text_tokenizer").symlink_to(tokenizer, target_is_directory=True)
        (staging / "checkpoint.json").write_text(
            json.dumps({EXPORT_METADATA_KEY: metadata}, indent=2) + "\n"
        )
        (staging / "preparation.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_weights": "ema",
                    "export_provenance": provenance,
                    "checkpoint_id": checkpoint_id,
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                    "reference_framework_revision": FRAMEWORK_REVISION,
                    "key_coverage": coverage,
                    "gpu_parity_verified": False,
                    "emphasize_control_in_prompt": emphasize_control_in_prompt,
                },
                indent=2,
            )
            + "\n"
        )
        fetch_release(staging)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def main() -> None:
    """Prepare a native student export without loading weights or using a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--hint", action="append", choices=CONTROL_HINTS, required=True)
    parser.add_argument(
        "--emphasize-control-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    print(
        prepare_export(
            args.source,
            args.output,
            tokenizer=args.tokenizer,
            checkpoint_id=args.checkpoint_id,
            hints=args.hint,
            emphasize_control_in_prompt=args.emphasize_control_in_prompt,
        )
    )


if __name__ == "__main__":
    main()
