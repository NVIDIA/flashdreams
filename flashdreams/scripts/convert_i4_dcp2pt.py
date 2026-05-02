#!/usr/bin/env python3
"""Convert an I4 Cosmos SIL DCP checkpoint to a FlashDreams .pt.

The I4 checkpoint stores the DiT under the ``net.`` prefix. This script
loads it through a matching wrapper and saves only the bare
``CosmosDiTNetwork`` state dict.

The exported checkpoint is intentionally pre-fusion: it preserves the
training-time padding-mask channel in ``x_embedder``. Normal FlashDreams
inference calls ``update_parameters_after_loading_checkpoint()`` after
``load_state_dict()``, which fuses the padding-mask channel and output
shuffle at load time. If you load the exported .pt outside the standard
``CosmosTransformer`` path, call that method before running the network.

Example:

    python flashdreams/scripts/convert_i4_dcp2pt.py \\
        --checkpoint_path s3://bucket/path/to/dcp/model \\
        --output_path checkpoints/bidirectional.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flashdreams.core.checkpoint.load import load_distributed_checkpoint
from flashdreams.recipes.alpadreams.config import ALPADREAMS_CONFIG_BUILDERS
from flashdreams.recipes.alpadreams.transformer import CosmosTransformerConfig
from flashdreams.recipes.alpadreams.transformer.impl.network import CosmosDiTNetwork

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREDENTIAL_PATH = REPO_ROOT / "credentials/s3_checkpoint.secret"
DEFAULT_CONFIG_NAME = "sv_35steps_chunk48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m"
assert DEFAULT_CONFIG_NAME in ALPADREAMS_CONFIG_BUILDERS


class NetCheckpointWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Destination .pt path for the exported CosmosDiTNetwork state dict.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="I4/SIL distributed checkpoint directory, either local or s3://.",
    )
    parser.add_argument(
        "--credential_path",
        type=Path,
        default=DEFAULT_CREDENTIAL_PATH,
        help=(
            "S3 credential JSON used when --checkpoint_path is s3://. "
            f"Defaults to {DEFAULT_CREDENTIAL_PATH}."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        choices=sorted(ALPADREAMS_CONFIG_BUILDERS.keys()),
        help=(
            "Alpadreams config builder used to construct the CosmosDiTNetwork "
            "shape before loading the DCP checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_config = ALPADREAMS_CONFIG_BUILDERS[args.config_name]()
    transformer_config = pipeline_config.diffusion_model.transformer
    assert isinstance(transformer_config, CosmosTransformerConfig), (
        "convert_i4_dcp2pt requires an Alpadreams Cosmos transformer config; "
        f"got {type(transformer_config).__name__}."
    )
    wrapper = NetCheckpointWrapper(
        CosmosDiTNetwork(
            config=transformer_config.network,
        )
    )
    load_distributed_checkpoint(
        model=wrapper,
        checkpoint_path=args.checkpoint_path,
        credential_path=str(args.credential_path),
        check_success=True,
        local_cache_dir=None,
    )
    network = wrapper.net
    torch.save(network.state_dict(), args.output_path)

    # Keep the exported state dict pre-fusion. CosmosTransformer applies
    # this after loading so the saved file remains compatible with the
    # original training-time checkpoint shape.
    # network.update_parameters_after_loading_checkpoint()

    print(f"saved pre-fusion FlashDreams checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
