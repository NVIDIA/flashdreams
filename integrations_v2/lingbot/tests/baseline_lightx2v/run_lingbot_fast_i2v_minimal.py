# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal LightX2V LingBot launcher.

This avoids importing ``lightx2v.infer`` (which eagerly imports many runner
families and optional operator stacks) and instead registers only the LingBot
fast runner path needed by this baseline.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

# ``lightx2v`` and ``lightx2v_platform`` live in the cloned ``LightX2V/`` tree
# and are imported at runtime via ``PYTHONPATH`` set by ``run.sh``. They are
# not pip-installed into this venv, so static type checkers cannot resolve
# them; suppress per-import.
from lightx2v.common.ops.attn.torch_sdpa import (  # ty: ignore[unresolved-import]
    TorchSDPAWeight,
)
from lightx2v.models.runners.wan.wan_lingbot_fast_runner import (  # ty: ignore[unresolved-import]
    LingbotFastRunner,  # noqa: F401
)
from lightx2v.utils.input_info import (  # ty: ignore[unresolved-import]
    init_empty_input_info,
    update_input_info_from_dict,
)
from lightx2v.utils.registry_factory import (  # ty: ignore[unresolved-import]
    RUNNER_REGISTER,
)
from lightx2v.utils.set_config import (  # ty: ignore[unresolved-import]
    set_config,
    set_parallel_config,
)
from lightx2v.utils.utils import (  # ty: ignore[unresolved-import]
    seed_all,
    validate_config_paths,
)
from lightx2v_platform.registry_factory import (  # ty: ignore[unresolved-import]
    PLATFORM_DEVICE_REGISTER,
)
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel


def _force_cudnn_sdpa_backend() -> None:
    """Force DiT ``torch_sdpa`` calls to use cuDNN backend only."""

    original_apply = TorchSDPAWeight.apply

    def _apply_with_cudnn(self, *args, **kwargs):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return original_apply(self, *args, **kwargs)

    TorchSDPAWeight.apply = _apply_with_cudnn  # type: ignore[method-assign]
    logger.info("Forced cuDNN backend inside torch_sdpa attention calls")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cls", type=str, default="lingbot_world_fast")
    parser.add_argument("--task", type=str, default="i2v")
    parser.add_argument("--support_tasks", type=str, nargs="+", default=[])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--dit_original_ckpt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--action_path", type=str, default="")
    parser.add_argument("--save_result_path", type=str, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    seed_all(args.seed)

    _force_cudnn_sdpa_backend()

    config = set_config(args)
    config["self_attn_1_type"] = "torch_sdpa"
    config["cross_attn_1_type"] = "torch_sdpa"
    config["cross_attn_2_type"] = "torch_sdpa"
    if args.dit_original_ckpt:
        config["dit_original_ckpt"] = args.dit_original_ckpt
    logger.info(
        "Using attention backends: self_attn_1_type={}, cross_attn_1_type={}, cross_attn_2_type={}",
        config["self_attn_1_type"],
        config["cross_attn_1_type"],
        config["cross_attn_2_type"],
    )
    validate_config_paths(config)

    if config.get("parallel"):
        platform_device = PLATFORM_DEVICE_REGISTER.get(
            os.getenv("PLATFORM", "cuda"), None
        )
        if platform_device is not None:
            platform_device.init_parallel_env()
            set_parallel_config(config)

    input_info = init_empty_input_info(args.task, args.support_tasks)
    update_input_info_from_dict(input_info, vars(args))

    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    runner.run_pipeline(input_info)

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
