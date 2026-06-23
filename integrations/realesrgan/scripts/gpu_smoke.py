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

"""Run a checkpoint-backed Real-ESRGAN GPU smoke test."""

from __future__ import annotations

import argparse
import time
from typing import Literal, cast

import torch

from realesrgan.upsampler import RealESRGANUpsampler, default_model_name


def main() -> None:
    """Load a Real-ESRGAN checkpoint and run one synthetic RGB frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=2)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    scale = cast(Literal[2, 4], args.scale)
    model_name = args.model_name or default_model_name(scale)
    model_start = time.perf_counter()
    upsampler = RealESRGANUpsampler(
        scale=scale,
        model_name=model_name,
        tile=args.tile,
        pre_pad=0,
        half=not args.fp32,
        device=args.device,
    )
    _synchronize(args.device)
    model_elapsed = time.perf_counter() - model_start

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    frame = torch.zeros(3, args.size, args.size)
    infer_start = time.perf_counter()
    output = upsampler.upsample_frame_tensor(frame)
    _synchronize(args.device)
    infer_elapsed = time.perf_counter() - infer_start

    expected_shape = (3, args.size * scale, args.size * scale)
    if output.shape != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, got {output.shape}"
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("Real-ESRGAN output contains non-finite values.")

    peak_mb = (
        torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
        if args.device.startswith("cuda")
        else 0.0
    )
    device_name = (
        torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda")
        else args.device
    )
    print(f"device={device_name}")
    print(
        f"output_shape={tuple(output.shape)} dtype={output.dtype} "
        f"range=({float(output.min()):.4f}, {float(output.max()):.4f})"
    )
    print(
        f"model_load_s={model_elapsed:.3f} "
        f"infer_s={infer_elapsed:.3f} peak_allocated_mb={peak_mb:.1f}"
    )


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
