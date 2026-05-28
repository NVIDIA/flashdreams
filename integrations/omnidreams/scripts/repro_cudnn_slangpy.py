# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal cuDNN MHA + SlangPy/Vulkan reproducer.

This script intentionally bypasses OmniDreams model loading and calls the same
low-level cuDNN SDPA op used by ContextParallelAttention. The default tensor
shape/stride matches the first failing attention call captured from
interactive-drive with the cuDNN backend:

    q/k/v: (1, 16, 7040, 128), bf16
    q stride: (14417920, 128, 2048, 1)
    k/v stride: (43253760, 128, 2048, 1)

Typical runs:

    # Baseline: cuDNN only, expected to pass.
    .venv/bin/python integrations/omnidreams/scripts/repro_cudnn_slangpy.py --slang none

    # Same-process Vulkan device first, expected to fail on affected stacks.
    xvfb-run -a .venv/bin/python integrations/omnidreams/scripts/repro_cudnn_slangpy.py

    # Prime cuDNN on the same worker thread before SlangPy creates Vulkan.
    xvfb-run -a .venv/bin/python integrations/omnidreams/scripts/repro_cudnn_slangpy.py \
      --thread-mode worker-prime-before-slang
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Literal


SlangMode = Literal["none", "import", "window", "device", "window-device"]
ThreadMode = Literal["main", "worker-after-slang", "worker-prime-before-slang"]


@dataclass
class SlangObjects:
    module: Any | None = None
    window: Any | None = None
    device: Any | None = None


def _set_default_nvidia_icd() -> None:
    icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if "VK_ICD_FILENAMES" not in os.environ and os.path.exists(icd):
        os.environ["VK_ICD_FILENAMES"] = icd


def _print_env(torch: Any) -> None:
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"torch.version.cuda: {torch.version.cuda}", flush=True)
    print(f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}", flush=True)
    for name in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "VK_ICD_FILENAMES"):
        print(f"{name}: {os.environ.get(name)}", flush=True)


def _make_qkv(
    torch: Any,
    *,
    seq_len: int,
    batch: int,
    heads: int,
    head_dim: int,
    compact: bool,
) -> tuple[Any, Any, Any]:
    device = "cuda"
    dtype = torch.bfloat16
    if compact:
        q = torch.randn(
            (batch, heads, seq_len, head_dim), device=device, dtype=dtype
        ).contiguous()
        k = torch.randn(
            (batch, heads, seq_len, head_dim), device=device, dtype=dtype
        ).contiguous()
        v = torch.randn(
            (batch, heads, seq_len, head_dim), device=device, dtype=dtype
        ).contiguous()
        return q, k, v

    q_base = torch.randn(
        (batch, seq_len, heads, head_dim), device=device, dtype=dtype
    )
    kv_base = torch.randn(
        (batch, 3, seq_len, heads, head_dim), device=device, dtype=dtype
    )
    q = q_base.transpose(1, 2)
    k = kv_base[:, 0].transpose(1, 2)
    v = kv_base[:, 1].transpose(1, 2)
    return q, k, v


def _run_cudnn_attention(
    torch: Any,
    *,
    label: str,
    seq_len: int,
    batch: int,
    heads: int,
    head_dim: int,
    compact: bool,
    loops: int,
) -> None:
    q, k, v = _make_qkv(
        torch,
        seq_len=seq_len,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        compact=compact,
    )
    print(
        f"[{label}] q shape={tuple(q.shape)} dtype={q.dtype} stride={tuple(q.stride())}",
        flush=True,
    )
    print(
        f"[{label}] k shape={tuple(k.shape)} dtype={k.dtype} stride={tuple(k.stride())}",
        flush=True,
    )
    for index in range(loops):
        out, lse, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q,
            k,
            v,
            None,
            True,
        )
        torch.cuda.synchronize()
        print(
            f"[{label}] ok loop={index} out={tuple(out.shape)} lse={tuple(lse.shape)}",
            flush=True,
        )


def _setup_slang(mode: SlangMode, *, width: int, height: int) -> SlangObjects:
    if mode == "none":
        return SlangObjects()

    import slangpy as spy

    result = SlangObjects(module=spy)
    print("[slang] imported", flush=True)

    if mode in ("window", "window-device"):
        result.window = spy.Window(
            width=width,
            height=height,
            title="cudnn-slangpy-repro",
            resizable=False,
        )
        print("[slang] window created", flush=True)

    if mode in ("device", "window-device"):
        result.device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=False,
        )
        print(f"[slang] device={result.device.info.adapter_name}", flush=True)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce cuDNN MHA failure after SlangPy/Vulkan device init."
    )
    parser.add_argument("--seq-len", type=int, default=7040)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact contiguous [B,H,S,D] tensors instead of captured strides.",
    )
    parser.add_argument(
        "--slang",
        choices=("none", "import", "window", "device", "window-device"),
        default="device",
        help="Which SlangPy objects to create before the tested cuDNN call.",
    )
    parser.add_argument(
        "--thread-mode",
        choices=("main", "worker-after-slang", "worker-prime-before-slang"),
        default="main",
        help=(
            "main: run cuDNN on main thread after SlangPy. "
            "worker-after-slang: create SlangPy first, then run cuDNN on a worker. "
            "worker-prime-before-slang: worker primes cuDNN, main creates SlangPy, "
            "then the same worker runs cuDNN again."
        ),
    )
    parser.add_argument(
        "--prime-seq-len",
        type=int,
        default=None,
        help="Run one cuDNN call before SlangPy creation on the main thread.",
    )
    parser.add_argument(
        "--no-force-nvidia-icd",
        action="store_true",
        help="Do not set VK_ICD_FILENAMES to the NVIDIA ICD when it is unset.",
    )
    parser.add_argument("--window-width", type=int, default=320)
    parser.add_argument("--window-height", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.no_force_nvidia_icd:
        _set_default_nvidia_icd()

    import torch

    _print_env(torch)

    common = dict(
        seq_len=args.seq_len,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        compact=args.compact,
        loops=args.loops,
    )

    if args.thread_mode == "worker-prime-before-slang":
        ready = threading.Event()
        go = threading.Event()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                _run_cudnn_attention(
                    torch,
                    label="worker-prime",
                    seq_len=args.prime_seq_len or args.seq_len,
                    batch=args.batch,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    compact=args.compact,
                    loops=1,
                )
                ready.set()
                go.wait()
                _run_cudnn_attention(torch, label="worker-after-slang", **common)
            except BaseException as exc:
                errors.append(exc)
                ready.set()

        thread = threading.Thread(target=worker, name="cudnn-primer")
        thread.start()
        ready.wait()
        slang_objects = _setup_slang(
            args.slang,
            width=args.window_width,
            height=args.window_height,
        )
        go.set()
        thread.join()
        if errors:
            raise errors[0]
        # Keep objects live until the end of the script.
        print(f"[done] kept slang objects: {bool(slang_objects.module)}", flush=True)
        return

    if args.prime_seq_len is not None:
        _run_cudnn_attention(
            torch,
            label="main-prime",
            seq_len=args.prime_seq_len,
            batch=args.batch,
            heads=args.heads,
            head_dim=args.head_dim,
            compact=args.compact,
            loops=1,
        )

    slang_objects = _setup_slang(
        args.slang,
        width=args.window_width,
        height=args.window_height,
    )

    if args.thread_mode == "worker-after-slang":
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                _run_cudnn_attention(torch, label="worker-after-slang", **common)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, name="cudnn-after-slang")
        thread.start()
        thread.join()
        if errors:
            raise errors[0]
    else:
        _run_cudnn_attention(torch, label="main-after-slang", **common)

    print(f"[done] kept slang objects: {bool(slang_objects.module)}", flush=True)


if __name__ == "__main__":
    main()
