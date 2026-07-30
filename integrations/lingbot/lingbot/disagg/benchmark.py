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

"""Three-GPU LingBot disaggregation benchmark with Mooncake transfers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flashdreams.infra.transfer import (
    MooncakeTensorTransport,
    TensorBundle,
    TensorTransferTicket,
    TransferStats,
    describe_tensor_bundle,
)
from torch import Tensor

from lingbot.config import PIPELINE_CONFIGS
from lingbot.disagg.stages import (
    LingbotDecoderStage,
    LingbotDiTStage,
    LingbotEncoderStage,
    conditioning_from_bundle,
    conditioning_to_bundle,
    encoder_output_from_bundle,
    encoder_output_to_bundle,
)
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.encoder.utils import get_Ks_transformed, preprocess_example_poses
from lingbot.runner import (
    _INTRINSICS_REFERENCE_HEIGHT,
    _INTRINSICS_REFERENCE_WIDTH,
    EXAMPLE_DATA_BASE_URL,
    ensure_example_data_downloaded,
)
from lingbot.transformer import LingbotWorldTransformerConfig

_ENCODER_RANK = 0
"""Rank that owns text/image/VAE/camera encoders."""

_DIT_RANK = 1
"""Rank that owns the scheduler, DiT, and session KV cache."""

_DECODER_RANK = 2
"""Rank that owns the streaming VAE decoder."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(PIPELINE_CONFIGS),
        default="lingbot-world-fast-taehv-window15-sink3",
    )
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--warmup-blocks", type=int, default=6)
    parser.add_argument("--measured-blocks", type=int, default=5)
    parser.add_argument("--pixel-height", type=int, default=464)
    parser.add_argument("--pixel-width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--rdma-device", default=None)
    parser.add_argument("--bandwidth-probe-mib", type=int, default=256)
    parser.add_argument("--bandwidth-probe-iters", type=int, default=8)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Run Mooncake bandwidth probes without loading model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_disagg"),
    )
    return parser.parse_args()


def _broadcast_object(value: Any, *, source: int) -> Any:
    payload = [value if torch.distributed.get_rank() == source else None]
    torch.distributed.broadcast_object_list(payload, src=source)
    return payload[0]


def _timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end)


def _transfer_bundle(
    transport: MooncakeTensorTransport,
    *,
    source: int,
    destination: int,
    source_bundle: TensorBundle | None,
    device: torch.device,
) -> tuple[TensorBundle | None, TransferStats, float]:
    rank = torch.distributed.get_rank()
    started = time.perf_counter()
    source_descriptors = None
    if rank == source:
        assert source_bundle is not None
        source_descriptors = describe_tensor_bundle(source_bundle)
    descriptors = _broadcast_object(
        source_descriptors,
        source=source,
    )
    destination_bundle = (
        transport.allocate(descriptors, device=device) if rank == destination else None
    )
    ticket = _broadcast_object(
        transport.make_ticket(destination_bundle)
        if rank == destination and destination_bundle is not None
        else None,
        source=destination,
    )
    stats = (
        transport.send(source_bundle, ticket)
        if rank == source and source_bundle is not None
        else None
    )
    torch.distributed.barrier()
    if rank == source and source_bundle is not None:
        transport.unregister(source_bundle)
    stats = _broadcast_object(stats, source=source)
    source_handoff_ms = _broadcast_object(
        (time.perf_counter() - started) * 1000.0 if rank == source else None,
        source=source,
    )
    return destination_bundle, stats, source_handoff_ms


def _bandwidth_probe(
    transport: MooncakeTensorTransport,
    *,
    source: int,
    destination: int,
    size_mib: int,
    iterations: int,
    device: torch.device,
) -> list[TransferStats]:
    rank = torch.distributed.get_rank()
    numel = size_mib * 1024 * 1024 // torch.empty((), dtype=torch.uint8).element_size()
    source_bundle = (
        {"probe": torch.empty(numel, dtype=torch.uint8, device=device)}
        if rank == source
        else None
    )
    descriptors = _broadcast_object(
        describe_tensor_bundle(source_bundle) if source_bundle is not None else None,
        source=source,
    )
    destination_bundle = (
        transport.allocate(descriptors, device=device) if rank == destination else None
    )
    ticket: TensorTransferTicket = _broadcast_object(
        transport.make_ticket(destination_bundle)
        if destination_bundle is not None
        else None,
        source=destination,
    )
    sender_stats: list[TransferStats] | None = None
    if source_bundle is not None:
        transport.register(source_bundle)
        transport.send(source_bundle, ticket)  # connection warmup
        sender_stats = [
            transport.send(source_bundle, ticket) for _ in range(iterations)
        ]
    torch.distributed.barrier()
    if source_bundle is not None:
        transport.unregister(source_bundle)
    if destination_bundle is not None:
        transport.unregister(destination_bundle)
    return _broadcast_object(sender_stats, source=source)


def _load_encoder_inputs(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[str, Tensor, Tensor, Tensor, float]:
    from flashdreams.infra.runner_io import load_first_frame_tensor

    example_dir = ensure_example_data_downloaded(
        is_rank_zero=True,
        example_idx=args.example_idx,
    )
    prompt_path = example_dir / "prompt.txt"
    prompt = (
        prompt_path.read_text().splitlines()[0].strip() if prompt_path.exists() else ""
    )
    image = load_first_frame_tensor(
        example_dir / "image.jpg",
        pixel_height=args.pixel_height,
        pixel_width=args.pixel_width,
        device=device,
        dtype=torch.bfloat16,
        interpolation="cubic",
        install_hint="Install the lingbot plugin.",
    )
    intrinsics = torch.from_numpy(np.load(example_dir / "intrinsics.npy")).to(
        device=device,
        dtype=torch.float32,
    )
    intrinsics = get_Ks_transformed(
        intrinsics,
        height_org=_INTRINSICS_REFERENCE_HEIGHT,
        width_org=_INTRINSICS_REFERENCE_WIDTH,
        height_resize=args.pixel_height,
        width_resize=args.pixel_width,
        height_final=args.pixel_height,
        width_final=args.pixel_width,
    )
    poses, world_scale = preprocess_example_poses(np.load(example_dir / "poses.npy"))
    poses_tensor = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
    return prompt, image, intrinsics, poses_tensor, float(world_scale)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def _environment(
    args: argparse.Namespace,
    *,
    prompt: str | None,
    world_size: int = 3,
    module_name: str | None = None,
) -> dict[str, Any]:
    config = PIPELINE_CONFIGS[args.model]
    transformer = config.diffusion_model.transformer
    assert isinstance(transformer, LingbotWorldTransformerConfig)
    scheduler = config.diffusion_model.scheduler
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        worktree_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        worktree_dirty = None
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    try:
        mooncake_version = importlib.metadata.version("mooncake-transfer-engine-cuda13")
    except importlib.metadata.PackageNotFoundError:
        mooncake_version = "unknown"
    if module_name is None:
        module_name = (
            __spec__.name if __spec__ is not None else "lingbot.disagg.benchmark"
        )
    command = shlex.join(
        [
            "uv",
            "run",
            "--package",
            "flashdreams-lingbot",
            "torchrun",
            "--standalone",
            f"--nproc_per_node={world_size}",
            "-m",
            module_name,
            *sys.argv[1:],
        ]
    )
    return {
        "command": command,
        "commit": commit,
        "worktree_dirty": worktree_dirty,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "model": args.model,
        "checkpoint": transformer.checkpoint_path,
        "precision": str(transformer.dtype).removeprefix("torch."),
        "decoder_config": type(config.decoder).__name__,
        "seed": config.diffusion_model.seed,
        "example_index": args.example_idx,
        "example_url": f"{EXAMPLE_DATA_BASE_URL}/{args.example_idx:02d}",
        "prompt": prompt,
        "resolution": [args.pixel_height, args.pixel_width],
        "target_fps": args.fps,
        "latent_frames_per_chunk": transformer.len_t,
        "window_size_t": transformer.window_size_t,
        "sink_size_t": transformer.sink_size_t,
        "guidance_scale": transformer.guidance_scale,
        "num_inference_steps": getattr(scheduler, "num_inference_steps", None),
        "compile_network": transformer.compile_network,
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": args.measured_blocks,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "mooncake": mooncake_version,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "hf_home": os.environ.get("HF_HOME"),
        "gpus": [torch.cuda.get_device_name(index) for index in range(world_size)],
        "rdma_device": args.rdma_device,
    }


def _write_report(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    probe: dict[str, list[TransferStats]],
    environment: dict[str, Any],
) -> None:
    measured = [item for item in records if not item["warmup"]]
    frame_count = sum(item["output_frames"] for item in measured)
    total_seconds = sum(item["end_to_end_ms"] for item in measured) / 1000.0
    summary = {
        "fps": frame_count / total_seconds,
        "latency_ms": _metric_summary([item["end_to_end_ms"] for item in measured]),
        "encoder_ms": _metric_summary([item["encoder_ms"] for item in measured]),
        "dit_ms": _metric_summary([item["dit_ms"] for item in measured]),
        "finalize_ms": _metric_summary([item["finalize_ms"] for item in measured]),
        "decoder_ms": _metric_summary([item["decoder_ms"] for item in measured]),
        "encoder_to_dit": {
            "payload_mib": measured[0]["encoder_to_dit"]["payload_bytes"] / 2**20,
            "transfer_ms": _metric_summary(
                [item["encoder_to_dit"]["transfer_ms"] for item in measured]
            ),
            "bandwidth_gbps": _metric_summary(
                [item["encoder_to_dit"]["bandwidth_gbps"] for item in measured]
            ),
            "handoff_ms": _metric_summary(
                [item["encoder_to_dit_handoff_ms"] for item in measured]
            ),
        },
        "dit_to_decoder": {
            "payload_mib": measured[0]["dit_to_decoder"]["payload_bytes"] / 2**20,
            "transfer_ms": _metric_summary(
                [item["dit_to_decoder"]["transfer_ms"] for item in measured]
            ),
            "bandwidth_gbps": _metric_summary(
                [item["dit_to_decoder"]["bandwidth_gbps"] for item in measured]
            ),
            "handoff_ms": _metric_summary(
                [item["dit_to_decoder_handoff_ms"] for item in measured]
            ),
        },
        "bandwidth_probe_gbps": {
            edge: _metric_summary([item.bandwidth_gbps for item in samples])
            for edge, samples in probe.items()
        },
    }
    median_latency_ms = summary["latency_ms"]["median"]
    summary["transfer_overhead_percent"] = {
        "synchronous_copy": 100.0
        * (
            summary["encoder_to_dit"]["transfer_ms"]["median"]
            + summary["dit_to_decoder"]["transfer_ms"]["median"]
        )
        / median_latency_ms,
        "full_handoff": 100.0
        * (
            summary["encoder_to_dit"]["handoff_ms"]["median"]
            + summary["dit_to_decoder"]["handoff_ms"]["median"]
        )
        / median_latency_ms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "benchmark.json"
    raw_path.write_text(
        json.dumps(
            {
                "environment": environment,
                "summary": summary,
                "records": records,
                "bandwidth_probe": {
                    edge: [vars(item) for item in samples]
                    for edge, samples in probe.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    revision_label = (
        "Repository base" if environment.get("worktree_dirty") else "Commit"
    )
    dirty_note = (
        "; the benchmark ran from a modified worktree"
        if environment.get("worktree_dirty")
        else ""
    )
    markdown = f"""# LingBot three-stage disaggregation benchmark

For the full tested configuration, methodology, findings, Slurm setup, and
limitations, see the
[experiment report](../disaggregated_inference_experiment.md).

## Result

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | {summary["latency_ms"]["median"]:.2f} ms | {summary["latency_ms"]["p90"]:.2f} ms |
| Encoder compute | {summary["encoder_ms"]["median"]:.2f} ms | {summary["encoder_ms"]["p90"]:.2f} ms |
| DiT denoise | {summary["dit_ms"]["median"]:.2f} ms | {summary["dit_ms"]["p90"]:.2f} ms |
| DiT cache finalize | {summary["finalize_ms"]["median"]:.2f} ms | {summary["finalize_ms"]["p90"]:.2f} ms |
| Decoder compute | {summary["decoder_ms"]["median"]:.2f} ms | {summary["decoder_ms"]["p90"]:.2f} ms |
| Encoder → DiT handoff | {summary["encoder_to_dit"]["handoff_ms"]["median"]:.2f} ms | {summary["encoder_to_dit"]["handoff_ms"]["p90"]:.2f} ms |
| DiT → decoder handoff | {summary["dit_to_decoder"]["handoff_ms"]["median"]:.2f} ms | {summary["dit_to_decoder"]["handoff_ms"]["p90"]:.2f} ms |
| Encoder → DiT payload bandwidth | {summary["encoder_to_dit"]["bandwidth_gbps"]["median"]:.2f} GB/s | {summary["encoder_to_dit"]["bandwidth_gbps"]["p90"]:.2f} GB/s |
| DiT → decoder payload bandwidth | {summary["dit_to_decoder"]["bandwidth_gbps"]["median"]:.2f} GB/s | {summary["dit_to_decoder"]["bandwidth_gbps"]["p90"]:.2f} GB/s |
| 256 MiB encoder → DiT probe | {summary["bandwidth_probe_gbps"]["encoder_to_dit"]["median"]:.2f} GB/s | {summary["bandwidth_probe_gbps"]["encoder_to_dit"]["p90"]:.2f} GB/s |
| 256 MiB DiT → decoder probe | {summary["bandwidth_probe_gbps"]["dit_to_decoder"]["median"]:.2f} GB/s | {summary["bandwidth_probe_gbps"]["dit_to_decoder"]["p90"]:.2f} GB/s |

Steady-state throughput: **{summary["fps"]:.2f} generated FPS**.

The headline excludes {args.warmup_blocks} warmup block(s). Mooncake was
configured with the RDMA protocol. Effective payload bandwidth includes the
synchronous transfer call but excludes receiver allocation and control-plane
ticket exchange; handoff timing in `benchmark.json` includes those costs.
The real payloads were {summary["encoder_to_dit"]["payload_mib"]:.2f} MiB
(encoder → DiT) and {summary["dit_to_decoder"]["payload_mib"]:.2f} MiB
(DiT → decoder). The two synchronous copy calls account for
{summary["transfer_overhead_percent"]["synchronous_copy"]:.2f}% of median
chunk latency; complete allocation, metadata, synchronization, and copy
handoffs account for {summary["transfer_overhead_percent"]["full_handoff"]:.2f}%.

## Reproduction

```bash
{environment["command"]}
```

- {revision_label}: `{environment["commit"]}`{dirty_note}
- Slurm: job `{environment["slurm_job_id"]}` on `{environment["hostname"]}`
- GPU: `{environment["gpus"][0]}` × 3
- Resolution: `{args.pixel_width}x{args.pixel_height}`
- Model: `{args.model}`
"""
    (args.output_dir / "README.md").write_text(markdown)


def main() -> None:
    """Run the fixed three-rank encoder → DiT → decoder benchmark."""
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 3:
        raise ValueError(
            "Launch with exactly three processes: torchrun --nproc_per_node=3 "
            "-m lingbot.disagg.benchmark ..."
        )
    if args.warmup_blocks < 0 or args.measured_blocks <= 0:
        raise ValueError("warmup-blocks must be >= 0 and measured-blocks must be > 0.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    config = PIPELINE_CONFIGS[args.model]

    if args.transport_only:
        torch.distributed.init_process_group("gloo")
        transport = MooncakeTensorTransport(device_name=args.rdma_device)
        probe = {
            "encoder_to_dit": _bandwidth_probe(
                transport,
                source=_ENCODER_RANK,
                destination=_DIT_RANK,
                size_mib=args.bandwidth_probe_mib,
                iterations=args.bandwidth_probe_iters,
                device=device,
            ),
            "dit_to_decoder": _bandwidth_probe(
                transport,
                source=_DIT_RANK,
                destination=_DECODER_RANK,
                size_mib=args.bandwidth_probe_mib,
                iterations=args.bandwidth_probe_iters,
                device=device,
            ),
        }
        if rank == _ENCODER_RANK:
            print(
                json.dumps(
                    {
                        edge: _metric_summary([item.bandwidth_gbps for item in samples])
                        for edge, samples in probe.items()
                    },
                    indent=2,
                )
            )
        transport.close()
        torch.distributed.destroy_process_group()
        return

    # Build stage-local weights before initializing torch.distributed. LingBot's
    # transformer interprets an initialized process group as context parallelism,
    # while these three ranks are independent pipeline stages.
    encoder_stage = (
        LingbotEncoderStage(config).to(device).eval() if rank == _ENCODER_RANK else None
    )
    dit_stage = LingbotDiTStage(config).to(device).eval() if rank == _DIT_RANK else None
    decoder_stage = (
        LingbotDecoderStage(config).to(device).eval() if rank == _DECODER_RANK else None
    )

    encoder_inputs = (
        _load_encoder_inputs(args, device=device) if rank == _ENCODER_RANK else None
    )
    torch.distributed.init_process_group("gloo")
    transport = MooncakeTensorTransport(device_name=args.rdma_device)

    conditioning_bundle = None
    encoder_cache = None
    height_width = None
    prompt = None
    if encoder_stage is not None and encoder_inputs is not None:
        prompt, image, intrinsics, poses, world_scale = encoder_inputs
        (encoder_cache, conditioning), _ = _timed_cuda(
            lambda: encoder_stage.initialize_cache(text=[prompt], image=image)
        )
        conditioning_bundle = conditioning_to_bundle(conditioning)
        height_width = (conditioning.height, conditioning.width)
    height_width = _broadcast_object(height_width, source=_ENCODER_RANK)
    received_context, _, _ = _transfer_bundle(
        transport,
        source=_ENCODER_RANK,
        destination=_DIT_RANK,
        source_bundle=conditioning_bundle,
        device=device,
    )

    dit_cache = None
    if dit_stage is not None and received_context is not None:
        conditioning = conditioning_from_bundle(
            received_context,
            height=height_width[0],
            width=height_width[1],
        )
        dit_cache = dit_stage.initialize_cache(conditioning)
    decoder_cache = (
        decoder_stage.initialize_cache() if decoder_stage is not None else None
    )

    probe = {
        "encoder_to_dit": _bandwidth_probe(
            transport,
            source=_ENCODER_RANK,
            destination=_DIT_RANK,
            size_mib=args.bandwidth_probe_mib,
            iterations=args.bandwidth_probe_iters,
            device=device,
        ),
        "dit_to_decoder": _bandwidth_probe(
            transport,
            source=_DIT_RANK,
            destination=_DECODER_RANK,
            size_mib=args.bandwidth_probe_mib,
            iterations=args.bandwidth_probe_iters,
            device=device,
        ),
    }

    total_blocks = args.warmup_blocks + args.measured_blocks
    records: list[dict[str, Any]] = []
    frame_start = 0
    for autoregressive_index in range(total_blocks):
        torch.distributed.barrier()
        step_started = time.perf_counter()
        local: dict[str, Any] = {}

        encoded_bundle = None
        if encoder_stage is not None and encoder_cache is not None:
            num_input_frames = encoder_stage.get_num_input_frames(autoregressive_index)
            frame_end = frame_start + num_input_frames
            if frame_end > poses.shape[0]:
                raise RuntimeError(
                    f"Example camera trajectory ended at frame {poses.shape[0]} "
                    f"before AR block {autoregressive_index}."
                )
            control = CamCtrlInput(
                intrinsics=intrinsics[frame_start:frame_end],
                poses=poses[frame_start:frame_end],
                world_scale=world_scale,
            )
            encoded, local["encoder_ms"] = _timed_cuda(
                partial(
                    encoder_stage.encode,
                    autoregressive_index=autoregressive_index,
                    cache=encoder_cache,
                    input=control,
                )
            )
            encoded_bundle = encoder_output_to_bundle(encoded)
            frame_start = frame_end

        received_encoded, encoder_transfer, encoder_handoff_ms = _transfer_bundle(
            transport,
            source=_ENCODER_RANK,
            destination=_DIT_RANK,
            source_bundle=encoded_bundle,
            device=device,
        )
        local["encoder_to_dit_handoff_ms"] = encoder_handoff_ms
        local["encoder_to_dit"] = vars(encoder_transfer)

        clean_bundle = None
        if dit_stage is not None and dit_cache is not None and received_encoded:
            encoded = encoder_output_from_bundle(received_encoded)
            clean_latent, local["dit_ms"] = _timed_cuda(
                partial(
                    dit_stage.generate,
                    autoregressive_index=autoregressive_index,
                    cache=dit_cache,
                    input=encoded,
                )
            )
            _, local["finalize_ms"] = _timed_cuda(
                partial(
                    dit_stage.finalize,
                    autoregressive_index=autoregressive_index,
                    cache=dit_cache,
                )
            )
            clean_bundle = {"clean_latent": clean_latent.contiguous()}
            transport.unregister(received_encoded)

        received_clean, decoder_transfer, decoder_handoff_ms = _transfer_bundle(
            transport,
            source=_DIT_RANK,
            destination=_DECODER_RANK,
            source_bundle=clean_bundle,
            device=device,
        )
        local["dit_to_decoder_handoff_ms"] = decoder_handoff_ms
        local["dit_to_decoder"] = vars(decoder_transfer)

        if decoder_stage is not None and decoder_cache is not None and received_clean:
            decoded, local["decoder_ms"] = _timed_cuda(
                partial(
                    decoder_stage.decode,
                    input=received_clean["clean_latent"],
                    autoregressive_index=autoregressive_index,
                    cache=decoder_cache,
                )
            )
            local["output_frames"] = decoded.shape[-4]
            transport.unregister(received_clean)

        torch.distributed.barrier()
        local["end_to_end_ms"] = (time.perf_counter() - step_started) * 1000.0
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        if rank == _ENCODER_RANK:
            record: dict[str, Any] = {
                "autoregressive_index": autoregressive_index,
                "warmup": autoregressive_index < args.warmup_blocks,
            }
            for rank_record in gathered:
                assert rank_record is not None
                record.update(rank_record)
            records.append(record)

    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
    peak_memory_by_rank: list[float | None] = [None] * world_size
    torch.distributed.all_gather_object(peak_memory_by_rank, peak_memory)
    if rank == _ENCODER_RANK:
        environment = _environment(args, prompt=prompt)
        environment["peak_memory_gib_by_stage"] = {
            "encoder": peak_memory_by_rank[_ENCODER_RANK],
            "dit": peak_memory_by_rank[_DIT_RANK],
            "decoder": peak_memory_by_rank[_DECODER_RANK],
        }
        _write_report(
            args,
            records=records,
            probe=probe,
            environment=environment,
        )

    transport.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
