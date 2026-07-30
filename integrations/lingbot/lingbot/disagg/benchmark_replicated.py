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

"""Eight-GPU LingBot benchmark with replicated session-affine DiT workers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from flashdreams.infra.transfer import TransferStats

from lingbot.config import PIPELINE_CONFIGS
from lingbot.disagg.benchmark import (
    _bandwidth_probe,
    _broadcast_object,
    _environment,
    _load_encoder_inputs,
    _metric_summary,
    _timed_cuda,
    _transfer_bundle,
)
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

_DEFAULT_BASELINE = Path(
    "integrations/lingbot/docs/benchmark_h100_3stage/benchmark.json"
)
"""Tracked 1 encoder : 1 DiT : 1 decoder baseline."""


@dataclass(frozen=True, kw_only=True)
class StageAllocation:
    """Replica counts derived from measured per-session service time."""

    encoder_replicas: int
    """Number of encoder workers."""

    dit_replicas: int
    """Number of session-affine DiT workers."""

    decoder_replicas: int
    """Number of decoder workers."""

    @property
    def total_gpus(self) -> int:
        """Return the total number of stage GPUs."""
        return self.encoder_replicas + self.dit_replicas + self.decoder_replicas


def allocate_stage_replicas(
    *,
    total_gpus: int,
    encoder_service_ms: float,
    dit_service_ms: float,
    decoder_service_ms: float,
) -> StageAllocation:
    """Greedily allocate extra GPUs to the highest service-time-per-replica stage."""
    if total_gpus < 3:
        raise ValueError("At least three GPUs are required for stage disaggregation.")
    service_ms = {
        "encoder": encoder_service_ms,
        "dit": dit_service_ms,
        "decoder": decoder_service_ms,
    }
    if any(value <= 0.0 for value in service_ms.values()):
        raise ValueError("Every stage service time must be positive.")

    replicas = {"encoder": 1, "dit": 1, "decoder": 1}
    for _ in range(total_gpus - 3):
        bottleneck = max(
            service_ms,
            key=lambda stage: service_ms[stage] / replicas[stage],
        )
        replicas[bottleneck] += 1
    return StageAllocation(
        encoder_replicas=replicas["encoder"],
        dit_replicas=replicas["dit"],
        decoder_replicas=replicas["decoder"],
    )


def allocation_from_baseline(
    baseline: dict[str, Any],
    *,
    total_gpus: int,
) -> StageAllocation:
    """Derive a stage allocation from a three-stage benchmark document."""
    summary = baseline["summary"]
    encoder_service_ms = (
        summary["encoder_ms"]["median"]
        + summary["encoder_to_dit"]["handoff_ms"]["median"]
    )
    dit_service_ms = summary["dit_ms"]["median"] + summary["finalize_ms"]["median"]
    decoder_service_ms = (
        summary["decoder_ms"]["median"]
        + summary["dit_to_decoder"]["handoff_ms"]["median"]
    )
    return allocate_stage_replicas(
        total_gpus=total_gpus,
        encoder_service_ms=encoder_service_ms,
        dit_service_ms=dit_service_ms,
        decoder_service_ms=decoder_service_ms,
    )


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
    parser.add_argument("--dit-replicas", type=int, default=6)
    parser.add_argument("--rdma-device", default=None)
    parser.add_argument("--bandwidth-probe-mib", type=int, default=256)
    parser.add_argument("--bandwidth-probe-iters", type=int, default=8)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Probe every stage edge without loading model weights.",
    )
    parser.add_argument("--baseline-json", type=Path, default=_DEFAULT_BASELINE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_disagg_1e6d1d"),
    )
    return parser.parse_args()


def _read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline benchmark not found: {path}")
    return json.loads(path.read_text())


def _probe_edges(
    transport: Any,
    *,
    encoder_rank: int,
    dit_ranks: tuple[int, ...],
    decoder_rank: int,
    size_mib: int,
    iterations: int,
    device: torch.device,
) -> dict[str, list[TransferStats]]:
    probes: dict[str, list[TransferStats]] = {}
    for dit_rank in dit_ranks:
        probes[f"encoder_to_dit_{dit_rank}"] = _bandwidth_probe(
            transport,
            source=encoder_rank,
            destination=dit_rank,
            size_mib=size_mib,
            iterations=iterations,
            device=device,
        )
    for dit_rank in dit_ranks:
        probes[f"dit_{dit_rank}_to_decoder"] = _bandwidth_probe(
            transport,
            source=dit_rank,
            destination=decoder_rank,
            size_mib=size_mib,
            iterations=iterations,
            device=device,
        )
    return probes


def _summarize(
    *,
    records: list[dict[str, Any]],
    probes: dict[str, list[TransferStats]],
    baseline: dict[str, Any],
    dit_replicas: int,
) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    frame_count = sum(record["output_frames"] for record in measured)
    elapsed_s = sum(record["wave_latency_ms"] for record in measured) / 1000.0
    dit_worker_total_ms = [
        worker["dit_ms"] + worker["finalize_ms"]
        for record in measured
        for worker in record["dit_workers"]
    ]
    encoder_transfers = [
        transfer for record in measured for transfer in record["encoder_to_dit"]
    ]
    decoder_transfers = [
        transfer for record in measured for transfer in record["dit_to_decoder"]
    ]
    probe_bandwidth = [
        sample.bandwidth_gbps for samples in probes.values() for sample in samples
    ]
    aggregate_fps = frame_count / elapsed_s
    baseline_fps = baseline["summary"]["fps"]
    baseline_latency_ms = baseline["summary"]["latency_ms"]["median"]
    return {
        "aggregate_fps": aggregate_fps,
        "per_session_fps": aggregate_fps / dit_replicas,
        "throughput_speedup": aggregate_fps / baseline_fps,
        "gpu_normalized_speedup": (aggregate_fps / (dit_replicas + 2))
        / (baseline_fps / 3),
        "wave_latency_ms": _metric_summary(
            [record["wave_latency_ms"] for record in measured]
        ),
        "latency_vs_baseline": statistics.median(
            [record["wave_latency_ms"] for record in measured]
        )
        / baseline_latency_ms,
        "encoder_wave_ms": _metric_summary(
            [record["encoder_wave_ms"] for record in measured]
        ),
        "dit_critical_path_ms": _metric_summary(
            [
                max(
                    worker["dit_ms"] + worker["finalize_ms"]
                    for worker in record["dit_workers"]
                )
                for record in measured
            ]
        ),
        "dit_worker_total_ms": _metric_summary(dit_worker_total_ms),
        "decoder_wave_ms": _metric_summary(
            [record["decoder_wave_ms"] for record in measured]
        ),
        "encoder_to_dit": {
            "payload_mib_each": encoder_transfers[0]["payload_bytes"] / 2**20,
            "copy_ms_each": _metric_summary(
                [transfer["transfer_ms"] for transfer in encoder_transfers]
            ),
            "handoff_ms_each": _metric_summary(
                [transfer["handoff_ms"] for transfer in encoder_transfers]
            ),
            "aggregate_handoff_ms_per_wave": _metric_summary(
                [
                    sum(item["handoff_ms"] for item in record["encoder_to_dit"])
                    for record in measured
                ]
            ),
        },
        "dit_to_decoder": {
            "payload_mib_each": decoder_transfers[0]["payload_bytes"] / 2**20,
            "copy_ms_each": _metric_summary(
                [transfer["transfer_ms"] for transfer in decoder_transfers]
            ),
            "handoff_ms_each": _metric_summary(
                [transfer["handoff_ms"] for transfer in decoder_transfers]
            ),
            "aggregate_handoff_ms_per_wave": _metric_summary(
                [
                    sum(item["handoff_ms"] for item in record["dit_to_decoder"])
                    for record in measured
                ]
            ),
        },
        "bandwidth_probe_gbps": {
            "all_edges": _metric_summary(probe_bandwidth),
            "by_edge": {
                edge: _metric_summary([sample.bandwidth_gbps for sample in samples])
                for edge, samples in probes.items()
            },
        },
        "baseline": {
            "fps": baseline_fps,
            "latency_ms": baseline_latency_ms,
            "topology": "1 encoder : 1 DiT : 1 decoder",
        },
    }


def _write_report(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    probes: dict[str, list[TransferStats]],
    baseline: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    summary = _summarize(
        records=records,
        probes=probes,
        baseline=baseline,
        dit_replicas=args.dit_replicas,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "summary": summary,
                "records": records,
                "bandwidth_probe": {
                    edge: [vars(item) for item in samples]
                    for edge, samples in probes.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    allocation = environment["allocation"]
    peak_memory = environment["peak_memory_gib_by_rank"]
    dit_memory = peak_memory[1:-1]
    markdown = f"""# LingBot replicated-DiT disaggregation benchmark

## Result

Topology: **{allocation["encoder"]} encoder : {allocation["dit"]} DiT : {allocation["decoder"]} decoder**.
Each DiT worker owns one concurrent session and its resident autoregressive KV cache.

| Metric | Median | P90 |
| --- | ---: | ---: |
| {args.dit_replicas}-session wave latency | {summary["wave_latency_ms"]["median"]:.2f} ms | {summary["wave_latency_ms"]["p90"]:.2f} ms |
| Encoder wave | {summary["encoder_wave_ms"]["median"]:.2f} ms | {summary["encoder_wave_ms"]["p90"]:.2f} ms |
| DiT critical path | {summary["dit_critical_path_ms"]["median"]:.2f} ms | {summary["dit_critical_path_ms"]["p90"]:.2f} ms |
| Decoder wave | {summary["decoder_wave_ms"]["median"]:.2f} ms | {summary["decoder_wave_ms"]["p90"]:.2f} ms |
| Encoder → DiT handoff, each | {summary["encoder_to_dit"]["handoff_ms_each"]["median"]:.2f} ms | {summary["encoder_to_dit"]["handoff_ms_each"]["p90"]:.2f} ms |
| DiT → decoder handoff, each | {summary["dit_to_decoder"]["handoff_ms_each"]["median"]:.2f} ms | {summary["dit_to_decoder"]["handoff_ms_each"]["p90"]:.2f} ms |
| 256 MiB RDMA probes, all edges | {summary["bandwidth_probe_gbps"]["all_edges"]["median"]:.2f} GB/s | {summary["bandwidth_probe_gbps"]["all_edges"]["p90"]:.2f} GB/s |

- Aggregate throughput: **{summary["aggregate_fps"]:.2f} generated FPS**
- Per-session throughput: **{summary["per_session_fps"]:.2f} generated FPS**
- Throughput versus tracked 1:1:1 baseline: **{summary["throughput_speedup"]:.2f}×**
- Wave latency versus one-session baseline latency: **{summary["latency_vs_baseline"]:.2f}×**
- GPU-normalized throughput versus the three-GPU baseline: **{summary["gpu_normalized_speedup"]:.2f}×**

The headline excludes {args.warmup_blocks} warmup waves and measures
{args.measured_blocks} waves. It represents {args.dit_replicas} concurrent, session-affine
rollouts, not acceleration of one autoregressive session.

## Peak allocated memory

| Role | Peak |
| --- | ---: |
| Shared encoder | {peak_memory[0]:.2f} GiB |
| DiT workers | {min(dit_memory):.2f}–{max(dit_memory):.2f} GiB each |
| Shared decoder | {peak_memory[-1]:.2f} GiB |

## Reproduction

```bash
{environment["command"]}
```

- Repository revision: `{environment["commit"]}`{" (modified worktree)" if environment["worktree_dirty"] else ""}
- Slurm: job `{environment["slurm_job_id"]}` on `{environment["hostname"]}`
- GPU: `{environment["gpus"][0]}` × {len(environment["gpus"])}
- Model: `{args.model}`
"""
    (args.output_dir / "README.md").write_text(markdown)


def main() -> None:
    """Run one encoder, replicated DiTs, and one decoder on a single node."""
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = args.dit_replicas + 2
    if world_size != expected_world_size:
        raise ValueError(
            f"Launch with {expected_world_size} processes for one encoder, "
            f"{args.dit_replicas} DiTs, and one decoder; got {world_size}."
        )
    if args.dit_replicas < 1:
        raise ValueError("dit-replicas must be positive.")
    if args.warmup_blocks < 0 or args.measured_blocks <= 0:
        raise ValueError("warmup-blocks must be >= 0 and measured-blocks must be > 0.")

    baseline = _read_baseline(args.baseline_json)
    allocation = allocation_from_baseline(baseline, total_gpus=world_size)
    expected = StageAllocation(
        encoder_replicas=1,
        dit_replicas=args.dit_replicas,
        decoder_replicas=1,
    )
    if allocation != expected:
        raise ValueError(
            f"Measured service times recommend {allocation}, but the launch requests "
            f"{expected}."
        )

    encoder_rank = 0
    dit_ranks = tuple(range(1, 1 + args.dit_replicas))
    decoder_rank = world_size - 1
    session_count = len(dit_ranks)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    config = PIPELINE_CONFIGS[args.model]

    if args.transport_only:
        torch.distributed.init_process_group("gloo")
        from flashdreams.infra.transfer import MooncakeTensorTransport

        transport = MooncakeTensorTransport(device_name=args.rdma_device)
        probes = _probe_edges(
            transport,
            encoder_rank=encoder_rank,
            dit_ranks=dit_ranks,
            decoder_rank=decoder_rank,
            size_mib=args.bandwidth_probe_mib,
            iterations=args.bandwidth_probe_iters,
            device=device,
        )
        if rank == encoder_rank:
            print(
                json.dumps(
                    {
                        edge: _metric_summary([item.bandwidth_gbps for item in samples])
                        for edge, samples in probes.items()
                    },
                    indent=2,
                )
            )
        transport.close()
        torch.distributed.destroy_process_group()
        return

    encoder_stage = (
        LingbotEncoderStage(config).to(device).eval() if rank == encoder_rank else None
    )
    dit_stage = LingbotDiTStage(config).to(device).eval() if rank in dit_ranks else None
    decoder_stage = (
        LingbotDecoderStage(config).to(device).eval() if rank == decoder_rank else None
    )
    encoder_inputs = (
        _load_encoder_inputs(args, device=device) if rank == encoder_rank else None
    )

    torch.distributed.init_process_group("gloo")
    from flashdreams.infra.transfer import MooncakeTensorTransport

    transport = MooncakeTensorTransport(device_name=args.rdma_device)

    encoder_caches: list[Any] = []
    conditioning_bundle = None
    height_width = None
    prompt = None
    image = None
    intrinsics = None
    poses = None
    world_scale = None
    if encoder_stage is not None and encoder_inputs is not None:
        prompt, image, intrinsics, poses, world_scale = encoder_inputs
        for _ in range(session_count):
            cache, conditioning = encoder_stage.initialize_cache(
                text=[prompt],
                image=image,
            )
            encoder_caches.append(cache)
            if conditioning_bundle is None:
                conditioning_bundle = conditioning_to_bundle(conditioning)
                height_width = (conditioning.height, conditioning.width)

    height_width = _broadcast_object(height_width, source=encoder_rank)
    dit_cache = None
    for dit_rank in dit_ranks:
        received_context, _, _ = _transfer_bundle(
            transport,
            source=encoder_rank,
            destination=dit_rank,
            source_bundle=conditioning_bundle,
            device=device,
        )
        if rank == dit_rank and dit_stage is not None and received_context is not None:
            conditioning = conditioning_from_bundle(
                received_context,
                height=height_width[0],
                width=height_width[1],
            )
            dit_cache = dit_stage.initialize_cache(conditioning)
            transport.unregister(received_context)

    decoder_caches = (
        [decoder_stage.initialize_cache() for _ in range(session_count)]
        if decoder_stage is not None
        else []
    )
    probes = _probe_edges(
        transport,
        encoder_rank=encoder_rank,
        dit_ranks=dit_ranks,
        decoder_rank=decoder_rank,
        size_mib=args.bandwidth_probe_mib,
        iterations=args.bandwidth_probe_iters,
        device=device,
    )

    total_blocks = args.warmup_blocks + args.measured_blocks
    frame_starts = [0] * session_count
    records: list[dict[str, Any]] = []
    for autoregressive_index in range(total_blocks):
        torch.distributed.barrier()
        wave_started = time.perf_counter()
        local: dict[str, Any] = {}
        encoder_times: list[float] = []
        encoder_transfers: list[dict[str, Any]] = []
        received_encoded = None

        for session_index, dit_rank in enumerate(dit_ranks):
            encoded_bundle = None
            if encoder_stage is not None:
                assert intrinsics is not None
                assert poses is not None
                assert world_scale is not None
                num_input_frames = encoder_stage.get_num_input_frames(
                    autoregressive_index
                )
                frame_start = frame_starts[session_index]
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
                encoded, encoder_ms = _timed_cuda(
                    partial(
                        encoder_stage.encode,
                        autoregressive_index=autoregressive_index,
                        cache=encoder_caches[session_index],
                        input=control,
                    )
                )
                encoded_bundle = encoder_output_to_bundle(encoded)
                encoder_times.append(encoder_ms)
                frame_starts[session_index] = frame_end

            received, stats, handoff_ms = _transfer_bundle(
                transport,
                source=encoder_rank,
                destination=dit_rank,
                source_bundle=encoded_bundle,
                device=device,
            )
            if rank == dit_rank:
                received_encoded = received
            if rank == encoder_rank:
                transfer_record = dict(vars(stats))
                transfer_record["handoff_ms"] = handoff_ms
                encoder_transfers.append(transfer_record)

        clean_bundle = None
        if rank in dit_ranks:
            assert dit_stage is not None
            assert dit_cache is not None
            assert received_encoded is not None
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
            local["session_index"] = dit_ranks.index(rank)
            local["rank"] = rank
            clean_bundle = {"clean_latent": clean_latent.contiguous()}
            transport.unregister(received_encoded)

        torch.distributed.barrier()
        decoder_inputs: list[Any] = []
        decoder_transfers: list[dict[str, Any]] = []
        for dit_rank in dit_ranks:
            received_clean, stats, handoff_ms = _transfer_bundle(
                transport,
                source=dit_rank,
                destination=decoder_rank,
                source_bundle=clean_bundle if rank == dit_rank else None,
                device=device,
            )
            if rank == decoder_rank:
                assert received_clean is not None
                decoder_inputs.append(received_clean)
            if rank == encoder_rank:
                transfer_record = dict(vars(stats))
                transfer_record["handoff_ms"] = handoff_ms
                decoder_transfers.append(transfer_record)

        if decoder_stage is not None:
            decoder_times: list[float] = []
            output_frames = 0
            for session_index, received_clean in enumerate(decoder_inputs):
                decoded, decoder_ms = _timed_cuda(
                    partial(
                        decoder_stage.decode,
                        input=received_clean["clean_latent"],
                        autoregressive_index=autoregressive_index,
                        cache=decoder_caches[session_index],
                    )
                )
                decoder_times.append(decoder_ms)
                output_frames += decoded.shape[-4]
                transport.unregister(received_clean)
            local["decoder_times_ms"] = decoder_times
            local["decoder_wave_ms"] = sum(decoder_times)
            local["output_frames"] = output_frames

        torch.distributed.barrier()
        if rank == encoder_rank:
            local["encoder_times_ms"] = encoder_times
            local["encoder_wave_ms"] = sum(encoder_times)
            local["encoder_to_dit"] = encoder_transfers
            local["dit_to_decoder"] = decoder_transfers
            local["wave_latency_ms"] = (time.perf_counter() - wave_started) * 1000.0

        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        if rank == encoder_rank:
            encoder_record = gathered[encoder_rank]
            decoder_record = gathered[decoder_rank]
            assert encoder_record is not None
            assert decoder_record is not None
            dit_worker_records: list[dict[str, Any]] = []
            for dit_rank in dit_ranks:
                dit_worker_record = gathered[dit_rank]
                assert dit_worker_record is not None
                dit_worker_records.append(dit_worker_record)
            record = {
                "autoregressive_index": autoregressive_index,
                "warmup": autoregressive_index < args.warmup_blocks,
                **encoder_record,
                "decoder_times_ms": decoder_record["decoder_times_ms"],
                "decoder_wave_ms": decoder_record["decoder_wave_ms"],
                "output_frames": decoder_record["output_frames"],
                "dit_workers": dit_worker_records,
            }
            records.append(record)

    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
    peak_memory_by_rank: list[float | None] = [None] * world_size
    torch.distributed.all_gather_object(peak_memory_by_rank, peak_memory)
    if rank == encoder_rank:
        environment = _environment(
            args,
            prompt=prompt,
            world_size=world_size,
            module_name="lingbot.disagg.benchmark_replicated",
        )
        environment["allocation"] = {
            "encoder": 1,
            "dit": args.dit_replicas,
            "decoder": 1,
            "rank_roles": {
                "encoder": [encoder_rank],
                "dit": list(dit_ranks),
                "decoder": [decoder_rank],
            },
        }
        environment["sessions_per_wave"] = session_count
        environment["peak_memory_gib_by_rank"] = peak_memory_by_rank
        _write_report(
            args,
            records=records,
            probes=probes,
            baseline=baseline,
            environment=environment,
        )

    transport.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
