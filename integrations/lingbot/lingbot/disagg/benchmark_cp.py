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

"""LingBot single-session benchmark with one context-parallel DiT group."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.distributed import ProcessGroup

from flashdreams.infra.config import derive_config
from flashdreams.infra.transfer import (
    TensorBundle,
    TransferStats,
    describe_tensor_bundle,
)
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
    parser.add_argument("--cp-ranks", type=int, default=6)
    parser.add_argument(
        "--cp-method",
        choices=("ring", "ulysses"),
        default="ring",
        help="DiT attention collective; CP6 requires ring for the 40-head model.",
    )
    parser.add_argument("--rdma-device", default=None)
    parser.add_argument("--bandwidth-probe-mib", type=int, default=256)
    parser.add_argument("--bandwidth-probe-iters", type=int, default=8)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Probe Mooncake stage edges and DiT NCCL collectives without weights.",
    )
    parser.add_argument("--baseline-json", type=Path, default=_DEFAULT_BASELINE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_disagg_cp6"),
    )
    return parser.parse_args()


def _read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline benchmark not found: {path}")
    return json.loads(path.read_text())


def _create_cp_group(cp_ranks: tuple[int, ...]) -> ProcessGroup | None:
    group = torch.distributed.new_group(
        ranks=list(cp_ranks),
        backend="nccl",
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    if torch.distributed.get_rank() not in cp_ranks:
        return None
    assert isinstance(group, ProcessGroup)
    return group


def _empty_bundle_like_descriptors(
    descriptors: Any,
    *,
    device: torch.device,
) -> TensorBundle:
    return {
        descriptor.name: torch.empty(
            descriptor.shape,
            dtype=descriptor.dtype,
            device=device,
        ).contiguous()
        for descriptor in descriptors
    }


def _fanout_bundle_to_cp(
    *,
    source_bundle: TensorBundle | None,
    source: int,
    cp_ranks: tuple[int, ...],
    cp_group: ProcessGroup | None,
    device: torch.device,
) -> tuple[TensorBundle | None, float | None]:
    """Broadcast one leader-owned full bundle across the DiT CP subgroup."""
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
    if rank not in cp_ranks:
        return None, None
    assert cp_group is not None
    if rank == source:
        assert source_bundle is not None
        local_bundle = source_bundle
    else:
        local_bundle = _empty_bundle_like_descriptors(descriptors, device=device)
    for descriptor in descriptors:
        torch.distributed.broadcast(
            local_bundle[descriptor.name],
            src=source,
            group=cp_group,
        )
    torch.cuda.synchronize(device)
    return local_bundle, (time.perf_counter() - started) * 1000.0


def _max_elapsed_ms(elapsed_ms: float, cp_group: ProcessGroup) -> float:
    elapsed = torch.tensor(elapsed_ms, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        elapsed,
        op=torch.distributed.ReduceOp.MAX,
        group=cp_group,
    )
    return float(elapsed.item())


def _time_cp_collective(
    call: Any,
    *,
    cp_group: ProcessGroup,
) -> float:
    torch.distributed.barrier(group=cp_group)
    torch.cuda.synchronize()
    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    return _max_elapsed_ms((time.perf_counter() - started) * 1000.0, cp_group)


def _cp_collective_probe(
    *,
    cp_ranks: tuple[int, ...],
    cp_group: ProcessGroup | None,
    size_mib: int,
    iterations: int,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    """Measure CP broadcast and all-gather effective per-rank bandwidth."""
    rank = torch.distributed.get_rank()
    leader = cp_ranks[0]
    leader_samples: dict[str, list[dict[str, float]]] | None = None
    if rank in cp_ranks:
        assert cp_group is not None
        payload_bytes = size_mib * 2**20
        broadcast_buffer = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
        shard_bytes = payload_bytes // len(cp_ranks)
        shard = torch.empty(shard_bytes, dtype=torch.uint8, device=device)
        gathered = torch.empty(
            shard_bytes * len(cp_ranks),
            dtype=torch.uint8,
            device=device,
        )

        _time_cp_collective(
            lambda: torch.distributed.broadcast(
                broadcast_buffer,
                src=leader,
                group=cp_group,
            ),
            cp_group=cp_group,
        )
        _time_cp_collective(
            lambda: torch.distributed.all_gather_into_tensor(
                gathered,
                shard,
                group=cp_group,
            ),
            cp_group=cp_group,
        )

        broadcast_samples = []
        all_gather_samples = []
        for _ in range(iterations):
            broadcast_ms = _time_cp_collective(
                lambda: torch.distributed.broadcast(
                    broadcast_buffer,
                    src=leader,
                    group=cp_group,
                ),
                cp_group=cp_group,
            )
            all_gather_ms = _time_cp_collective(
                lambda: torch.distributed.all_gather_into_tensor(
                    gathered,
                    shard,
                    group=cp_group,
                ),
                cp_group=cp_group,
            )
            broadcast_samples.append(
                {
                    "payload_bytes": float(payload_bytes),
                    "transfer_ms": broadcast_ms,
                    "bandwidth_gbps": payload_bytes / (broadcast_ms / 1000.0) / 1e9,
                }
            )
            all_gather_samples.append(
                {
                    "payload_bytes": float(gathered.numel()),
                    "transfer_ms": all_gather_ms,
                    "bandwidth_gbps": gathered.numel() / (all_gather_ms / 1000.0) / 1e9,
                }
            )
        if rank == leader:
            leader_samples = {
                "broadcast": broadcast_samples,
                "all_gather": all_gather_samples,
            }
    return _broadcast_object(leader_samples, source=leader)


def _shutdown(
    transport: Any,
    *,
    cp_group: ProcessGroup | None,
    device: torch.device,
) -> None:
    """Drain the NCCL subgroup before destroying it and the control group."""
    if cp_group is not None:
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=cp_group)
    torch.distributed.barrier()
    transport.close()
    if cp_group is not None:
        torch.distributed.destroy_process_group(cp_group)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _summarize(
    *,
    records: list[dict[str, Any]],
    mooncake_probe: dict[str, list[TransferStats]],
    cp_probe: dict[str, list[dict[str, float]]],
    baseline: dict[str, Any],
    cp_size: int,
) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    frame_count = sum(record["output_frames"] for record in measured)
    elapsed_s = sum(record["end_to_end_ms"] for record in measured) / 1000.0
    baseline_summary = baseline["summary"]
    baseline_latency = baseline_summary["latency_ms"]["median"]
    baseline_dit = (
        baseline_summary["dit_ms"]["median"] + baseline_summary["finalize_ms"]["median"]
    )
    aggregate_fps = frame_count / elapsed_s
    cp_critical = [
        max(worker["dit_ms"] + worker["finalize_ms"] for worker in record["cp_workers"])
        for record in measured
    ]
    return {
        "fps": aggregate_fps,
        "latency_ms": _metric_summary([record["end_to_end_ms"] for record in measured]),
        "latency_speedup": baseline_latency
        / statistics.median([record["end_to_end_ms"] for record in measured]),
        "fps_speedup": aggregate_fps / baseline_summary["fps"],
        "encoder_ms": _metric_summary([record["encoder_ms"] for record in measured]),
        "encoder_to_cp_leader": {
            "payload_mib": measured[0]["encoder_to_cp_leader"]["payload_bytes"] / 2**20,
            "copy_ms": _metric_summary(
                [record["encoder_to_cp_leader"]["transfer_ms"] for record in measured]
            ),
            "handoff_ms": _metric_summary(
                [record["encoder_to_cp_leader_handoff_ms"] for record in measured]
            ),
        },
        "cp_input_fanout_ms": _metric_summary(
            [record["cp_input_fanout_ms"] for record in measured]
        ),
        "dit_ms": _metric_summary(
            [
                max(worker["dit_ms"] for worker in record["cp_workers"])
                for record in measured
            ]
        ),
        "finalize_ms": _metric_summary(
            [
                max(worker["finalize_ms"] for worker in record["cp_workers"])
                for record in measured
            ]
        ),
        "dit_critical_path_ms": _metric_summary(cp_critical),
        "dit_speedup": baseline_dit / statistics.median(cp_critical),
        "cp_efficiency": (baseline_dit / statistics.median(cp_critical)) / cp_size,
        "cp_leader_to_decoder": {
            "payload_mib": measured[0]["cp_leader_to_decoder"]["payload_bytes"] / 2**20,
            "copy_ms": _metric_summary(
                [record["cp_leader_to_decoder"]["transfer_ms"] for record in measured]
            ),
            "handoff_ms": _metric_summary(
                [record["cp_leader_to_decoder_handoff_ms"] for record in measured]
            ),
        },
        "decoder_ms": _metric_summary([record["decoder_ms"] for record in measured]),
        "mooncake_probe_gbps": {
            edge: _metric_summary([sample.bandwidth_gbps for sample in samples])
            for edge, samples in mooncake_probe.items()
        },
        "cp_probe_gbps": {
            collective: _metric_summary(
                [sample["bandwidth_gbps"] for sample in samples]
            )
            for collective, samples in cp_probe.items()
        },
        "baseline": {
            "fps": baseline_summary["fps"],
            "latency_ms": baseline_latency,
            "dit_critical_path_ms": baseline_dit,
            "topology": "1 encoder : 1 DiT : 1 decoder",
        },
    }


def _write_report(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    mooncake_probe: dict[str, list[TransferStats]],
    cp_probe: dict[str, list[dict[str, float]]],
    baseline: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    summary = _summarize(
        records=records,
        mooncake_probe=mooncake_probe,
        cp_probe=cp_probe,
        baseline=baseline,
        cp_size=args.cp_ranks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "summary": summary,
                "records": records,
                "mooncake_probe": {
                    edge: [vars(sample) for sample in samples]
                    for edge, samples in mooncake_probe.items()
                },
                "cp_probe": cp_probe,
            },
            indent=2,
        )
        + "\n"
    )
    peak_memory = environment["peak_memory_gib_by_rank"]
    cp_memory = peak_memory[1:-1]
    markdown = f"""# LingBot CP{args.cp_ranks} single-session disaggregation benchmark

## Result

Topology: **1 encoder : 1 DiT group with CP{args.cp_ranks} ({args.cp_method}) : 1 decoder**.
The {args.cp_ranks} DiT ranks cooperate on one autoregressive session.

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | {summary["latency_ms"]["median"]:.2f} ms | {summary["latency_ms"]["p90"]:.2f} ms |
| Encoder compute | {summary["encoder_ms"]["median"]:.2f} ms | {summary["encoder_ms"]["p90"]:.2f} ms |
| Encoder → CP leader handoff | {summary["encoder_to_cp_leader"]["handoff_ms"]["median"]:.2f} ms | {summary["encoder_to_cp_leader"]["handoff_ms"]["p90"]:.2f} ms |
| CP input fanout | {summary["cp_input_fanout_ms"]["median"]:.2f} ms | {summary["cp_input_fanout_ms"]["p90"]:.2f} ms |
| CP DiT critical path | {summary["dit_critical_path_ms"]["median"]:.2f} ms | {summary["dit_critical_path_ms"]["p90"]:.2f} ms |
| CP leader → decoder handoff | {summary["cp_leader_to_decoder"]["handoff_ms"]["median"]:.2f} ms | {summary["cp_leader_to_decoder"]["handoff_ms"]["p90"]:.2f} ms |
| Decoder compute | {summary["decoder_ms"]["median"]:.2f} ms | {summary["decoder_ms"]["p90"]:.2f} ms |
| 256 MiB Mooncake probes | {statistics.median([value["median"] for value in summary["mooncake_probe_gbps"].values()]):.2f} GB/s | — |
| 256 MiB-equivalent NCCL broadcast | {summary["cp_probe_gbps"]["broadcast"]["median"]:.2f} GB/s | {summary["cp_probe_gbps"]["broadcast"]["p90"]:.2f} GB/s |
| 256 MiB-equivalent NCCL all-gather | {summary["cp_probe_gbps"]["all_gather"]["median"]:.2f} GB/s | {summary["cp_probe_gbps"]["all_gather"]["p90"]:.2f} GB/s |

- Single-session throughput: **{summary["fps"]:.2f} generated FPS**
- Latency speedup versus tracked CP1 baseline: **{summary["latency_speedup"]:.2f}×**
- DiT critical-path speedup: **{summary["dit_speedup"]:.2f}×**
- CP scaling efficiency: **{summary["cp_efficiency"] * 100.0:.1f}%**

The headline excludes {args.warmup_blocks} warmup blocks and measures
{args.measured_blocks} blocks. It accelerates one session; it does not represent
independent concurrent sessions.

## Peak allocated memory

| Role | Peak |
| --- | ---: |
| Encoder | {peak_memory[0]:.2f} GiB |
| CP DiT ranks | {min(cp_memory):.2f}–{max(cp_memory):.2f} GiB each |
| Decoder | {peak_memory[-1]:.2f} GiB |

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
    """Run one encoder, one CP-sharded DiT session, and one decoder."""
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = args.cp_ranks + 2
    if world_size != expected_world_size:
        raise ValueError(
            f"Launch with {expected_world_size} processes for one encoder, "
            f"CP{args.cp_ranks} DiT, and one decoder; got {world_size}."
        )
    if args.cp_ranks < 2:
        raise ValueError("cp-ranks must be at least two.")
    if args.warmup_blocks < 0 or args.measured_blocks <= 0:
        raise ValueError("warmup-blocks must be >= 0 and measured-blocks must be > 0.")

    encoder_rank = 0
    cp_ranks = tuple(range(1, 1 + args.cp_ranks))
    cp_leader = cp_ranks[0]
    decoder_rank = world_size - 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    base_config = PIPELINE_CONFIGS[args.model]

    encoder_stage = (
        LingbotEncoderStage(base_config).to(device).eval()
        if rank == encoder_rank and not args.transport_only
        else None
    )
    dit_stage = None
    if rank in cp_ranks and not args.transport_only:
        base_seed = base_config.diffusion_model.seed
        local_cp_rank = rank - cp_leader
        config = derive_config(
            base_config,
            diffusion_model={
                "seed": None if base_seed is None else base_seed + local_cp_rank,
                "transformer": {"network": {"cp_method": args.cp_method}},
            },
        )
        dit_stage = LingbotDiTStage(config).to(device).eval()
    decoder_stage = (
        LingbotDecoderStage(base_config).to(device).eval()
        if rank == decoder_rank and not args.transport_only
        else None
    )
    encoder_inputs = (
        _load_encoder_inputs(args, device=device) if encoder_stage is not None else None
    )

    torch.distributed.init_process_group("gloo")
    cp_group = _create_cp_group(cp_ranks)
    if dit_stage is not None:
        assert cp_group is not None
        dit_stage.set_context_parallel_group(cp_group)

    from flashdreams.infra.transfer import MooncakeTensorTransport

    transport = MooncakeTensorTransport(device_name=args.rdma_device)
    mooncake_probe = {
        "encoder_to_cp_leader": _bandwidth_probe(
            transport,
            source=encoder_rank,
            destination=cp_leader,
            size_mib=args.bandwidth_probe_mib,
            iterations=args.bandwidth_probe_iters,
            device=device,
        ),
        "cp_leader_to_decoder": _bandwidth_probe(
            transport,
            source=cp_leader,
            destination=decoder_rank,
            size_mib=args.bandwidth_probe_mib,
            iterations=args.bandwidth_probe_iters,
            device=device,
        ),
    }
    cp_probe = _cp_collective_probe(
        cp_ranks=cp_ranks,
        cp_group=cp_group,
        size_mib=args.bandwidth_probe_mib,
        iterations=args.bandwidth_probe_iters,
        device=device,
    )
    if args.transport_only:
        if rank == encoder_rank:
            print(
                json.dumps(
                    {
                        "mooncake_gbps": {
                            edge: _metric_summary(
                                [sample.bandwidth_gbps for sample in samples]
                            )
                            for edge, samples in mooncake_probe.items()
                        },
                        "cp_collective_gbps": {
                            collective: _metric_summary(
                                [sample["bandwidth_gbps"] for sample in samples]
                            )
                            for collective, samples in cp_probe.items()
                        },
                    },
                    indent=2,
                )
            )
        _shutdown(transport, cp_group=cp_group, device=device)
        return

    conditioning_bundle = None
    encoder_cache = None
    height_width = None
    prompt = None
    image = None
    intrinsics = None
    poses = None
    world_scale = None
    if encoder_stage is not None and encoder_inputs is not None:
        prompt, image, intrinsics, poses, world_scale = encoder_inputs
        encoder_cache, conditioning = encoder_stage.initialize_cache(
            text=[prompt],
            image=image,
        )
        conditioning_bundle = conditioning_to_bundle(conditioning)
        height_width = (conditioning.height, conditioning.width)
    height_width = _broadcast_object(height_width, source=encoder_rank)
    received_context, _, _ = _transfer_bundle(
        transport,
        source=encoder_rank,
        destination=cp_leader,
        source_bundle=conditioning_bundle,
        device=device,
    )
    cp_context, _ = _fanout_bundle_to_cp(
        source_bundle=received_context,
        source=cp_leader,
        cp_ranks=cp_ranks,
        cp_group=cp_group,
        device=device,
    )
    torch.distributed.barrier()

    dit_cache = None
    if dit_stage is not None and cp_context is not None:
        conditioning = conditioning_from_bundle(
            cp_context,
            height=height_width[0],
            width=height_width[1],
        )
        dit_cache = dit_stage.initialize_cache(conditioning)
    decoder_cache = (
        decoder_stage.initialize_cache() if decoder_stage is not None else None
    )

    total_blocks = args.warmup_blocks + args.measured_blocks
    records: list[dict[str, Any]] = []
    frame_start = 0
    for autoregressive_index in range(total_blocks):
        torch.distributed.barrier()
        step_started = time.perf_counter()
        local: dict[str, Any] = {}

        encoded_bundle = None
        if encoder_stage is not None and encoder_cache is not None:
            assert intrinsics is not None
            assert poses is not None
            assert world_scale is not None
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
            source=encoder_rank,
            destination=cp_leader,
            source_bundle=encoded_bundle,
            device=device,
        )
        if rank == encoder_rank:
            local["encoder_to_cp_leader"] = dict(vars(encoder_transfer))
            local["encoder_to_cp_leader_handoff_ms"] = encoder_handoff_ms
        cp_encoded, fanout_ms = _fanout_bundle_to_cp(
            source_bundle=received_encoded,
            source=cp_leader,
            cp_ranks=cp_ranks,
            cp_group=cp_group,
            device=device,
        )
        if rank in cp_ranks:
            assert fanout_ms is not None
            local["cp_input_fanout_ms"] = fanout_ms
        torch.distributed.barrier()

        clean_bundle = None
        if dit_stage is not None and dit_cache is not None and cp_encoded is not None:
            encoded = encoder_output_from_bundle(cp_encoded)
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
            local["cp_rank"] = rank - cp_leader
            if rank == cp_leader:
                clean_bundle = {"clean_latent": clean_latent.contiguous()}
                transport.unregister(cp_encoded)

        received_clean, decoder_transfer, decoder_handoff_ms = _transfer_bundle(
            transport,
            source=cp_leader,
            destination=decoder_rank,
            source_bundle=clean_bundle,
            device=device,
        )
        if rank == encoder_rank:
            local["cp_leader_to_decoder"] = dict(vars(decoder_transfer))
            local["cp_leader_to_decoder_handoff_ms"] = decoder_handoff_ms

        if decoder_stage is not None and decoder_cache is not None:
            assert received_clean is not None
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
        if rank == encoder_rank:
            local["end_to_end_ms"] = (time.perf_counter() - step_started) * 1000.0
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        if rank == encoder_rank:
            encoder_record = gathered[encoder_rank]
            decoder_record = gathered[decoder_rank]
            assert encoder_record is not None
            assert decoder_record is not None
            cp_workers = []
            for cp_rank in cp_ranks:
                worker = gathered[cp_rank]
                assert worker is not None
                cp_workers.append(worker)
            records.append(
                {
                    "autoregressive_index": autoregressive_index,
                    "warmup": autoregressive_index < args.warmup_blocks,
                    **encoder_record,
                    "cp_input_fanout_ms": max(
                        worker["cp_input_fanout_ms"] for worker in cp_workers
                    ),
                    "cp_workers": cp_workers,
                    "decoder_ms": decoder_record["decoder_ms"],
                    "output_frames": decoder_record["output_frames"],
                }
            )

    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
    peak_memory_by_rank: list[float | None] = [None] * world_size
    torch.distributed.all_gather_object(peak_memory_by_rank, peak_memory)
    if rank == encoder_rank:
        environment = _environment(
            args,
            prompt=prompt,
            world_size=world_size,
            module_name="lingbot.disagg.benchmark_cp",
        )
        environment["allocation"] = {
            "encoder": [encoder_rank],
            "dit_cp_group": list(cp_ranks),
            "decoder": [decoder_rank],
            "cp_size": args.cp_ranks,
            "cp_method": args.cp_method,
        }
        environment["noise_seed_by_cp_rank"] = [
            None
            if base_config.diffusion_model.seed is None
            else base_config.diffusion_model.seed + cp_rank
            for cp_rank in range(args.cp_ranks)
        ]
        environment["peak_memory_gib_by_rank"] = peak_memory_by_rank
        _write_report(
            args,
            records=records,
            mooncake_probe=mooncake_probe,
            cp_probe=cp_probe,
            baseline=_read_baseline(args.baseline_json),
            environment=environment,
        )

    _shutdown(transport, cp_group=cp_group, device=device)


if __name__ == "__main__":
    main()
