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

"""Eight-GPU LingBot benchmark with two-rank pipeline-parallel DiT groups."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.infra.config import derive_config
from flashdreams.infra.transfer import (
    RegisteredTensorPool,
    TensorBundle,
    describe_tensor_bundle,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from lingbot.config import PIPELINE_CONFIGS
from lingbot.disagg.benchmark import (
    _broadcast_object,
    _environment,
    _load_encoder_inputs,
    _metric_summary,
    _timed_cuda,
    _transfer_bundle,
)
from lingbot.disagg.benchmark_replicated import (
    _TransferChannel,
    _create_transport,
    _finish_channel,
    _setup_channel,
    _submit_channel,
)
from lingbot.disagg.stages import (
    LingbotConditioning,
    LingbotDecoderStage,
    LingbotDiTStage,
    LingbotEncoderStage,
    conditioning_from_bundle,
    conditioning_to_bundle,
    encoder_output_from_bundle,
    encoder_output_to_bundle,
)
from lingbot.encoder.camctrl import CamCtrlInput, I2VCamCtrlEmbeddings

_DEFAULT_REPLICATED_BASELINE = Path(
    "integrations/lingbot/docs/benchmark_h100_1io7dit_optimized/summary.json"
)
"""Tracked seven-session, one-GPU-per-DiT comparison."""


@dataclass(frozen=True, kw_only=True)
class PipelineTopology:
    """Fixed I/O, DiT-group, and spare-rank allocation."""

    io_rank: int
    """Rank hosting the shared encoder and decoder."""

    dit_groups: tuple[tuple[int, int], ...]
    """Ordered two-rank DiT pipeline groups."""

    spare_ranks: tuple[int, ...]
    """Ranks intentionally left without model weights."""

    sessions_per_group: int
    """Fixed session microbatch held by each DiT group."""

    @property
    def session_count(self) -> int:
        """Return the number of concurrently generated sessions."""
        return len(self.dit_groups) * self.sessions_per_group

    @property
    def dit_ranks(self) -> tuple[int, ...]:
        """Return all ranks that own a DiT layer partition."""
        return tuple(rank for group in self.dit_groups for rank in group)


def build_pipeline_topology(
    *,
    world_size: int,
    sessions_per_group: int,
) -> PipelineTopology:
    """Build the supported one-I/O plus three-pair topology.

    Args:
        world_size: Number of local ranks in the launch.
        sessions_per_group: Fixed session batch assigned to each DiT group.

    Returns:
        Eight-rank topology with GPU 7 left spare.

    Raises:
        ValueError: The launch is not the required eight-rank layout.
    """
    if world_size != 8:
        raise ValueError(f"Pipeline benchmark requires eight ranks, got {world_size}.")
    if sessions_per_group < 1:
        raise ValueError("sessions_per_group must be positive.")
    return PipelineTopology(
        io_rank=0,
        dit_groups=((1, 2), (3, 4), (5, 6)),
        spare_ranks=(7,),
        sessions_per_group=sessions_per_group,
    )


def stack_conditioning(items: list[LingbotConditioning]) -> LingbotConditioning:
    """Stack per-session conditioning into one fixed DiT microbatch.

    Args:
        items: Session conditioning records with identical spatial layout.

    Returns:
        Batched conditioning for one DiT group.

    Raises:
        ValueError: No items were supplied or optional fields are inconsistent.
    """
    if not items:
        raise ValueError("At least one conditioning record is required.")
    height = items[0].height
    width = items[0].width
    if any(item.height != height or item.width != width for item in items):
        raise ValueError("All microbatched sessions must share one spatial layout.")

    def stack_optional(name: str) -> torch.Tensor | None:
        values = [getattr(item, name) for item in items]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Optional conditioning field {name} is inconsistent.")
        return torch.cat(cast(list[torch.Tensor], values), dim=0)

    return LingbotConditioning(
        height=height,
        width=width,
        text_embeddings=torch.cat([item.text_embeddings for item in items], dim=0),
        negative_text_embeddings=stack_optional("negative_text_embeddings"),
        image_embeddings=stack_optional("image_embeddings"),
    )


def stack_encoder_outputs(
    items: list[I2VCamCtrlEmbeddings],
) -> I2VCamCtrlEmbeddings:
    """Stack per-session encoder outputs into one fixed DiT microbatch.

    Args:
        items: Unpatchified encoder outputs for one DiT group.

    Returns:
        Batched I2V and camera-control payload.

    Raises:
        ValueError: The input list is empty or already patchified.
    """
    if not items:
        raise ValueError("At least one encoder output is required.")
    if any(item._is_patchified or item.i2v._is_patchified for item in items):
        raise ValueError("Pipeline input batching requires unpatchified tensors.")
    return I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.stack([item.i2v.latent for item in items], dim=0),
            mask=torch.stack([item.i2v.mask for item in items], dim=0),
            _is_patchified=False,
        ),
        plucker=torch.stack([item.plucker for item in items], dim=0),
        _is_patchified=False,
    )


def split_conditioning(
    conditioning: LingbotConditioning,
) -> tuple[LingbotConditioning, LingbotConditioning]:
    """Split one batch-two conditioning record into session records.

    Args:
        conditioning: Conditioning whose leading batch dimension is two.

    Returns:
        Two batch-one conditioning records.

    Raises:
        ValueError: A conditioning tensor does not have batch size two.
    """

    def split_optional(name: str) -> tuple[Tensor | None, Tensor | None]:
        value = getattr(conditioning, name)
        if value is None:
            return None, None
        if value.shape[0] != 2:
            raise ValueError(f"Conditioning field {name} must have batch size two.")
        return value.narrow(0, 0, 1), value.narrow(0, 1, 1)

    if conditioning.text_embeddings.shape[0] != 2:
        raise ValueError("Text conditioning must have batch size two.")
    negative = split_optional("negative_text_embeddings")
    image = split_optional("image_embeddings")
    return (
        LingbotConditioning(
            height=conditioning.height,
            width=conditioning.width,
            text_embeddings=conditioning.text_embeddings.narrow(0, 0, 1),
            negative_text_embeddings=negative[0],
            image_embeddings=image[0],
        ),
        LingbotConditioning(
            height=conditioning.height,
            width=conditioning.width,
            text_embeddings=conditioning.text_embeddings.narrow(0, 1, 1),
            negative_text_embeddings=negative[1],
            image_embeddings=image[1],
        ),
    )


def split_encoder_outputs(
    output: I2VCamCtrlEmbeddings,
) -> tuple[I2VCamCtrlEmbeddings, I2VCamCtrlEmbeddings]:
    """Split one batch-two encoder payload into session payloads.

    Args:
        output: Unpatchified encoder payload with leading batch size two.

    Returns:
        Two batch-one payloads.

    Raises:
        ValueError: The payload is patchified or does not have batch size two.
    """
    if output._is_patchified or output.i2v._is_patchified:
        raise ValueError("Double buffering requires unpatchified encoder output.")
    tensors = (output.i2v.latent, output.i2v.mask, output.plucker)
    if any(tensor.shape[0] != 2 for tensor in tensors):
        raise ValueError("Encoder payload fields must have batch size two.")

    def session(index: int) -> I2VCamCtrlEmbeddings:
        return I2VCamCtrlEmbeddings(
            i2v=I2VCtrl(
                latent=output.i2v.latent.narrow(0, index, 1),
                mask=output.i2v.mask.narrow(0, index, 1),
                _is_patchified=False,
            ),
            plucker=output.plucker.narrow(0, index, 1),
            _is_patchified=False,
        )

    return session(0), session(1)


def validate_double_buffered_schedule(*, sessions_per_group: int) -> None:
    """Validate the two-slot pipeline scheduling contract.

    Args:
        sessions_per_group: Number of session slots assigned to each DiT pair.

    Raises:
        ValueError: The session count is not exactly two.
    """
    if sessions_per_group != 2:
        raise ValueError("Double buffering requires exactly two sessions per group.")


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
    parser.add_argument("--sessions-per-group", type=int, default=2)
    parser.add_argument(
        "--double-buffered",
        action="store_true",
        help=(
            "Split batch-two input into session-affine microbatches and overlap "
            "the two DiT stages."
        ),
    )
    parser.add_argument(
        "--compile-network",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--transport",
        choices=("mooncake", "nixl"),
        default="mooncake",
    )
    parser.add_argument("--rdma-device", default=None)
    parser.add_argument("--bandwidth-probe-mib", type=int, default=256)
    parser.add_argument("--bandwidth-probe-iters", type=int, default=8)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=_DEFAULT_REPLICATED_BASELINE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_disagg_pipeline_3x2"),
    )
    return parser.parse_args()


def _create_pipeline_groups(
    topology: PipelineTopology,
    *,
    device: torch.device,
) -> dict[tuple[int, int], ProcessGroup]:
    """Create every NCCL pair in global rank order."""
    rank = torch.distributed.get_rank()
    local_groups: dict[tuple[int, int], ProcessGroup] = {}
    for ranks in topology.dit_groups:
        group = torch.distributed.new_group(
            ranks=list(ranks),
            backend="nccl",
            device_id=device,
        )
        if rank in ranks:
            assert isinstance(group, ProcessGroup)
            local_groups[ranks] = group
    return local_groups


def _empty_bundle(
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


def _fanout_bundle(
    *,
    source_bundle: TensorBundle | None,
    ranks: tuple[int, int],
    group: ProcessGroup | None,
    device: torch.device,
) -> tuple[TensorBundle | None, float | None]:
    """Broadcast a leader-owned bundle to the second DiT pipeline rank."""
    rank = torch.distributed.get_rank()
    leader = ranks[0]
    descriptors = _broadcast_object(
        describe_tensor_bundle(source_bundle)
        if rank == leader and source_bundle is not None
        else None,
        source=leader,
    )
    if rank not in ranks:
        return None, None
    assert group is not None
    local_bundle = (
        source_bundle if rank == leader else _empty_bundle(descriptors, device=device)
    )
    assert local_bundle is not None

    torch.distributed.barrier(group=group)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for descriptor in descriptors:
        torch.distributed.broadcast(
            local_bundle[descriptor.name],
            src=leader,
            group=group,
        )
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        (time.perf_counter() - started) * 1000.0,
        device=device,
    )
    torch.distributed.all_reduce(
        elapsed,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    return local_bundle, float(elapsed.item())


def _probe_pipeline_links(
    topology: PipelineTopology,
    *,
    local_groups: dict[tuple[int, int], ProcessGroup],
    size_mib: int,
    iterations: int,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    """Measure one-way NCCL point-to-point bandwidth for every DiT pair."""
    rank = torch.distributed.get_rank()
    payload_bytes = size_mib * 2**20
    local_results: dict[str, list[dict[str, float]]] = {}
    for ranks in topology.dit_groups:
        torch.distributed.barrier()
        if rank in ranks:
            group = local_groups[ranks]
            buffer = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
            for _ in range(2):
                if rank == ranks[0]:
                    torch.distributed.send(buffer, dst=ranks[1], group=group)
                else:
                    torch.distributed.recv(buffer, src=ranks[0], group=group)
            samples: list[dict[str, float]] = []
            for _ in range(iterations):
                torch.distributed.barrier(group=group)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                if rank == ranks[0]:
                    torch.distributed.send(buffer, dst=ranks[1], group=group)
                else:
                    torch.distributed.recv(buffer, src=ranks[0], group=group)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(
                    (time.perf_counter() - started) * 1000.0,
                    device=device,
                )
                torch.distributed.all_reduce(
                    elapsed,
                    op=torch.distributed.ReduceOp.MAX,
                    group=group,
                )
                if rank == ranks[0]:
                    elapsed_ms = float(elapsed.item())
                    samples.append(
                        {
                            "payload_bytes": float(payload_bytes),
                            "transfer_ms": elapsed_ms,
                            "bandwidth_gbps": payload_bytes
                            / (elapsed_ms / 1000.0)
                            / 1e9,
                        }
                    )
            if rank == ranks[0]:
                local_results[f"{ranks[0]}->{ranks[1]}"] = samples
        torch.distributed.barrier()

    gathered: list[dict[str, list[dict[str, float]]] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_results)
    if rank != topology.io_rank:
        return {}
    merged: dict[str, list[dict[str, float]]] = {}
    for result in gathered:
        if result:
            merged.update(result)
    return merged


def _summarize(
    *,
    records: list[dict[str, Any]],
    topology: PipelineTopology,
    p2p_probe: dict[str, list[dict[str, float]]],
    memory: dict[str, list[float]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    total_frames = sum(record["output_frames"] for record in measured)
    total_wall_s = sum(record["wave_latency_ms"] for record in measured) / 1000.0
    aggregate_fps = total_frames / total_wall_s
    probe_bandwidth = [
        sample["bandwidth_gbps"] for samples in p2p_probe.values() for sample in samples
    ]
    return {
        "aggregate_fps": aggregate_fps,
        "per_session_fps": aggregate_fps / topology.session_count,
        "wave_latency_ms": _metric_summary(
            [record["wave_latency_ms"] for record in measured]
        ),
        "encoder_wave_ms": _metric_summary(
            [record["encoder_wave_ms"] for record in measured]
        ),
        "dit_group_ms": _metric_summary(
            [
                worker["dit_ms"]
                for record in measured
                for worker in record["dit_group_leaders"]
            ]
        ),
        "finalize_group_ms": _metric_summary(
            [
                worker["finalize_ms"]
                for record in measured
                for worker in record["dit_group_leaders"]
            ]
        ),
        "decoder_wave_ms": _metric_summary(
            [record["decoder_wave_ms"] for record in measured]
        ),
        "pair_fanout_ms": _metric_summary(
            [value for record in measured for value in record["pair_fanout_ms"]]
        ),
        "p2p_probe_gbps": {
            "all_pairs": _metric_summary(probe_bandwidth),
            "by_pair": {
                pair: _metric_summary([sample["bandwidth_gbps"] for sample in samples])
                for pair, samples in p2p_probe.items()
            },
        },
        "memory": memory,
        "baseline": {
            "topology": "1 I/O + 7 full DiT replicas",
            "sessions": baseline["topology"]["sessions_per_wave"],
            "aggregate_fps": baseline["performance"]["aggregate_fps"],
            "per_session_fps": baseline["performance"]["per_session_fps"],
            "max_dit_peak_gib": max(baseline["peak_allocated_gib_by_rank"][1:]),
        },
    }


def _write_report(
    args: argparse.Namespace,
    *,
    environment: dict[str, Any],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    p2p_probe: dict[str, list[dict[str, float]]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "environment": environment,
        "summary": summary,
        "records": records,
        "p2p_probe": p2p_probe,
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    memory = summary["memory"]
    baseline = summary["baseline"]
    markdown = f"""# LingBot two-stage DiT pipeline benchmark

## Result

| Metric | Pipeline-parallel result | Replicated DiT reference |
| --- | ---: | ---: |
| Concurrent sessions | {environment["sessions"]} | {baseline["sessions"]} |
| Aggregate generated FPS | **{summary["aggregate_fps"]:.2f}** | {baseline["aggregate_fps"]:.2f} |
| Generated FPS per session | **{summary["per_session_fps"]:.2f}** | {baseline["per_session_fps"]:.2f} |
| Median wave latency | **{summary["wave_latency_ms"]["median"]:.2f} ms** | 2358.51 ms |
| Maximum DiT-rank capacity | **{max(memory["required_capacity_gib_by_rank"][1:7]):.2f} GiB** | {baseline["max_dit_peak_gib"]:.2f} GiB |
| Median 256 MiB NCCL P2P bandwidth | **{summary["p2p_probe_gbps"]["all_pairs"]["median"]:.2f} GB/s** | — |

## Topology

```text
GPU 0      shared encoder + decoder
GPU 1–2    DiT group A
GPU 3–4    DiT group B
GPU 5–6    DiT group C
GPU 7      spare
```

Each DiT group holds {args.sessions_per_group} sessions using the
{("double-buffered fill/drain schedule" if args.double_buffered else "fixed batched schedule")}.
The first rank owns layers 0–19; the second owns layers 20–39 and the output
head. CUDA graph capture is disabled because the graph boundary contains NCCL
point-to-point operations.

## Reproduction

```bash
{environment["command"]}
```

- Repository revision: `{environment["commit"]}`{" (modified worktree)" if environment["worktree_dirty"] else ""}
- Slurm job: `{environment["slurm_job_id"]}` on `{environment["hostname"]}`
- GPU: `{environment["gpus"][0]}` × {len(environment["gpus"])}
- Warmup / measured blocks: {args.warmup_blocks} / {args.measured_blocks}
"""
    (args.output_dir / "README.md").write_text(markdown)


def main() -> None:
    """Run three two-rank DiT pipelines with shared I/O on one node."""
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    topology = build_pipeline_topology(
        world_size=world_size,
        sessions_per_group=args.sessions_per_group,
    )
    if args.warmup_blocks < 0 or args.measured_blocks <= 0:
        raise ValueError("warmup-blocks must be >= 0 and measured-blocks must be > 0.")
    if args.double_buffered:
        validate_double_buffered_schedule(sessions_per_group=args.sessions_per_group)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    base_config = PIPELINE_CONFIGS[args.model]
    dit_batch_size = 1 if args.double_buffered else args.sessions_per_group
    dit_config = derive_config(
        base_config,
        diffusion_model=dict(
            transformer=dict(
                batch_shape=(dit_batch_size,),
                compile_network=args.compile_network,
                use_cuda_graph=False,
            )
        ),
    )

    encoder_stage = (
        LingbotEncoderStage(base_config).to(device).eval()
        if rank == topology.io_rank
        else None
    )
    decoder_stage = (
        LingbotDecoderStage(base_config).to(device).eval()
        if rank == topology.io_rank
        else None
    )
    dit_stage = LingbotDiTStage(dit_config) if rank in topology.dit_ranks else None
    encoder_inputs = (
        _load_encoder_inputs(args, device=device) if rank == topology.io_rank else None
    )

    torch.distributed.init_process_group("gloo")
    local_groups = _create_pipeline_groups(topology, device=device)
    local_group_ranks = next(
        (ranks for ranks in topology.dit_groups if rank in ranks),
        None,
    )
    if dit_stage is not None and local_group_ranks is not None:
        stage_index = local_group_ranks.index(rank)
        dit_stage.configure_pipeline_parallel(
            stage_index=stage_index,
            stage_count=2,
            group=local_groups[local_group_ranks],
            ranks=local_group_ranks,
        )
        dit_stage = dit_stage.to(device).eval()

    weight_memory_gib = torch.cuda.memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    transport = _create_transport(args, rank=rank)
    pool = RegisteredTensorPool(transport, max_buffers_per_bucket=16)

    encoder_caches: list[Any] = []
    conditioning_bundles: list[TensorBundle] = []
    prompt = None
    image = None
    intrinsics = None
    poses = None
    world_scale = None
    if encoder_stage is not None and encoder_inputs is not None:
        prompt, image, intrinsics, poses, world_scale = encoder_inputs
        conditionings: list[LingbotConditioning] = []
        for _ in range(topology.session_count):
            cache, conditioning = encoder_stage.initialize_cache(
                text=[prompt],
                image=image,
            )
            encoder_caches.append(cache)
            conditionings.append(conditioning)
        for group_index in range(len(topology.dit_groups)):
            start = group_index * topology.sessions_per_group
            end = start + topology.sessions_per_group
            conditioning_bundles.append(
                conditioning_to_bundle(stack_conditioning(conditionings[start:end]))
            )

    height_width = (
        (conditionings[0].height, conditionings[0].width)
        if rank == topology.io_rank
        else None
    )
    height_width = _broadcast_object(height_width, source=topology.io_rank)
    dit_caches: list[Any] = []
    for group_index, ranks in enumerate(topology.dit_groups):
        leader = ranks[0]
        received_context, _, _ = _transfer_bundle(
            transport,
            source=topology.io_rank,
            destination=leader,
            source_bundle=(
                conditioning_bundles[group_index] if rank == topology.io_rank else None
            ),
            device=device,
        )
        fanned_context, _ = _fanout_bundle(
            source_bundle=received_context if rank == leader else None,
            ranks=ranks,
            group=local_groups.get(ranks),
            device=device,
        )
        if rank in ranks:
            assert dit_stage is not None
            assert fanned_context is not None
            conditioning = conditioning_from_bundle(
                fanned_context,
                height=height_width[0],
                width=height_width[1],
            )
            if args.double_buffered:
                dit_caches.extend(
                    dit_stage.initialize_cache(session_conditioning)
                    for session_conditioning in split_conditioning(conditioning)
                )
            else:
                dit_caches.append(dit_stage.initialize_cache(conditioning))
        if rank == leader and received_context is not None:
            transport.unregister(received_context)

    decoder_caches = (
        [decoder_stage.initialize_cache() for _ in range(topology.session_count)]
        if decoder_stage is not None
        else []
    )
    cache_initialization_peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)

    p2p_probe = _probe_pipeline_links(
        topology,
        local_groups=local_groups,
        size_mib=args.bandwidth_probe_mib,
        iterations=args.bandwidth_probe_iters,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)

    total_blocks = args.warmup_blocks + args.measured_blocks
    frame_starts = [0] * topology.session_count
    encoder_channels: list[_TransferChannel | None] = [None] * len(topology.dit_groups)
    decoder_channels: list[_TransferChannel | None] = [None] * len(topology.dit_groups)
    records: list[dict[str, Any]] = []
    for autoregressive_index in range(total_blocks):
        torch.distributed.barrier()
        wave_started = time.perf_counter()
        local: dict[str, Any] = {}
        encoder_times: list[float] = []
        group_encoder_bundles: list[TensorBundle] = []
        if encoder_stage is not None:
            assert intrinsics is not None
            assert poses is not None
            assert world_scale is not None
            encoded_sessions: list[I2VCamCtrlEmbeddings] = []
            for session_index in range(topology.session_count):
                num_input_frames = encoder_stage.get_num_input_frames(
                    autoregressive_index
                )
                frame_start = frame_starts[session_index]
                frame_end = frame_start + num_input_frames
                if frame_end > poses.shape[0]:
                    raise RuntimeError(
                        f"Camera trajectory ended before AR block {autoregressive_index}."
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
                encoded_sessions.append(encoded)
                encoder_times.append(encoder_ms)
                frame_starts[session_index] = frame_end
            for group_index in range(len(topology.dit_groups)):
                start = group_index * topology.sessions_per_group
                end = start + topology.sessions_per_group
                group_encoder_bundles.append(
                    encoder_output_to_bundle(
                        stack_encoder_outputs(encoded_sessions[start:end])
                    )
                )

        pending_encoder = []
        received_leader_bundle = None
        for group_index, ranks in enumerate(topology.dit_groups):
            leader = ranks[0]
            source_bundle = (
                group_encoder_bundles[group_index] if rank == topology.io_rank else None
            )
            channel = encoder_channels[group_index]
            if channel is None:
                channel = _setup_channel(
                    transport,
                    pool,
                    source=topology.io_rank,
                    destination=leader,
                    source_bundle=source_bundle,
                    device=device,
                )
                encoder_channels[group_index] = channel
            handle, started = _submit_channel(transport, channel, source_bundle)
            pending_encoder.append((channel, handle, started, source_bundle))
            if rank == leader:
                assert channel.receiver is not None
                received_leader_bundle = channel.receiver.bundle

        encoder_transfers = []
        for channel, handle, started, source_bundle in pending_encoder:
            stats, handoff_ms = _finish_channel(
                transport,
                channel,
                handle,
                started,
            )
            if rank == topology.io_rank:
                assert source_bundle is not None
                transport.unregister(source_bundle)
                encoder_transfers.append({**vars(stats), "handoff_ms": handoff_ms})

        received_encoded = None
        for ranks in topology.dit_groups:
            fanned, fanout_ms = _fanout_bundle(
                source_bundle=(received_leader_bundle if rank == ranks[0] else None),
                ranks=ranks,
                group=local_groups.get(ranks),
                device=device,
            )
            if rank in ranks:
                received_encoded = fanned
                assert fanout_ms is not None
                local["pair_fanout_ms"] = fanout_ms

        clean_bundle = None
        if dit_stage is not None:
            assert received_encoded is not None
            encoded_batch = encoder_output_from_bundle(received_encoded)
            if args.double_buffered:
                assert len(dit_caches) == 2
                session_inputs = split_encoder_outputs(encoded_batch)
                clean_latents, local["dit_ms"] = _timed_cuda(
                    partial(
                        dit_stage.generate_double_buffered,
                        autoregressive_index=autoregressive_index,
                        caches=(dit_caches[0], dit_caches[1]),
                        inputs=session_inputs,
                    )
                )
                clean_latent = torch.cat(clean_latents, dim=0)
            else:
                assert len(dit_caches) == 1
                clean_latent, local["dit_ms"] = _timed_cuda(
                    partial(
                        dit_stage.generate,
                        autoregressive_index=autoregressive_index,
                        cache=dit_caches[0],
                        input=encoded_batch,
                    )
                )
            assert local_group_ranks is not None
            local["group"] = list(local_group_ranks)
            local["stage_index"] = local_group_ranks.index(rank)
            local["schedule"] = (
                "double-buffered" if args.double_buffered else "fixed-batch"
            )
            if rank == local_group_ranks[0]:
                clean_bundle = {"clean_latent": clean_latent.contiguous()}

        torch.distributed.barrier()
        pending_decoder = []
        decoder_inputs: list[TensorBundle] = []
        for group_index, ranks in enumerate(topology.dit_groups):
            leader = ranks[0]
            channel = decoder_channels[group_index]
            if channel is None:
                channel = _setup_channel(
                    transport,
                    pool,
                    source=leader,
                    destination=topology.io_rank,
                    source_bundle=clean_bundle if rank == leader else None,
                    device=device,
                )
                decoder_channels[group_index] = channel
            handle, started = _submit_channel(
                transport,
                channel,
                clean_bundle if rank == leader else None,
            )
            pending_decoder.append(
                (channel, handle, started, clean_bundle if rank == leader else None)
            )
            if rank == topology.io_rank:
                assert channel.receiver is not None
                decoder_inputs.append(channel.receiver.bundle)

        if dit_stage is not None:
            if args.double_buffered:
                assert len(dit_caches) == 2
                _, local["finalize_ms"] = _timed_cuda(
                    partial(
                        dit_stage.finalize_double_buffered,
                        autoregressive_index=autoregressive_index,
                        caches=(dit_caches[0], dit_caches[1]),
                    )
                )
            else:
                assert len(dit_caches) == 1
                _, local["finalize_ms"] = _timed_cuda(
                    partial(
                        dit_stage.finalize,
                        autoregressive_index=autoregressive_index,
                        cache=dit_caches[0],
                    )
                )

        decoder_transfers = []
        for channel, handle, started, source_bundle in pending_decoder:
            stats, handoff_ms = _finish_channel(
                transport,
                channel,
                handle,
                started,
            )
            if rank == channel.source:
                assert source_bundle is not None
                transport.unregister(source_bundle)
            if rank == topology.io_rank:
                decoder_transfers.append({**vars(stats), "handoff_ms": handoff_ms})

        if decoder_stage is not None:
            decoder_times: list[float] = []
            output_frames = 0
            session_index = 0
            for group_bundle in decoder_inputs:
                clean_batch = group_bundle["clean_latent"]
                for clean_session in clean_batch.split(1, dim=0):
                    decoded, decoder_ms = _timed_cuda(
                        partial(
                            decoder_stage.decode,
                            input=clean_session,
                            autoregressive_index=autoregressive_index,
                            cache=decoder_caches[session_index],
                        )
                    )
                    decoder_times.append(decoder_ms)
                    output_frames += decoded.shape[-4]
                    session_index += 1
            local["decoder_wave_ms"] = sum(decoder_times)
            local["output_frames"] = output_frames

        torch.distributed.barrier()
        if rank == topology.io_rank:
            local["encoder_wave_ms"] = sum(encoder_times)
            local["encoder_to_dit"] = encoder_transfers
            local["dit_to_decoder"] = decoder_transfers
            local["wave_latency_ms"] = (time.perf_counter() - wave_started) * 1000.0

        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        if rank == topology.io_rank:
            io_record = gathered[topology.io_rank]
            assert io_record is not None
            group_leaders = []
            pair_fanout_ms = []
            for ranks in topology.dit_groups:
                leader_record = gathered[ranks[0]]
                follower_record = gathered[ranks[1]]
                assert leader_record is not None and follower_record is not None
                group_leaders.append(leader_record)
                pair_fanout_ms.append(
                    max(
                        leader_record["pair_fanout_ms"],
                        follower_record["pair_fanout_ms"],
                    )
                )
            records.append(
                {
                    "autoregressive_index": autoregressive_index,
                    "warmup": autoregressive_index < args.warmup_blocks,
                    **io_record,
                    "dit_group_leaders": group_leaders,
                    "pair_fanout_ms": pair_fanout_ms,
                }
            )

    rollout_peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    memory_records = {
        "weight_gib": weight_memory_gib,
        "cache_initialization_peak_gib": cache_initialization_peak_gib,
        "rollout_peak_gib": rollout_peak_gib,
        "required_capacity_gib": max(cache_initialization_peak_gib, rollout_peak_gib),
    }
    gathered_memory: list[dict[str, float] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered_memory, memory_records)

    if rank == topology.io_rank:
        baseline = json.loads(args.baseline_json.read_text())
        memory = {
            key + "_by_rank": [
                cast(dict[str, float], item)[key] for item in gathered_memory
            ]
            for key in memory_records
        }
        summary = _summarize(
            records=records,
            topology=topology,
            p2p_probe=p2p_probe,
            memory=memory,
            baseline=baseline,
        )
        environment = _environment(
            args,
            prompt=prompt,
            world_size=world_size,
            module_name="lingbot.disagg.benchmark_pipeline",
        )
        environment.update(
            {
                "allocation": {
                    "io_rank": topology.io_rank,
                    "dit_groups": [list(group) for group in topology.dit_groups],
                    "spare_ranks": list(topology.spare_ranks),
                },
                "sessions": topology.session_count,
                "sessions_per_group": topology.sessions_per_group,
                "schedule": (
                    "double-buffered" if args.double_buffered else "fixed-batch"
                ),
                "compile_network": args.compile_network,
                "cuda_graph": False,
                "pipeline_layers": [[0, 20], [20, 40]],
                "dit_internal_transport": "NCCL P2P over NVLink/NVSwitch",
                "stage_transport": args.transport,
            }
        )
        _write_report(
            args,
            environment=environment,
            records=records,
            summary=summary,
            p2p_probe=p2p_probe,
        )

    for channel in (*encoder_channels, *decoder_channels):
        if channel is not None and channel.receiver is not None:
            pool.release(channel.receiver)
    pool.close()
    transport.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
