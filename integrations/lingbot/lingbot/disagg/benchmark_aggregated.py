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

"""Benchmark the replicated full LingBot pipeline with WORLD context parallelism."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.distributed import ProcessGroup

from flashdreams.core.distributed import init as init_distributed
from flashdreams.core.distributed import shutdown as shutdown_distributed
from flashdreams.infra.config import derive_config
from lingbot.config import PIPELINE_CONFIGS
from lingbot.disagg.benchmark import (
    _environment,
    _load_encoder_inputs,
    _metric_summary,
)
from lingbot.disagg.benchmark_cp import _cp_collective_probe
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.transformer import LingbotWorldTransformerConfig

_DEFAULT_COMPARISON = Path(
    "integrations/lingbot/docs/benchmark_h100_cp6_single_session/benchmark.json"
)
_WAN_SPATIAL_COMPRESSION = 8


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
    parser.add_argument(
        "--pixel-height",
        type=int,
        default=448,
        help="448 is the closest height to 464 whose LingBot token grid divides by CP8.",
    )
    parser.add_argument("--pixel-width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--cp-method",
        choices=("ring", "ulysses"),
        default="ulysses",
    )
    parser.add_argument("--bandwidth-probe-mib", type=int, default=256)
    parser.add_argument("--bandwidth-probe-iters", type=int, default=8)
    parser.add_argument("--rdma-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--comparison-json", type=Path, default=_DEFAULT_COMPARISON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lingbot_aggregated_cp8"),
    )
    return parser.parse_args()


def _token_layout(
    *,
    pixel_height: int,
    pixel_width: int,
    len_t: int,
    patch_size: tuple[int, int, int],
    cp_size: int,
) -> dict[str, int]:
    """Return the LingBot token layout after validating CP divisibility."""
    kt, kh, kw = patch_size
    pixel_patch_height = _WAN_SPATIAL_COMPRESSION * kh
    pixel_patch_width = _WAN_SPATIAL_COMPRESSION * kw
    if pixel_height % pixel_patch_height or pixel_width % pixel_patch_width:
        raise ValueError(
            f"Pixel resolution {pixel_width}x{pixel_height} must be divisible by "
            f"{pixel_patch_width}x{pixel_patch_height} for the Wan VAE and DiT patch."
        )
    if len_t % kt:
        raise ValueError(f"len_t={len_t} must be divisible by temporal patch {kt}.")
    latent_height = pixel_height // _WAN_SPATIAL_COMPRESSION
    latent_width = pixel_width // _WAN_SPATIAL_COMPRESSION
    total_tokens = (len_t // kt) * (latent_height // kh) * (latent_width // kw)
    if total_tokens % cp_size:
        raise ValueError(
            f"Resolution {pixel_width}x{pixel_height} produces {total_tokens} tokens, "
            f"which cannot be evenly sharded over CP{cp_size}."
        )
    return {
        "latent_height": latent_height,
        "latent_width": latent_width,
        "tokens_per_chunk": total_tokens,
        "tokens_per_rank": total_tokens // cp_size,
    }


def _read_comparison(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _summarize(
    *,
    records: list[dict[str, Any]],
    tokens_per_chunk: int,
    cp_probe: dict[str, list[dict[str, float]]],
    peak_memory_gib_by_rank: list[float],
    steady_memory_gib_by_rank: list[float],
    initialization_peak_gib_by_rank: list[float],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    latency_values = [record["end_to_end_ms"] for record in measured]
    elapsed_s = sum(latency_values) / 1000.0
    output_frames = sum(record["output_frames"] for record in measured)
    summary: dict[str, Any] = {
        "fps": output_frames / elapsed_s,
        "latency_ms": _metric_summary(latency_values),
        "encoder_ms": _metric_summary(
            [record["critical_rank"]["encode_ms"] for record in measured]
        ),
        "dit_ms": _metric_summary(
            [record["critical_rank"]["diffuse_ms"] for record in measured]
        ),
        "decoder_ms": _metric_summary(
            [record["critical_rank"]["decode_ms"] for record in measured]
        ),
        "finalize_ms": _metric_summary(
            [record["critical_rank"]["finalize_ms"] for record in measured]
        ),
        "tokens_per_chunk": tokens_per_chunk,
        "token_throughput_per_second": tokens_per_chunk * len(measured) / elapsed_s,
        "cp_probe_gbps": {
            collective: _metric_summary(
                [sample["bandwidth_gbps"] for sample in samples]
            )
            for collective, samples in cp_probe.items()
        },
        "memory": {
            "peak_gib_by_rank": peak_memory_gib_by_rank,
            "steady_allocated_gib_by_rank": steady_memory_gib_by_rank,
            "initialization_peak_gib_by_rank": initialization_peak_gib_by_rank,
            "node_peak_gib": sum(peak_memory_gib_by_rank),
            "node_steady_allocated_gib": sum(steady_memory_gib_by_rank),
            "per_rank_peak_gib": _metric_summary(peak_memory_gib_by_rank),
        },
    }
    if comparison is not None:
        previous = comparison["summary"]
        previous_environment = comparison["environment"]
        previous_tokens = (
            previous_environment["latent_frames_per_chunk"]
            * (previous_environment["resolution"][0] // 16)
            * (previous_environment["resolution"][1] // 16)
        )
        previous_memory = previous_environment["peak_memory_gib_by_rank"]
        allocation = previous_environment.get("allocation", {})
        summary["comparison"] = {
            "topology": (
                f"1 encoder : CP{allocation.get('cp_size', '?')} DiT : 1 decoder"
            ),
            "resolution": previous_environment["resolution"],
            "fps": previous["fps"],
            "latency_ms": previous["latency_ms"]["median"],
            "tokens_per_chunk": previous_tokens,
            "token_throughput_per_second": previous_tokens * previous["fps"] / 12.0,
            "node_peak_gib": sum(previous_memory),
            "latency_speedup": previous["latency_ms"]["median"]
            / summary["latency_ms"]["median"],
            "fps_ratio": summary["fps"] / previous["fps"],
            "token_throughput_ratio": summary["token_throughput_per_second"]
            / (previous_tokens * previous["fps"] / 12.0),
            "node_peak_memory_ratio": summary["memory"]["node_peak_gib"]
            / sum(previous_memory),
        }
    return summary


def _write_report(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    cp_probe: dict[str, list[dict[str, float]]],
    environment: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "summary": summary,
                "records": records,
                "cp_probe": cp_probe,
            },
            indent=2,
        )
        + "\n"
    )
    memory = summary["memory"]
    comparison = summary.get("comparison")
    comparison_rows = ""
    if comparison is not None:
        comparison_rows = "\n".join(
            (
                f"| Median chunk latency | {comparison['latency_ms']:.2f} ms | "
                f"{summary['latency_ms']['median']:.2f} ms | "
                f"{comparison['latency_speedup']:.2f}× faster |",
                f"| Generated FPS | {comparison['fps']:.2f} | "
                f"{summary['fps']:.2f} | {comparison['fps_ratio']:.2f}× |",
                f"| DiT token throughput | "
                f"{comparison['token_throughput_per_second']:.0f} token/s | "
                f"{summary['token_throughput_per_second']:.0f} token/s | "
                f"{comparison['token_throughput_ratio']:.2f}× |",
                f"| Node peak allocated HBM | "
                f"{comparison['node_peak_gib']:.2f} GiB | "
                f"{memory['node_peak_gib']:.2f} GiB | "
                f"{comparison['node_peak_memory_ratio']:.2f}× |",
            )
        )
    markdown = f"""# LingBot aggregated CP{environment["allocation"]["cp_size"]} benchmark

All ranks own the complete encoder, DiT, and decoder pipeline. The DiT token
axis is context-parallel across WORLD with {args.cp_method} attention. Encoder
and decoder work is replicated on every rank; there are no RDMA stage
boundaries in this topology.

## Result

| Metric | Median | P90 |
| --- | ---: | ---: |
| End-to-end chunk latency | {summary["latency_ms"]["median"]:.2f} ms | {summary["latency_ms"]["p90"]:.2f} ms |
| Encoder critical-rank compute | {summary["encoder_ms"]["median"]:.2f} ms | {summary["encoder_ms"]["p90"]:.2f} ms |
| DiT critical-rank denoise | {summary["dit_ms"]["median"]:.2f} ms | {summary["dit_ms"]["p90"]:.2f} ms |
| Decoder critical-rank compute | {summary["decoder_ms"]["median"]:.2f} ms | {summary["decoder_ms"]["p90"]:.2f} ms |
| DiT cache finalize | {summary["finalize_ms"]["median"]:.2f} ms | {summary["finalize_ms"]["p90"]:.2f} ms |
| NCCL broadcast probe | {summary["cp_probe_gbps"]["broadcast"]["median"]:.2f} GB/s | {summary["cp_probe_gbps"]["broadcast"]["p90"]:.2f} GB/s |
| NCCL all-gather probe | {summary["cp_probe_gbps"]["all_gather"]["median"]:.2f} GB/s | {summary["cp_probe_gbps"]["all_gather"]["p90"]:.2f} GB/s |

- Generated throughput: **{summary["fps"]:.2f} FPS**
- DiT token throughput: **{summary["token_throughput_per_second"]:.0f} token/s**
- Peak allocated HBM: **{memory["per_rank_peak_gib"]["min"]:.2f}–{memory["per_rank_peak_gib"]["max"]:.2f} GiB per rank**, **{memory["node_peak_gib"]:.2f} GiB node total**
- Steady allocated HBM after rollout: **{memory["node_steady_allocated_gib"]:.2f} GiB node total**

## Comparison with disaggregated CP

| Metric | Disaggregated CP6 | Aggregated CP8 | Change |
| --- | ---: | ---: | ---: |
{comparison_rows}

The resolutions differ because the tracked 832×464 grid has 4,524 tokens,
which is not divisible by eight. CP8 uses 832×448 and 4,368 tokens (3.45%
fewer). Token throughput is therefore the fairest compute-rate comparison.

## Reproduction

```bash
{environment["command"]}
```

- Repository revision: `{environment["commit"]}`{" (modified worktree)" if environment["worktree_dirty"] else ""}
- Slurm: job `{environment["slurm_job_id"]}` on `{environment["hostname"]}`
- GPU: `{environment["gpus"][0]}` × {len(environment["gpus"])}
- Resolution: `{args.pixel_width}x{args.pixel_height}`
- Warmup / measured blocks: {args.warmup_blocks} / {args.measured_blocks}
"""
    (args.output_dir / "README.md").write_text(markdown)


def main() -> None:
    """Run the complete pipeline on every rank with WORLD context parallelism."""
    args = _parse_args()
    if args.warmup_blocks < 0 or args.measured_blocks <= 0:
        raise ValueError("warmup-blocks must be >= 0 and measured-blocks must be > 0.")

    init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 8:
        raise ValueError(
            f"Aggregated baseline requires exactly 8 ranks; got {world_size}."
        )
    device = torch.device(f"cuda:{local_rank}")

    base_config = PIPELINE_CONFIGS[args.model]
    transformer_config = base_config.diffusion_model.transformer
    assert isinstance(transformer_config, LingbotWorldTransformerConfig)
    layout = _token_layout(
        pixel_height=args.pixel_height,
        pixel_width=args.pixel_width,
        len_t=transformer_config.len_t,
        patch_size=transformer_config.network.patch_size,
        cp_size=world_size,
    )
    if (
        args.cp_method == "ulysses"
        and transformer_config.network.num_heads % world_size
    ):
        raise ValueError(
            f"Ulysses requires {transformer_config.network.num_heads} attention heads "
            f"to divide CP{world_size}."
        )

    base_seed = base_config.diffusion_model.seed
    config = derive_config(
        base_config,
        diffusion_model={
            "seed": None if base_seed is None else base_seed + rank,
            "transformer": {"network": {"cp_method": args.cp_method}},
        },
    )
    pipeline = config.setup().to(device).eval()
    assert isinstance(pipeline, LingbotWorldInferencePipeline)
    prompt, image, intrinsics, poses, world_scale = _load_encoder_inputs(
        args,
        device=device,
    )
    cache = pipeline.initialize_cache(text=[prompt], image=image)

    world_group = torch.distributed.group.WORLD
    assert isinstance(world_group, ProcessGroup)
    cp_probe = _cp_collective_probe(
        cp_ranks=tuple(range(world_size)),
        cp_group=world_group,
        size_mib=args.bandwidth_probe_mib,
        iterations=args.bandwidth_probe_iters,
        device=device,
    )
    initialization_peak = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    frame_start = 0
    total_blocks = args.warmup_blocks + args.measured_blocks
    for autoregressive_index in range(total_blocks):
        num_input_frames = pipeline.get_num_input_frames(autoregressive_index)
        frame_end = frame_start + num_input_frames
        if frame_end > poses.shape[0]:
            raise RuntimeError(
                f"Camera trajectory ended at frame {poses.shape[0]} before "
                f"AR block {autoregressive_index}."
            )
        control = CamCtrlInput(
            intrinsics=intrinsics[frame_start:frame_end],
            poses=poses[frame_start:frame_end],
            world_scale=world_scale,
        )
        frame_start = frame_end

        torch.distributed.barrier()
        started = time.perf_counter()
        video = pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=control,
        )
        stats = pipeline.finalize(
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        assert stats is not None
        local_record = {
            **stats,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "output_frames": video.shape[-4],
            "rank": rank,
        }
        per_rank: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(per_rank, local_record)
        if rank == 0:
            rank_records = [item for item in per_rank if item is not None]
            records.append(
                {
                    "autoregressive_index": autoregressive_index,
                    "warmup": autoregressive_index < args.warmup_blocks,
                    "end_to_end_ms": max(item["wall_ms"] for item in rank_records),
                    "output_frames": rank_records[0]["output_frames"],
                    "critical_rank": {
                        metric: max(item[metric] for item in rank_records)
                        for metric in (
                            "encode_ms",
                            "diffuse_ms",
                            "decode_ms",
                            "finalize_ms",
                        )
                    },
                    "per_rank": rank_records,
                }
            )

    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
    steady_memory = torch.cuda.memory_allocated(device) / 2**30
    resource_record = {
        "peak": peak_memory,
        "steady": steady_memory,
        "initialization_peak": initialization_peak,
    }
    resources: list[dict[str, float] | None] = [None] * world_size
    torch.distributed.all_gather_object(resources, resource_record)
    if rank == 0:
        rank_resources = [item for item in resources if item is not None]
        environment = _environment(
            args,
            prompt=prompt,
            world_size=world_size,
            module_name="lingbot.disagg.benchmark_aggregated",
        )
        environment["allocation"] = {
            "full_pipeline_replicas": list(range(world_size)),
            "dit_cp_group": list(range(world_size)),
            "cp_size": world_size,
            "cp_method": args.cp_method,
        }
        environment["token_layout"] = layout
        environment["transport"] = {
            "stage_handoffs": "none",
            "dit_collectives": "NCCL",
        }
        environment["noise_seed_by_rank"] = [
            None if base_seed is None else base_seed + local_rank
            for local_rank in range(world_size)
        ]
        summary = _summarize(
            records=records,
            tokens_per_chunk=layout["tokens_per_chunk"],
            cp_probe=cp_probe,
            peak_memory_gib_by_rank=[item["peak"] for item in rank_resources],
            steady_memory_gib_by_rank=[item["steady"] for item in rank_resources],
            initialization_peak_gib_by_rank=[
                item["initialization_peak"] for item in rank_resources
            ],
            comparison=_read_comparison(args.comparison_json),
        )
        _write_report(
            args,
            records=records,
            cp_probe=cp_probe,
            environment=environment,
            summary=summary,
        )

    shutdown_distributed(synchronize=True, terminate_process=True)


if __name__ == "__main__":
    main()
