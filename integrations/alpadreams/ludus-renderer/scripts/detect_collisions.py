#!/usr/bin/env python3
"""Detect ego-obstacle collisions across a list of AV2 scene tars.

Loads each scene to GPU via the fast C++ pipeline (~10ms), runs 2D OBB
collision detection on-device (~0.1ms), and writes results as JSONLines.

Usage (single GPU):

    uv run python scripts/detect_collisions.py \
        --scene-list example_data/scene_paths_all.txt \
        --output-dir collision_results/

Resume after interruption (skips already-processed scenes):

    uv run python scripts/detect_collisions.py \
        --scene-list example_data/scene_paths_all.txt \
        --output-dir collision_results/ --resume

Manual sharding (e.g. for parallel jobs without Skippy):

    uv run python scripts/detect_collisions.py \
        --scene-list example_data/scene_paths_all.txt \
        --output-dir collision_results/ --start 0 --end 10000

Merge shard outputs into final collision list + metadata:

    uv run python scripts/detect_collisions.py \
        --merge --output-dir collision_results/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _process_scenes(args):
    import torch
    from ludus_renderer.clipgt import load_av2_scene
    from ludus_renderer.collision import detect_collisions_from_scene

    scene_list = Path(args.scene_list)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_list) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(all_paths)
    paths = all_paths[start:end]

    shard_tag = f"{start}_{end}"
    results_path = out_dir / f"results_{shard_tag}.jsonl"
    collision_list_path = out_dir / f"collisions_{shard_tag}.txt"

    done_keys = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add(rec["path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_keys)} scenes already processed")

    device = torch.device("cuda")

    n_total = len(paths)
    n_collision = 0
    n_error = 0
    t_start = time.perf_counter()

    results_f = open(results_path, "a")
    collision_f = open(collision_list_path, "a")

    try:
        for i, tar_path in enumerate(paths):
            if tar_path in done_keys:
                continue

            record = {"path": tar_path, "collision": False, "n_events": 0, "events": []}

            try:
                scene = load_av2_scene(tar_path, device=device)
                result = detect_collisions_from_scene(scene)

                if result.has_collision:
                    n_collision += 1
                    record["collision"] = True
                    record["n_events"] = len(result.events)
                    record["events"] = [
                        {
                            "timestamp_us": e.timestamp_us,
                            "track_idx": e.track_idx,
                            "min_dist_m": round(e.distance_m, 4),
                        }
                        for e in result.events
                    ]
                    collision_f.write(tar_path + "\n")
                    collision_f.flush()

            except Exception as exc:
                n_error += 1
                record["error"] = str(exc)[:200]

            results_f.write(json.dumps(record) + "\n")

            if (i + 1) % 100 == 0 or (i + 1) == n_total:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate if rate > 0 else 0
                print(
                    f"[{i+1}/{n_total}] "
                    f"collisions={n_collision} errors={n_error} "
                    f"{rate:.1f} scenes/s  ETA {eta/60:.0f}m",
                    flush=True,
                )
                results_f.flush()
    finally:
        results_f.close()
        collision_f.close()

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone: {n_total} scenes in {elapsed:.1f}s "
        f"({n_total/elapsed:.1f}/s), "
        f"{n_collision} collisions, {n_error} errors"
    )
    print(f"Results: {results_path}")
    print(f"Collision list: {collision_list_path}")


def _merge(args):
    out_dir = Path(args.output_dir)

    collision_paths = sorted(out_dir.glob("collisions_*.txt"))
    result_paths = sorted(out_dir.glob("results_*.jsonl"))

    merged_list = out_dir / "scene_paths_collision.txt"
    merged_meta = out_dir / "collision_metadata.jsonl"

    seen = set()
    n_collisions = 0
    n_total = 0

    with open(merged_list, "w") as flist, open(merged_meta, "w") as fmeta:
        for rp in result_paths:
            with open(rp) as f:
                for line in f:
                    n_total += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("collision") and rec["path"] not in seen:
                        seen.add(rec["path"])
                        n_collisions += 1
                        flist.write(rec["path"] + "\n")
                        fmeta.write(line)
                    elif not rec.get("collision"):
                        pass

    # Also pick up any collision-list-only entries not in jsonl
    for cp in collision_paths:
        with open(cp) as f:
            for line in f:
                p = line.strip()
                if p and p not in seen:
                    seen.add(p)
                    n_collisions += 1
                    with open(merged_list, "a") as flist:
                        flist.write(p + "\n")

    print(f"Merged {n_total} scene results → {n_collisions} collisions")
    print(f"  {merged_list}")
    print(f"  {merged_meta}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated ego-obstacle collision detection"
    )
    parser.add_argument("--scene-list", type=str, help="Text file with scene tar paths")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--start", type=int, default=None, help="Start index in scene list")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed scenes")
    parser.add_argument("--merge", action="store_true", help="Merge shard outputs instead of processing")
    args = parser.parse_args()

    if args.merge:
        _merge(args)
    else:
        if not args.scene_list:
            parser.error("--scene-list is required unless --merge is used")
        _process_scenes(args)


if __name__ == "__main__":
    main()
