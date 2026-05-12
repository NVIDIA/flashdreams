#!/usr/bin/env python3
"""High-throughput CPU collision detection via multiprocessing.

Usage:
    # Default (cpu_count workers):
    uv run python scripts/detect_collisions_batched.py \
        --scene-list example_data/scene_paths_collision_35k.txt \
        --output-dir collision_results/

    # Custom worker count:
    uv run python scripts/detect_collisions_batched.py \
        --scene-list example_data/scene_paths_collision_35k.txt \
        --output-dir collision_results/ --workers 32

    # Benchmark sweep (find optimal worker count):
    uv run python scripts/detect_collisions_batched.py \
        --scene-list example_data/scene_paths_collision_35k.txt \
        --sweep --sweep-scenes 500
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def _worker_init():
    """Silence noisy imports in worker processes."""
    import warnings
    warnings.filterwarnings("ignore")


def _process_one(tar_path: str) -> dict:
    """Worker function: CPU-only collision detection on a single scene."""
    from ludus_renderer.collision import detect_collisions_cpu

    record = {"path": tar_path, "collision": False, "n_events": 0, "events": []}
    try:
        result = detect_collisions_cpu(tar_path)
        if result.skipped:
            record["skipped"] = True
        elif result.has_collision:
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
    except Exception as exc:
        record["error"] = str(exc)[:200]
    return record


def _run(args):
    scene_list = Path(args.scene_list)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_list) as f:
        all_paths = [line.strip() for line in f if line.strip()]

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(all_paths)
    paths = all_paths[start:end]

    shard_tag = f"{start}_{end}"
    results_path = out_dir / f"results_{shard_tag}.jsonl"
    collision_list_path = out_dir / f"collisions_{shard_tag}.txt"

    done_keys: set = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_keys)} scenes already processed")

    paths = [p for p in paths if p not in done_keys]
    n_total = len(paths)
    n_workers = args.workers or os.cpu_count() or 4

    n_collision = 0
    n_error = 0
    n_skipped = 0
    n_done = 0
    t_start = time.perf_counter()

    results_f = open(results_path, "a")
    collision_f = open(collision_list_path, "a")

    print(f"Processing {n_total} scenes with {n_workers} CPU workers "
          f"(cpu_count={os.cpu_count()})")

    try:
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            for record in pool.imap_unordered(
                _process_one, paths,
                chunksize=max(1, n_total // (n_workers * 10)),
            ):
                if record.get("skipped"):
                    n_skipped += 1
                elif record.get("collision"):
                    n_collision += 1
                    collision_f.write(record["path"] + "\n")
                    collision_f.flush()
                if "error" in record:
                    n_error += 1
                results_f.write(json.dumps(record) + "\n")
                n_done += 1
                if n_done % 200 == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t_start
                    rate = n_done / elapsed
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(
                        f"[{n_done}/{n_total}] "
                        f"collisions={n_collision} skipped={n_skipped} errors={n_error} "
                        f"{rate:.1f} scenes/s  ETA {eta / 60:.0f}m",
                        flush=True,
                    )
                    results_f.flush()
    finally:
        results_f.close()
        collision_f.close()

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone: {n_total} scenes in {elapsed:.1f}s "
        f"({n_total / elapsed:.1f}/s), "
        f"{n_collision} collisions, {n_skipped} skipped, {n_error} errors"
    )


def _sweep(args):
    """Benchmark sequential + multiple worker counts to find optimal throughput."""
    with open(args.scene_list) as f:
        all_paths = [line.strip() for line in f if line.strip()]

    n_test = min(args.sweep_scenes, len(all_paths))
    test_paths = all_paths[:n_test]
    cpu_count = os.cpu_count() or 4

    from ludus_renderer.collision import detect_collisions_cpu
    detect_collisions_cpu(test_paths[0])

    print(f"Benchmarking with {n_test} scenes, cpu_count={cpu_count}")
    print(f"\n{'mode':>20} {'scenes/s':>10} {'ms/scene':>10} {'speedup':>8}")
    print("-" * 52)

    # Sequential baseline
    t0 = time.perf_counter()
    for p in test_paths:
        detect_collisions_cpu(p)
    baseline = time.perf_counter() - t0
    baseline_rate = n_test / baseline
    print(f"{'sequential':>20} {baseline_rate:>10.1f} "
          f"{baseline / n_test * 1000:>10.1f} {'1.00x':>8}")

    worker_counts = sorted(set([2, 4, 8, 16, 32, 64,
                                cpu_count // 2, cpu_count, cpu_count * 2]))
    worker_counts = [w for w in worker_counts if 2 <= w <= cpu_count * 2]

    best_rate = baseline_rate
    best_label = "sequential"

    for n_workers in worker_counts:
        t0 = time.perf_counter()
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            list(pool.imap_unordered(
                _process_one, test_paths,
                chunksize=max(1, n_test // (n_workers * 4)),
            ))
        elapsed = time.perf_counter() - t0
        rate = n_test / elapsed
        speedup = rate / baseline_rate
        tag = " *" if n_workers == cpu_count else ""
        label = f"mp x{n_workers}"
        print(f"{label:>20} {rate:>10.1f} "
              f"{elapsed / n_test * 1000:>10.1f} {speedup:>7.2f}x{tag}")
        if rate > best_rate:
            best_rate = rate
            best_label = label

    print(f"\nBest: {best_label.strip()} -> {best_rate:.1f} scenes/s")
    print(f"  (* = cpu_count)")
    print(f"\nExtrapolated 23M scenes:")
    print(f"  Sequential:  {23e6 / baseline_rate / 3600:.0f}h")
    print(f"  {best_label.strip()}:  {23e6 / best_rate / 3600:.0f}h")


def main():
    parser = argparse.ArgumentParser(
        description="High-throughput CPU collision detection"
    )
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="collision_results")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU worker count (default: os.cpu_count())")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Benchmark worker counts to find optimal throughput")
    parser.add_argument("--sweep-scenes", type=int, default=500,
                        help="Number of scenes for sweep benchmark")
    args = parser.parse_args()

    if args.sweep:
        _sweep(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
