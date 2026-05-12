#!/usr/bin/env python3
"""Launch massive scene preprocessing via Skippy + Slurm.

Reads a scene-list text file, generates a key→path manifest, and submits
a Skippy pipeline that distributes scene caching across Slurm nodes.

Example — preprocess all scenes with default settings (auto batch size, 32 nodes):

    python scripts/skippy/run_pipeline.py \
        --scene-list example_data/scene_paths_all.txt

Dry run (print launch command without submitting):

    python scripts/skippy/run_pipeline.py \
        --scene-list example_data/scene_paths_all.txt \
        --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable (skippy also adds it via PYTHONPATH at runtime)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skippy.pipeline import Pipeline

from scripts.skippy.tasks import PreprocessSceneCacheTask


def _build_manifest(scene_list_path: str, output_path: str) -> list[str]:
    """Read scene list, write dbm manifest (O(1) lookup), return list of keys."""
    import dbm

    from ludus_renderer.scene_cache import scene_key

    keys = []
    with dbm.open(output_path, "c") as db, open(scene_list_path) as fin:
        for line in fin:
            tar_path = line.strip()
            if not tar_path:
                continue
            key = scene_key(tar_path)
            db[key] = tar_path
            keys.append(key)
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Launch scene preprocessing via Skippy"
    )
    parser.add_argument(
        "--scene-list", type=str, required=True,
        help="Text file with scene tar paths, one per line",
    )
    default_cache_dir = (
        f"/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos"
        f"/users/{os.environ['USER']}/ludus_renderer_cache"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=default_cache_dir,
        help=f"Output directory for cached .pt scenes (default: {default_cache_dir})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Items per Slurm job (default: auto-sized to saturate cluster)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=32,
        help="Max concurrent Slurm nodes (default: 32)",
    )
    parser.add_argument(
        "--concurrent-per-node", type=int, default=48,
        help="Concurrent processes per node (default: 48)",
    )
    parser.add_argument(
        "--time-limit", type=str, default="23:59:00",
        help="Slurm time limit per job (default: 23:59:00)",
    )
    parser.add_argument(
        "--mem", type=str, default="256G",
        help="Memory per Slurm node (default: 256G)",
    )
    parser.add_argument(
        "--partition", type=str, default="cpu",
        help="Slurm partition (default: cpu)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Pipeline name for monitoring (default: auto-generated)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N scenes (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print launch command without submitting",
    )
    args = parser.parse_args()

    env_path = sys.executable.rsplit("/bin/", 1)[0]

    user = os.environ["USER"]
    base = f"/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos/users/{user}"
    manifest_dir = f"{base}/pipeline_manifests"
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)

    from skippy.time import Time
    ts = Time.get_now_str()
    manifest_path = f"{manifest_dir}/scene_manifest_{ts}"

    print(f"Building manifest from {args.scene_list} ...")
    all_keys = _build_manifest(args.scene_list, manifest_path)
    if args.limit:
        all_keys = all_keys[: args.limit]
    print(f"  {len(all_keys)} scenes → {manifest_path}")

    # Auto-size batches to always saturate the cluster.
    # Aim for at least 2x max_nodes batches so the scheduler stays busy.
    min_batches = args.max_nodes * 2
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = max(1, len(all_keys) // min_batches)
    print(f"  batch_size={batch_size} → ~{(len(all_keys) + batch_size - 1) // batch_size} jobs across {args.max_nodes} nodes")

    pipeline = Pipeline(
        name=args.name or f"preprocess_cache_{len(all_keys)}",
        batch_size=batch_size,
        env_path=env_path,
        gpus_per_task=0,
        gpus_per_node=args.concurrent_per_node,
        max_concurrent_nodes=args.max_nodes,
        time_limit=args.time_limit,
        mem=args.mem,
        partition=args.partition,
    )

    error_log_dir = f"{args.cache_dir}/errors"
    task = PreprocessSceneCacheTask(
        output_root=args.cache_dir,
        name="preprocess",
        cache_dir=args.cache_dir,
        manifest_path=manifest_path,
        error_log_dir=error_log_dir,
    )
    pipeline.add_task(task, deps=())

    print(pipeline)

    pipeline.launch(tuple(all_keys), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
