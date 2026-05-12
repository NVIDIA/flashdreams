#!/usr/bin/env python3
"""Launch distributed collision detection as a Slurm array job.

Each array task processes a shard of the scene list using CPU multiprocessing.
Jobs are automatically visible in slurm-dash; logs are tailed via --output.

Usage:
    # Dry run (print sbatch script without submitting):
    uv run python scripts/launch_collision_slurm.py \
        --scene-list example_data/scene_paths_all.txt --dry-run

    # Launch 8 array tasks:
    uv run python scripts/launch_collision_slurm.py \
        --scene-list example_data/scene_paths_all.txt --num-tasks 8

    # Merge results after completion:
    uv run python scripts/launch_collision_slurm.py --merge \
        --output-dir /lustre/.../collision_results
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{max_array_idx}
#SBATCH --output={log_dir}/slurm-%A_%a.out
#SBATCH --error={log_dir}/slurm-%A_%a.err
#SBATCH --comment=logdir:{output_dir}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --ntasks=1

# Register with slurm-dash for directory browsing
command -v slurm-dash &>/dev/null && slurm-dash register {output_dir}

TOTAL={total_scenes}
NUM_TASKS={num_tasks}
SCENES_PER_TASK=$(( (TOTAL + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * SCENES_PER_TASK ))
END=$(( START + SCENES_PER_TASK ))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

echo "Task $SLURM_ARRAY_TASK_ID: scenes $START..$END (of $TOTAL)"

cd {project_root}
uv run python scripts/detect_collisions_batched.py \\
    --scene-list {scene_list} \\
    --output-dir {output_dir} \\
    --start $START --end $END \\
    --workers {workers} \\
    --resume
"""


def _count_scenes(scene_list: str) -> int:
    n = 0
    with open(scene_list) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _merge(output_dir: str):
    """Merge per-shard JSONL results into final collision list."""
    out = Path(output_dir)
    collision_list = out / "scene_paths_collision.txt"
    metadata_file = out / "collision_metadata.jsonl"

    n_total = 0
    n_collision = 0
    n_error = 0

    with open(collision_list, "w") as flist, open(metadata_file, "w") as fmeta:
        for results_file in sorted(out.glob("results_*.jsonl")):
            with open(results_file) as f:
                for line in f:
                    n_total += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_error += 1
                        continue
                    if rec.get("collision"):
                        n_collision += 1
                        flist.write(rec["path"] + "\n")
                        fmeta.write(line)

    # Also gather from per-shard collision lists
    existing = set()
    with open(collision_list) as f:
        existing = {l.strip() for l in f}

    for coll_file in sorted(out.glob("collisions_*.txt")):
        with open(coll_file) as f:
            for line in f:
                p = line.strip()
                if p and p not in existing:
                    existing.add(p)

    with open(collision_list, "w") as f:
        for p in sorted(existing):
            f.write(p + "\n")

    print(f"Merged {n_total} results: {n_collision} collisions, {n_error} errors")
    print(f"  {collision_list}")
    print(f"  {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch collision detection as a Slurm array job"
    )
    parser.add_argument("--scene-list", type=str)
    parser.add_argument("--num-tasks", type=int, default=8,
                        help="Number of Slurm array tasks (default: 8)")
    parser.add_argument("--workers", type=int, default=32,
                        help="CPU workers per task (default: 32)")
    parser.add_argument("--cpus-per-task", type=int, default=36,
                        help="CPUs requested per Slurm task (default: 36)")
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time-limit", type=str, default="04:00:00")
    parser.add_argument("--partition", type=str, default="cpu")
    parser.add_argument("--job-name", type=str, default="collision_detect")
    parser.add_argument("--account", type=str, default=None,
                        help="Slurm account (run: sacctmgr -nP show assoc "
                             "where user=$(whoami) format=account)")

    user = os.environ.get("USER", "unknown")
    default_output = (
        f"/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos"
        f"/users/{user}/collision_results"
    )
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch script without submitting")
    parser.add_argument("--merge", action="store_true",
                        help="Merge shard results instead of launching")
    args = parser.parse_args()

    if args.merge:
        _merge(args.output_dir)
        return

    if not args.scene_list:
        parser.error("--scene-list is required unless --merge is used")

    scene_list = str(Path(args.scene_list).resolve())
    project_root = str(Path(__file__).resolve().parents[1])
    output_dir = str(Path(args.output_dir).resolve())
    log_dir = f"{output_dir}/logs"

    print(f"Counting scenes in {scene_list} ...")
    total = _count_scenes(scene_list)
    scenes_per_task = (total + args.num_tasks - 1) // args.num_tasks
    print(f"  {total} scenes / {args.num_tasks} tasks = ~{scenes_per_task} scenes/task")
    print(f"  {args.workers} workers/task, ~{scenes_per_task / 735:.0f}min/task at 735 scenes/s")

    if not args.account:
        import subprocess as _sp
        result = _sp.run(
            ["sacctmgr", "-nP", "show", "assoc",
             f"where user={os.environ.get('USER', '')}", "format=account"],
            capture_output=True, text=True,
        )
        accounts = [a.strip() for a in result.stdout.strip().split("\n") if a.strip()]
        if not accounts:
            print("Error: no Slurm account found. Pass --account explicitly.")
            print("  sacctmgr -nP show assoc where user=$(whoami) format=account")
            return
        args.account = accounts[0]
        if len(accounts) > 1:
            print(f"  Multiple accounts found: {accounts}, using '{args.account}'")
        else:
            print(f"  Using account: {args.account}")

    script = SBATCH_TEMPLATE.format(
        job_name=args.job_name,
        max_array_idx=args.num_tasks - 1,
        log_dir=log_dir,
        output_dir=output_dir,
        time_limit=args.time_limit,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        partition=args.partition,
        account=args.account,
        total_scenes=total,
        num_tasks=args.num_tasks,
        scene_list=scene_list,
        project_root=project_root,
        workers=args.workers,
    )

    if args.dry_run:
        print("\n--- sbatch script ---")
        print(script)
        print("--- end ---")
        print(f"\nOutput dir: {output_dir}")
        print("Re-run without --dry-run to submit.")
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", prefix="collision_", delete=False
    ) as f:
        f.write(script)
        sbatch_path = f.name

    print(f"\nSubmitting {sbatch_path} ...")
    result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}")
    else:
        print(f"\nOutput dir: {output_dir}")
        print(f"Logs: {log_dir}/slurm-<jobid>_<task>.out")
        print(f"After completion: uv run python {__file__} --merge --output-dir {output_dir}")


if __name__ == "__main__":
    main()
