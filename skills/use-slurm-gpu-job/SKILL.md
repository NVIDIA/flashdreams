---
name: use-slurm-gpu-job
description: Run FlashDreams builds, tests, inference, benchmarks, and other nontrivial commands on the Slurm cluster's 8-GPU compute node through srun.sh instead of burdening the login node. Use whenever work in this FlashDreams checkout may consume meaningful CPU, memory, GPU, compilation time, test time, or download bandwidth, or when attaching to an existing Slurm allocation.
---

# Use a Slurm GPU job

Treat the login node as a control plane. Limit work there to file inspection, search,
small edits, Git metadata, shell syntax checks, and allocation management. Run package
installation, builds, test suites, model loading, generation, benchmarks, and other
heavy commands inside the allocation.

## Start or attach to a job

Always inspect the user's jobs before starting or attaching:

```bash
squeue -u "$USER"
```

If a job is running, reuse its job ID for every command and test in the task. Do not
request a second allocation. From the directory containing the site-provided
`srun.sh` launcher (commonly `$HOME/work`), attach through the script in a PTY:

```bash
cd "${FLASHDREAMS_SLURM_LAUNCH_DIR:-$HOME/work}"
./srun.sh 1 <job-id>
```

If no job is running, start the script in a PTY without a job ID:

```bash
cd "${FLASHDREAMS_SLURM_LAUNCH_DIR:-$HOME/work}"
./srun.sh
```

The script requests one interactive node for four hours with eight GPUs. It mounts the
canonical Lustre checkout at `/workspace/flashdreams` inside the container. It also
keeps the uv environment, caches, Hugging Face data, and Triton cache on Lustre. Do not
replace the script with a raw `srun` command or create a working copy under `/home`.

The site launcher should inspect the current user's queue as a guard. Unless given an
explicit job ID, it should reuse a running job owned by that user and request a new
allocation only when none is running. Inspect the launcher before first use because
accounts, partitions, images, and default checkout paths are site-specific.

The first positional argument is the node count and the second is the job ID. Prefer
one node unless the user explicitly asks for multi-node work. `SLURM_JOB_ID` and
`SLURM_JOB_NUM_NODES` may also supply those values.

When operating through Codex, launch the script with a TTY and a short yield time. Keep
the returned terminal session ID. If the allocation is queued, poll that same session;
do not start duplicate allocations. Once the shell is ready, send every heavy command
and test to the same session for the rest of the task.

## Verify the shell

Before substantive work, verify that execution is on the allocated node and in the
mounted project:

```bash
hostname
pwd
nvidia-smi -L
git rev-parse HEAD
```

Expect `pwd` to be `/workspace/flashdreams` and eight GPUs to be visible. Compare the
commit to the login-node checkout when there is any doubt about the mount. Stop and fix
the mount if it differs; never test a stale checkout.

## Run work

Run commands from `/workspace/flashdreams`. Reuse the allocated terminal for the whole
task, including follow-up tests. Use FlashDreams' narrowest relevant test command first,
then broaden only when useful. Keep large generated data and outputs on Lustre rather
than in `/home`.

Remember that the allocation has a four-hour wall-clock limit. Preserve useful logs or
results before it expires. When finished, send `exit` to release the allocation. Report
whether validation ran on the compute node and name the commands used.
