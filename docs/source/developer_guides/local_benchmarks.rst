.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Local Demo Benchmarks
=====================

``flashdreams-benchmark`` is the local-first harness for repeatable demo and
runner measurements. It launches existing commands, captures logs, keeps MP4s
and runner stats in one run directory, normalizes metrics to JSON/CSV, and
writes a simple HTML report. It does not gate CI yet.

List available scenarios:

.. code-block:: bash

   uv run flashdreams-benchmark --list-scenarios

Run one scenario:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario self-forcing-taehv-smoke \
     --output-dir artifacts/benchmarks/self-forcing-taehv

The scenario command still owns its runtime requirements. Public runner
scenarios need the usual runner extras, host FFmpeg for MP4 writing, GPU access,
and any checkpoints or input assets that runner would normally resolve.

Artifacts
---------

Each run writes:

* ``manifest.json`` with scenarios, commands, artifacts, and environment data.
* ``environment.json`` with git, Python, Torch, CUDA, GPU, and selected env metadata.
* ``metrics.ndjson`` with raw normalized metric records.
* ``metrics.csv`` for spreadsheet inspection.
* ``report.html`` with run metadata, model summary cards, scenario status, and
  startup/wall-time highlights, plus ``reports/<model>.html`` detail pages with
  unit-aware metric summaries, simple median charts, logs, video links, and
  side-by-side baseline/candidate MP4 previews when quality comparison is
  enabled.
* ``scenarios/<id>/`` with command logs, MP4s, ``stats_*.json``, and compact quality outputs.

The CLI prints the scenario id, command, log path, and a periodic heartbeat
while a command is still running. Child process stdout/stderr is still written
to ``scenarios/<id>/command.log`` so long runs remain easy to inspect without
mixing model logs into the terminal.

Custom Scenarios
----------------

Use a JSON scenario file when a demo is not a public ``flashdreams-run`` preset
or when a local command needs private checkpoint paths. Commands are argv lists,
not shell strings. The harness expands ``{repo_root}``, ``{run_root}``,
``{output_dir}``, ``{scenario_id}``, ``{log_path}``, and ``{env:NAME}``.
Set ``report_group`` to control which ``reports/<group>.html`` detail page a
scenario appears on. It can be a string id, or an object with ``id`` and
``name`` when the display name should differ from the filename.
If ``report_group`` is omitted, the report falls back to the first ``-``
separated segment of the scenario id.

.. code-block:: json

   {
     "schema_version": 1,
     "scenarios": [
       {
         "id": "my-runner-smoke",
         "name": "My runner smoke",
         "report_group": {
           "id": "my-model",
           "name": "My Model"
         },
         "command": [
           "flashdreams-run",
           "self-forcing-wan2.1-t2v-1.3b-taehv",
           "--total-blocks",
           "8"
         ],
         "warmup_steps": 1
       }
     ]
   }

Run it with:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file local_scenarios.json \
     --scenario my-runner-smoke

One-Minute Demo Suite
---------------------

``configs/one_minute_demo_benchmarks.json`` contains manual GPU scenarios for
the real LingBot and Omnidreams demos. Run both:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/one_minute_demo_benchmarks.json \
     --scenario lingbot-world-fast-taehv-one-minute \
     --scenario omnidreams-sv-one-minute \
     --output-dir artifacts/benchmarks/one-minute-demos

Or run one scenario at a time:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/one_minute_demo_benchmarks.json \
     --scenario omnidreams-sv-one-minute

The LingBot and Omnidreams runner scenarios request enough AR blocks for about
one minute of MP4 output. Those runners stop early if the selected conditioned
input stream is shorter than the requested duration. ``interactive-drive`` is
left out of this shipped MP4 suite for now because its public CLI is a live
presenter rather than a file-writing runner.

Quality Hooks
-------------

For local MP4 regression checks, first keep a known-good benchmark run. The
recommended suite includes 30-second seeded clips for quality comparison and
one-minute clips for performance metrics plus manual video review. The commands
below intentionally list scenario ids instead of using ``--all``; ``--all``
also includes built-in smoke scenarios that are not part of this LingBot and
Omnidreams quality workflow.

Standard Quality And Review Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this four-scenario run for standard local LingBot and Omnidreams benchmark
results: 30-second seeded clips for baseline quality metrics, one-minute clips
for performance and manual review, and PAI-Bench-Long scores using the default
PAI-Bench dimension set. PAI-Bench profiles are only attached to selected
scenarios tagged ``one-minute`` or ``pai-bench``; in this standard suite the
30-second seeded quality clips run baseline comparison only, while the
one-minute review clips get PAI-Bench scores.

PAI-Bench and its Python dependencies are not FlashDreams dependencies. Before
running the PAI-Bench quality profile, create or select a separate evaluator
environment whose dependencies and licenses have been reviewed for your use
case.

The following setup was used for the local PAI-Bench runs. Run it from the
FlashDreams repository root. If the virtual environment already exists, ``uv
venv`` asks whether to replace it:

.. code-block:: bash

   uv sync --python 3.12 \
     --package flashdreams \
     --no-dev \
     --group cuda13 \
     --extra runners

   uv venv --python 3.12 ~/.venvs/flashdreams-paibench

   export PAI_BENCH_PYTHON="$HOME/.venvs/flashdreams-paibench/bin/python"

   uv pip install --python "$PAI_BENCH_PYTHON" \
     torch torchvision \
     opencv-python-headless \
     omegaconf \
     openai-clip \
     "pyiqa>=0.1.15,<0.1.16" \
     "setuptools<81"

   "$PAI_BENCH_PYTHON" -c "import clip, cv2, omegaconf, pyiqa, torch; print('PAI-Bench environment OK')"

First create the baseline:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/deterministic_quality_benchmarks.json \
     --scenario lingbot-world-fast-taehv-quality-smoke \
     --scenario omnidreams-sv-ci-quality-smoke \
     --scenario lingbot-world-fast-taehv-one-minute-review \
     --scenario omnidreams-sv-one-minute-review \
     --quality-profile pai-bench-long \
     --pai-bench-python "$PAI_BENCH_PYTHON" \
     --output-dir artifacts/benchmarks/quality-and-review-baseline

Then compare a later candidate run against that baseline:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/deterministic_quality_benchmarks.json \
     --scenario lingbot-world-fast-taehv-quality-smoke \
     --scenario omnidreams-sv-ci-quality-smoke \
     --scenario lingbot-world-fast-taehv-one-minute-review \
     --scenario omnidreams-sv-one-minute-review \
     --quality-profile pai-bench-long \
     --pai-bench-python "$PAI_BENCH_PYTHON" \
     --quality-baseline-dir artifacts/benchmarks/quality-and-review-baseline \
     --output-dir artifacts/benchmarks/quality-and-review-candidate

The two-scenario one-minute suite in
``configs/one_minute_demo_benchmarks.json`` remains useful for
performance-only or manual-review runs, but it is not recommended as the
baseline quality comparison workflow because long generated clips drift more.

The built-in comparison is non-gating in this local-developer version: it
does not change the scenario pass/fail status. It compares the first MP4
artifact for each matching scenario id that has ``quality_baseline_compare``
enabled and writes metrics under
``scenarios/<id>/quality/baseline-clip-compare/``. The one-minute review
scenarios in the shipped quality suite disable baseline scoring so their known
run-to-run drift does not pollute the quality summary; when a matching baseline
MP4 exists, their baseline and candidate MP4s still appear side by side in the
relevant ``reports/<model>.html`` detail page for manual review.

The report focuses on a small cross-demo set:

* ``quality_score``: 0-1 blended score combining baseline similarity with
  visual sanity checks. Higher is better.
* ``quality_similarity_score``: 0-1 score for closeness to the baseline MP4,
  using SSIM, RMSE, PSNR, and frame-count agreement. Higher is better.
* ``quality_visual_sanity_score``: 0-1 no-reference guardrail for obvious
  blank, flat, heavily striped, or unstable clips. Higher is better.
* ``quality_temporal_score``: 0-1 frame-to-frame stability proxy. Higher is
  better.
* ``quality_ssim_score``, ``quality_rmse``, ``quality_mean_abs``, and
  ``quality_psnr_db``: supporting comparison metrics. RMSE and mean absolute
  difference are in 8-bit pixel units; PSNR is in dB.

The HTML report includes rough interpretation bands for the common comparison
metrics. They are meant for local debugging, not pass/fail decisions: score
metrics near 1.0 are better, RMSE or mean absolute difference near 0 is better,
RMSE below about 15 is usually a small pixel-space difference, RMSE above about
40 is usually large visual drift, PSNR above about 30 dB is usually close, and
PSNR below about 20 dB is usually a large difference. Always compare runs with
the same scenario, seed, input assets, and ``--quality-compare-region``.

Determinism and Seeded Quality Checks
-------------------------------------

The one-minute demo scenarios are performance and visual-inspection runs, not
strict pixel-stability gates. A fixed diffusion seed is necessary but not
sufficient for long generated MP4s: CUDA kernel choices, compile paths, and
long autoregressive accumulation can still produce visible drift. The
Omnidreams one-minute scenarios use the same stable non-perf runner as the
quality smoke scenario because broader VAE compile/autotune paths can be
hardware sensitive on local developer systems.

For a stronger same-seed quality signal, use the 30-second quality scenarios in
``configs/deterministic_quality_benchmarks.json``. Its Omnidreams quality
scenario mirrors the existing same-seed CI setup: non-perf runner, fixed
example input, explicit seed, deterministic cuBLAS workspace, stable CUDA
allocator behavior, ``torch.use_deterministic_algorithms(..., warn_only=True)``,
and a roughly 30-second rollout. The LingBot quality scenario uses the same
local strict launcher and explicit seed, but it is currently a local signal
rather than a CI-backed bitwise reproducibility guarantee.

Scenario duration is approximate and comes from the runner command's
``--total-blocks`` value in the JSON file, not from a separate duration field.
The runner can still stop early if the selected conditioning stream is shorter
than the requested block count.

The CLI's default ``--quality-compare-region scenario-default`` uses each
quality scenario's configured compare region. In this suite LingBot compares
the full generated MP4, while Omnidreams compares the generated lower half of
its HDMap/RGB stacked output. Pass ``--quality-compare-region full`` or
``--quality-compare-region bottom-half`` only when you want to override those
scenario defaults.

Detailed component values and per-sampled-frame measurements remain in the
quality JSON under ``diagnostics`` for debugging, but they are not promoted as
top-level report metrics. Use ``--quality-frame-indices`` for an explicit frame
list or ``--quality-sample-count`` for evenly spaced samples. For stacked MP4s
where only the generated lower half should be compared outside a shipped
scenario default, pass ``--quality-compare-region bottom-half``. The comparison
reads MP4s through the existing clip-comparison utilities, so run it from an
environment with the runner/dev media dependencies installed.
``--quality-compute-flip`` adds FLIP metrics when ``flip-evaluator`` is
installed; if it is missing, the hook logs a warning and still reports the
other quality metrics.

Quality evaluators are optional command hooks. They run after the scenario and
do not change the scenario pass/fail status in this local-developer version.
Write a JSON metrics file and point ``metrics_path`` at it.

.. code-block:: json

   {
     "id": "generated-quality",
     "command": [
       "python",
       "-m",
       "my_quality_tool",
       "--video",
       "{first_video}",
       "--out",
       "{quality_dir}/metrics.json"
     ],
     "metrics_path": "{quality_dir}/metrics.json"
   }

Optional PAI-Bench-Long Profile
-------------------------------

The benchmark CLI can also append a named PAI-Bench-Long quality profile to the
selected scenarios. This is intended for longer MP4 review clips, such as the
one-minute LingBot and Omnidreams scenarios. It stages the first generated MP4,
splits it into local segments, runs public PAI-Bench-G on those segments, and
reports 0-100 ``pai_bench_long_*`` metrics in the HTML report. These metrics
are non-gating and separate from the baseline similarity metrics.

The adapter follows the same external-evaluator pattern as WorldLens: it owns
the checkout/staging/summary glue, while PAI-Bench itself remains an external
checkout. By default it clones
``https://github.com/SHI-Labs/physical-ai-bench.git`` at the pinned revision
declared in ``tools.benchmarks.pai_bench_profile`` under
``<repo-root>/.cache/flashdreams/evaluators/physical-ai-bench`` when the
checkout is missing, so syncing ``artifacts/`` does not include the evaluator
checkout or its environment.

The default ``--pai-bench-runner local`` mode runs the public PAI-Bench
entrypoint with the selected evaluator Python and injects a FlashDreams
OpenCV-backed ``decord`` compatibility shim. This avoids the upstream
``decord`` wheel resolution failure on aarch64. Pass ``--pai-bench-python`` to
point at a separately prepared evaluator environment; PAI-Bench dependencies are
not declared in FlashDreams because they are evaluator-only and have separate
license terms.

Before launching ``torch.distributed.run``, the adapter imports the requested
PAI-Bench dimension modules with the same Python and ``PYTHONPATH`` that the
evaluator will use. Missing imports are reported in
``scenarios/<id>/quality/<profile>/pai_bench_preflight.log`` and in the hook's
``metrics.json``. For example, a missing ``clip`` import means the command was
not run with an evaluator Python that contains the requested PAI-Bench
dependencies.

For baseline/candidate copy-paste commands that include PAI-Bench, use the
``Quality Hooks`` section above. For ad hoc one-off evaluation, run one-minute
local clips with the PAI-Bench-Long profile. These ad hoc commands are useful
for debugging, but they do not produce the standard baseline/candidate report:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/one_minute_demo_benchmarks.json \
     --scenario lingbot-world-fast-taehv-one-minute \
     --scenario omnidreams-sv-one-minute \
     --quality-profile pai-bench-long \
     --pai-bench-python "$PAI_BENCH_PYTHON" \
     --output-dir artifacts/benchmarks/one-minute-pai-bench

Use ``--pai-bench-python`` to point at a curated evaluator Python that contains
the dependencies for the requested dimensions. On aarch64, avoid
``--pai-bench-python "uv run python"`` with the upstream PAI-Bench checkout
unless that checkout has already been patched or locked for aarch64-compatible
dependencies.

For one-off checks, the profile also supports public PAI-Bench-G on the full
MP4 without local segmentation. This is also not the standard local benchmark
workflow:

.. code-block:: bash

   uv run flashdreams-benchmark \
     --scenario-file configs/one_minute_demo_benchmarks.json \
     --scenario lingbot-world-fast-taehv-one-minute \
     --quality-profile pai-bench-g \
     --pai-bench-python "$PAI_BENCH_PYTHON" \
     --output-dir artifacts/benchmarks/lingbot-pai-bench-g

Pass ``--pai-bench-runner upstream`` only when you intentionally want the old
upstream execution mode, for example on an x86_64 machine with a working
PAI-Bench environment.

By default, the adapter removes copied or segmented MP4 inputs from
``scenarios/<id>/quality/<profile>/staged/videos`` after PAI-Bench finishes.
The durable benchmark artifacts are the original scenario MP4s, logs,
``metrics.json``, ``staged_inputs.json``, the prompt JSON, and parsed evaluator
result JSON. Pass ``--pai-bench-keep-staged-videos`` only when debugging the
external evaluator input staging.

The default dimensions are the non-I2V PAI-Bench quality dimensions:
``aesthetic_quality``, ``background_consistency``, ``imaging_quality``,
``motion_smoothness``, ``overall_consistency``, and
``subject_consistency``. Pass ``--pai-bench-dimension`` multiple times to run a
smaller or custom dimension set for local triage only; doing so changes the
reported PAI-Bench metric set, so those results should not be compared against
standard runs. ``--pai-bench-segment-duration-s`` controls the local long-video
segment size; it defaults to 10 seconds. Pass ``--no-pai-bench-fetch`` to
avoid fetching an existing git checkout.

Custom scenarios can opt into PAI-Bench profiles by including either
``one-minute`` or ``pai-bench`` in their ``tags`` list. This keeps PAI-Bench
off short deterministic quality-smoke scenarios, where short or stacked clips
can produce noisy evaluator failures and crowded reports.

Internal Cosmos-Interactive Example
-----------------------------------

The public harness does not import ``internal/`` modules, but it can run the
GitLab-only cosmos-interactive MP4 renderer after this work is cherry-picked.

.. code-block:: json

   {
     "schema_version": 1,
     "scenarios": [
       {
         "id": "cosmos-interactive-mp4-12f",
         "name": "Cosmos interactive MP4 12-frame window",
         "report_group": {
           "id": "cosmos-interactive",
           "name": "Cosmos Interactive"
         },
         "command": [
           "uv",
           "run",
           "--no-sync",
           "python",
           "-m",
           "cosmos_interactive_internal.generate_native_video",
           "{env:CAMERA_CKPT}",
           "--t2v-checkpoint",
           "{env:T2V_CKPT}",
           "--init-image",
           "{env:DAYDREAM_INIT_IMAGE}",
           "--out",
           "{output_dir}/cosmos-interactive.mp4",
           "--stats-out",
           "{output_dir}/stats_cosmos_interactive.json",
           "--num-chunks",
           "8",
           "--chunk-size",
           "4",
           "--window-frames",
           "12",
           "--kv-cache-mode",
           "block",
           "--overlap-cache-update",
           "--compile-denoise",
           "--auto-compile-prewarm",
           "--cuda-graph-denoise",
           "--perf-summary",
           "--perf-warmup-steps",
           "1",
           "--perf-summary-window",
           "4"
         ],
         "output_dir_arg": null,
         "warmup_steps": 1,
         "timeout_s": 7200
       }
     ]
   }
