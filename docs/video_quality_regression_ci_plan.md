<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Video Quality Regression CI Plan

## Context

The public FlashDreams repository currently has GitHub Actions coverage for CPU tests, GPU tests, docs, REUSE compliance, merge queue checks, and PyPI publishing. There is no public video-quality regression workflow yet. The new system should run in this public GitHub repo, use Hugging Face for evaluation assets, and keep the design small enough to be reliable before it becomes broad.

Relevant current CI behavior:

| Workflow | Current role | Design implication |
|---|---|---|
| `.github/workflows/ci.yml` | Runs `cpu` and `gpu` jobs on `main`, `pull-request/[0-9]+`, tags, and `merge_group`; GPU job already receives `HF_TOKEN`. | Add video regression as a separate GPU workflow or job with matching trusted triggers and a required check name. |
| `.github/workflows/doc.yml` | Builds docs on PR/merge queue and publishes GitHub Pages on trusted events. | The review app can use GitHub Pages, but publishing should happen only from trusted events. |
| `.github/workflows/reuse-lint.yml` | Enforces licensing/compliance. | New scripts need SPDX headers; docs can use the existing documentation license convention. |

## Goals

- Detect obvious generated-video regressions before they merge.
- Start by preparing the input data and proving the metrics catch known bad cases.
- Keep the first metric set small, deterministic, and easy to debug.
- Store ground-truth golden clips and known-good inferenced clips in Hugging Face datasets, not S3.
- Keep generated CI outputs as GitHub Actions artifacts, with optional internal mirroring for serving and long-term triage.
- Provide a path to add more metrics later, including a VLM-based evaluator.
- Block merge when the per-commit regression job fails.
- Alert when the nightly regression job fails.

## Non-Goals

- Replace human review for model quality.
- Run expensive distribution metrics on every commit.
- Depend on internal-only S3 for public CI correctness.
- Make VLM judging a blocking metric in the first version.

## Proposed Architecture

Use a manifest-driven harness with public-repo paths:

| Path | Purpose |
|---|---|
| `configs/video_quality_cases.yml` | Versioned case catalog: asset refs, golden refs, generation config, metric thresholds, and suite membership. |
| `tools/video_quality/run_regression.py` | Downloads assets, runs generation, computes metrics, checks thresholds, and writes a run manifest. |
| `tools/video_quality/metrics.py` | Small deterministic metric implementations. |
| `tools/video_quality/evaluators/` | Optional extension point for learned/VLM evaluators. |
| `.github/workflows/video-quality-regression.yml` | Per-commit, nightly, calibration, and manual workflows. |
| `docs/video_quality_regression_ci_plan.md` | This design plan. |
| `docs/source/_static/video_quality/` | Optional static review-app assets if the app is published with docs/GitHub Pages. |

The runner should produce one `manifest.json` per run with:

- GitHub metadata: commit SHA, ref, event name, run ID, attempt, job name, actor.
- Case metadata: case ID, suite, Hugging Face asset revision, golden revision, generation config.
- Output paths: generated video, sampled frames, event/tail clips, metrics JSON.
- Metric values: per-case summary plus selected per-window values.
- Decision: pass/fail, failed thresholds, severity, whether the case gates merge.

## Phase 1: Data Prep and Metric Calibration

The first phase should not be a blocking CI job. It should answer: do the proposed metrics separate known-good output from the regressions we care about?

Deliverables:

| Deliverable | Details |
|---|---|
| Regression fixture inventory | For each known regression, collect input conditioning data, prompt/config, known-good inferred video, known-bad inferred video, and failure-window labels. |
| Hugging Face dataset layout | Upload inputs, ground-truth clips where available, known-good inferenced clips, and known-bad calibration clips. |
| Metric calibration report | Run the candidate metrics on good and bad clips, record distributions, choose thresholds, and mark flaky/weak metrics. |
| Synthetic metric tests | Tiny generated videos for grey, blank, blurry, and striped patterns so metric code has unit tests independent of model inference. |
| Initial case manifest | Add cases in `calibration` suite first; promote only the stable subset to `per_commit`. |

Exit criteria:

- Each historical regression has at least one metric with clear separation from known-good output.
- Thresholds are based on recorded calibration data, not guesses.
- The public CI harness can run in dry-run/evaluate-only mode without model inference.
- No case is promoted to blocking until it has passed several trusted runs without flaking.

## Hugging Face Data Storage

Use a Hugging Face dataset as the public source of truth for quality-regression assets. Suggested dataset: `nvidia/flashdreams-video-quality-regression` or a gated equivalent if the clips cannot be fully public.

Suggested layout:

```text
cases/
  <case_id>/
    inputs/
      first_frame.png
      conditioning.mp4
      prompt.txt
      metadata.yml
    golden/
      ground_truth.mp4
      known_good/
        <golden_version>/
          output.mp4
          manifest.json
    calibration/
      known_bad/
        <regression_id>/
          output.mp4
          labels.yml
index/
  cases.yml
  golden_versions.yml
  calibration_runs.yml
```

Storage rules:

| Asset | Storage | Notes |
|---|---|---|
| Input conditioning videos | Hugging Face dataset | Pin by dataset revision and include sha256 in the repo manifest. |
| Prompts/config metadata | Repo manifest plus Hugging Face copy | Repo manifest is the reviewed source; HF copy keeps assets self-describing. |
| Ground-truth golden clips | Hugging Face dataset | Used for full-reference or visual comparison when real GT exists. |
| Known-good inferenced clips | Hugging Face dataset | Baseline for visual comparison and relative regression checks. |
| Known-bad calibration clips | Hugging Face dataset, calibration-only | Used to validate metrics, not as production golden. |
| Per-commit outputs | GitHub Actions artifacts | Short-lived, tied to the run that generated them. |
| Nightly outputs | GitHub Actions artifacts plus optional HF upload for promoted baselines | Do not make every nightly output permanent by default. |
| Internal serving mirror | Periodic sync from Hugging Face/GitHub artifacts to internal S3 | Mirror is for web-app serving/triage convenience, not the public source of truth. |

## Case Manifest

The public repo manifest should reference Hugging Face assets and keep thresholds reviewable:

```yaml
cases:
  - id: long_rollout_blur_grey
    description: Tail of a long rollout should not collapse into grey blur.
    suites: [calibration, nightly]
    hf_dataset: nvidia/flashdreams-video-quality-regression
    hf_revision: 2026-06-03-v1
    assets:
      first_frame: cases/long_rollout_blur_grey/inputs/first_frame.png
      conditioning: cases/long_rollout_blur_grey/inputs/conditioning.mp4
      prompt: cases/long_rollout_blur_grey/inputs/prompt.txt
      known_good: cases/long_rollout_blur_grey/golden/known_good/v1/output.mp4
      ground_truth: null
      sha256:
        conditioning: "<sha256>"
        known_good: "<sha256>"
    generation:
      runner: omnidreams
      seed: 42
      duration_s: 120
      extra_args: []
    windows:
      head: {start_s: 0, end_s: 10}
      tail: {start_s: 90, end_s: 120}
    thresholds:
      - metric: sharpness_tail_head_ratio
        op: ">="
        value: 0.55
        severity: critical
```

Suggested suite labels:

| Suite | Meaning |
|---|---|
| `calibration` | Metric-development suite. Does not block. Includes known-good and known-bad clips. |
| `per_commit` | Blocking suite for trusted PR branches and merge queue. Small and stable. |
| `nightly` | Scheduled broader suite. Alerts on failure. |
| `quarantine` | New/flaky cases that collect data but do not gate. |
| `vlm_experimental` | Optional evaluator pass for semantic/subjective failures. Non-blocking at first. |

## Small Metric Set

The first implementation should use four core metrics. These are cheap, deterministic, interpretable, and target the failures that have already escaped.

| Metric | Captures | Output values |
|---|---|---|
| Decode/metadata validity | Missing, corrupt, too short, wrong FPS, wrong resolution, wrong duration | `decode_ok`, `frame_count`, `fps`, `duration_s`, `height`, `width` |
| Grey/blank score | All-grey frames, empty output, long-tail grey collapse | `luma_std`, `rgb_channel_std`, `saturation_mean`, `grey_pixel_ratio` |
| Sharpness score | Blur, blob collapse, long-tail degradation | `laplacian_variance`, `high_frequency_energy`, `sharpness_tail_head_ratio` |
| Stripe score | Horizontal/vertical strips, scanline artifacts, periodic bands | `fft_axis_energy_ratio`, `row_autocorr_peak`, `col_autocorr_peak` |

Optional metrics can be added behind evaluator plugins:

| Evaluator | Initial status | Use case |
|---|---|---|
| Reference similarity | Calibration/nightly | Compare against ground-truth or known-good clips with SSIM/LPIPS/DINO when deterministic reference is meaningful. |
| Temporal continuity | Nightly/quarantine | Detect flicker, duplicated frames, frozen video, or stutter. |
| VLM evaluator | Experimental/non-blocking | Ask a VLM to score semantic failures such as prompt mismatch, off-road behavior, collision aftermath, or obvious visual artifacts. |
| Distribution metrics | Nightly only | FVD/KVD or similar broad trend metrics across many clips. |

The evaluator interface should be simple:

```python
class VideoQualityEvaluator:
    name: str

    def evaluate(self, case, generated_video, assets, windows) -> dict:
        """Return scalar metrics and optional per-window details."""
```

Blocking thresholds should initially reference only the four core metrics. VLM outputs should be logged, trended, and shown in the review app before they become gating.

## Regression Case Matrix

| Case | Failure signature | First metrics to use | Testing data needed | First suite |
|---|---|---|---|---|
| Long rollout turns blurry and grey | First frames look OK, tail after 90s+ loses texture/color | Grey/blank score by window; sharpness tail/head ratio | Long conditioning clip, known-good inferred video, known-bad calibration video, tail labels | Calibration, then Nightly |
| All frames are empty or grey | Output decodes but all frames are grey or blank | Decode/metadata; grey/blank score; min sharpness | Short canonical input, known-good inferred video, known-bad grey output | Calibration, then Per-commit |
| Driving into walls/objects creates blurry blobs | Nominal route is fine, off-nominal post-event frames collapse | Event-window sharpness; grey/blank score; optional VLM note | Paired nominal/off-nominal input, event timestamp, known-good and known-bad videos | Calibration, then one Per-commit sentinel plus Nightly expansion |
| Obvious strip patterns | Repeated horizontal/vertical bands or scanlines | Stripe score; sharpness as secondary | Short clip/config known to expose striping, synthetic striped calibration clips | Calibration, then Per-commit |
| Corrupt or short output | Generation exits but artifact is missing, unreadable, or wrong length | Decode/metadata | Any generated case | Per-commit |
| Color/exposure collapse | Output too dark, overexposed, desaturated, or washed out | Grey/blank score; optional reference similarity | Known-good inferred clip and generated output | Nightly, promote if stable |
| Temporal flicker/freeze | Frames duplicate, freeze, or alternate quality | Optional temporal continuity evaluator | Motion-rich clip and known-good inferred clip | Nightly/quarantine |
| Prompt/conditioning ignored | Output looks plausible but semantically wrong | Optional VLM evaluator; reference similarity if available | Strongly constrained prompt/conditioning cases | VLM experimental |

## GitHub Actions Design

Add `.github/workflows/video-quality-regression.yml`.

Per-commit job:

- Trigger on trusted refs matching the current CI style: `push` to `main`, `push` to `pull-request/[0-9]+`, and `merge_group`.
- Run on a GPU runner.
- Use `HF_TOKEN: ${{ secrets.HF_TOKEN }}` to read gated Hugging Face datasets if needed.
- Run only cases marked `per_commit`.
- Upload `manifest.json`, metric JSON, frames, and generated videos via `actions/upload-artifact`.
- Fail the job on critical threshold failures.
- Configure branch protection/merge queue to require the check.

Nightly job:

- Trigger on `schedule` and `workflow_dispatch`.
- Run cases marked `nightly`.
- Use `continue-on-error: false` for the evaluation job, then run a notification step/job with `if: failure()`.
- Send Slack alert through a secret webhook or GitHub-to-Slack integration.
- Include failed case IDs, commit SHA, GitHub Actions run URL, artifact URL, and dashboard URL.

Calibration job:

- Trigger on `workflow_dispatch`.
- Supports `case_id`, `hf_revision`, `suite`, and `evaluate_only` inputs.
- Can run against known-good and known-bad clips without model inference.
- Writes calibration reports as artifacts and optionally uploads blessed reports to Hugging Face.

Fork safety:

- Do not require this job for arbitrary external `pull_request` events unless the assets are public and no secrets are required.
- For gated HF datasets, run on trusted PR mirror refs or merge queue events where secrets are available.

## Review App

Phase 1 can be a generated static HTML report artifact. Later, publish a richer app through GitHub Pages.

Minimum UI:

| View | Capability |
|---|---|
| Run overview | Show failed cases first with threshold reasons. |
| Case comparison | Compare generated output against known-good inferred clip and ground truth when available. |
| Video review | Side-by-side playback, frame stepping, and jump buttons for head/tail/event windows. |
| Metric timeline | Plot the four core metrics by frame/window with thresholds. |
| Artifact links | Link to GitHub Actions artifacts and Hugging Face golden clips. |
| VLM panel | Show non-blocking VLM text verdicts and confidence once enabled. |

Serving model:

- Public source of truth: Hugging Face dataset for golden and known-good media.
- CI output: GitHub Actions artifacts for generated clips.
- Optional internal mirror: a scheduled internal sync can copy HF assets and selected GitHub artifacts to internal S3 so an internal dashboard can serve videos quickly and persist failed outputs.

## Implementation Milestones

| Phase | Deliverables | Exit criteria |
|---|---|---|
| 1. Prepare data and calibrate metrics | HF dataset layout, known-good/known-bad clips, case manifest in `calibration`, synthetic metric tests, calibration report | Metrics show clear separation for the listed regressions. |
| 2. Build harness | `tools/video_quality` runner, four core metrics, artifact manifest, evaluate-only mode | Local and GitHub manual runs produce comparable manifests. |
| 3. Add per-commit gate | `per_commit` workflow, small stable case set, artifact report, required check | Known-bad calibration cases fail; blessed known-good cases pass. |
| 4. Add nightly alerting | Scheduled workflow, broader case set, Slack alert, retained artifacts | Nightly failure sends actionable alert with report links. |
| 5. Build review app | Static report, then GitHub Pages app if needed | Reviewer can compare generated vs known-good/GT without manual downloads. |
| 6. Add extensible evaluators | Reference similarity, temporal continuity, VLM experimental evaluator | Non-core metrics are logged and trended before any gating decision. |

## First Blocking Suite Proposal

Start with only the cases that the four core metrics can capture reliably:

| Case | Data | Blocking checks |
|---|---|---|
| Short canonical smoke | One short HF input plus known-good inferred clip | Decode/metadata, grey/blank, sharpness, stripe score. |
| Empty/grey sentinel | Same input with known-bad calibration output for metric tests; generated output in CI | Grey-pixel ratio, luma/RGB std, saturation. |
| Striping sentinel | Short input known to expose striping plus synthetic striped tests | FFT axis energy and row/column autocorrelation. |
| Off-nominal sentinel | One wall/object trajectory with event timestamp | Post-event sharpness and grey/blank checks. |

Keep long 90s+ rollout in nightly until the runtime and thresholds are stable.
