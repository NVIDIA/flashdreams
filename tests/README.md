# flashdreams test runners

Two entrypoints for running the flashdreams test suite. Pick the one that matches your dev setup.

| Script | Audience | What it does |
| --- | --- | --- |
| [`run_tests_local.sh`](./run_tests_local.sh) | dev already inside a container with deps installed | Just discovers tests and runs `pytest`. No install, no container. |
| [`run_tests_docker.sh`](./run_tests_docker.sh) | local machine with GPU + docker | `docker run` → install deps → run `pytest`. |

Both scripts resolve paths relative to their own location and can be invoked from anywhere.

`run_tests_docker` dispatches internally, after establishing the environment, to `run_tests_local`.

## Quick examples

```bash
# Already inside a dev container (your venv has flashdreams[dev] + integrations)
./tests/run_tests_local.sh
./tests/run_tests_local.sh flashdreams/tests/test_attention.py

# Local machine with docker + GPU
./tests/run_tests_docker.sh
./tests/run_tests_docker.sh flashdreams/tests/test_attention.py
```

## What gets run

When no `TEST_TARGET` is given, each script performs global discovery of `**/test_*.py`:

Pytest is invoked with `-m "not manual"` so any test marked `@pytest.mark.manual`
is skipped.

## Omnidreams quality regression

`integrations/omnidreams/tests/test_quality_regression.py` is the golden-clip
gate for generated driving video. It is marked `ci_gpu` and skips in local
runs until a reference clip and deterministic input assets are provided.

For PR gating, prefer a short clip: `TOTAL_BLOCKS=4` on the default chunk2
runner produces roughly one second at 30fps. A longer 5 second clip is better
as a nightly/manual check because it costs more GPU time and is more exposed to
small non-deterministic drift.

The test needs two things:

- A reference clip containing generated frames only. In CI this is
  `reference_compare_region.mp4`.
- The same deterministic rollout inputs used to create the reference. For the
  default single-view gate, use the public Omnidreams example data UUID
  `239560dc-33d1-11ef-9720-00044bcbccac`; for custom data, provide matching
  `HDMAP_VIDEO_PATHS` and `FIRST_FRAME_PATHS`.

Minimum single-view setup with explicit local inputs:

```bash
export FLASHDREAMS_OMNIDREAMS_QUALITY_REFERENCE_CLIP=/abs/path/reference.mp4
export FLASHDREAMS_OMNIDREAMS_QUALITY_HDMAP_VIDEO_PATHS=/abs/path/hdmap.mp4
export FLASHDREAMS_OMNIDREAMS_QUALITY_FIRST_FRAME_PATHS=/abs/path/first_frame.png
uv run pytest integrations/omnidreams/tests/test_quality_regression.py -v
```

To use the runner's public single-view example data instead of supplying
`HDMAP_VIDEO_PATHS` and `FIRST_FRAME_PATHS`, set:

```bash
export FLASHDREAMS_OMNIDREAMS_QUALITY_EXAMPLE_DATA=1
```

The default example UUID is `239560dc-33d1-11ef-9720-00044bcbccac`; override it
with `FLASHDREAMS_OMNIDREAMS_QUALITY_EXAMPLE_DATA_UUID=<uuid>` if you want a
different sample from `nvidia/omni-dreams-samples`.

To generate a new short reference candidate from the default public example
data, run on a CUDA machine with `ffmpeg` installed. Export `HF_TOKEN` first if
your environment needs Hugging Face authentication for model or dataset access.

```bash
mkdir -p /tmp/omnidreams_quality_ref /tmp/omnidreams_quality_artifacts

uv run --project integrations/omnidreams flashdreams-run \
  omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae \
  --example-data True \
  --example_data_uuid "239560dc-33d1-11ef-9720-00044bcbccac" \
  --total-blocks 4 \
  --output-dir /tmp/omnidreams_quality_ref
```

That writes the normal Omnidreams runner MP4, with HDMap condition stacked above
generated frames:

```text
/tmp/omnidreams_quality_ref/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae.mp4
```

Run the quality test once against that full MP4 to create the exact comparison
artifact:

```bash
FLASHDREAMS_OMNIDREAMS_QUALITY_REFERENCE_CLIP=/tmp/omnidreams_quality_ref/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae.mp4 \
FLASHDREAMS_OMNIDREAMS_QUALITY_EXAMPLE_DATA=1 \
FLASHDREAMS_OMNIDREAMS_QUALITY_EXTRACT_GENERATED_REGION_FROM_REFERENCE=1 \
FLASHDREAMS_OMNIDREAMS_QUALITY_ARTIFACT_DIR=/tmp/omnidreams_quality_artifacts \
FLASHDREAMS_OMNIDREAMS_QUALITY_TOTAL_BLOCKS=4 \
uv run --project integrations/omnidreams pytest \
  integrations/omnidreams/tests/test_quality_regression.py -v -s
```

After visually inspecting the artifacts, promote this file as the reference
used by CI:

```text
/tmp/omnidreams_quality_artifacts/reference_compare_region.mp4
```

Keep `reference_original.mp4` only as an optional debug artifact. Do not promote
`candidate_original.mp4` or `candidate_compare_region.mp4`; those are outputs
from one test run. Store a small `metadata.json` next to the promoted reference
recording the runner name, example UUID, `total_blocks`, producing commit, and
thresholds.

By default, the candidate MP4 is treated as the normal Omnidreams runner output:
HDMap condition stacked above generated frames. The test extracts the generated
lower half before comparison. The reference should normally be generated frames
only; if you promote the full runner MP4 as the reference, also set
`FLASHDREAMS_OMNIDREAMS_QUALITY_EXTRACT_GENERATED_REGION_FROM_REFERENCE=1`.
Set `FLASHDREAMS_OMNIDREAMS_QUALITY_ARTIFACT_DIR=/abs/path/artifacts` to copy
the original reference/candidate MP4s and the exact comparison-region MP4s to a
stable directory for visual inspection.

If prompt/image embeddings have been precomputed, set
`FLASHDREAMS_OMNIDREAMS_QUALITY_EMBEDDINGS_PATH=/abs/path/embeddings.pt` instead
of `FIRST_FRAME_PATHS`; the test then skips loading the one-shot text/image
encoders. Tune the metric gates with `MAX_MEAN_ABS`, `MAX_RMSE`,
`MIN_PSNR_DB`, `MAX_MEAN_FLIP`, and `MAX_FRAME_FLIP`. Refresh the reference by
running the same config, inspecting the generated MP4, and promoting it to the
reference location.

In GitHub Actions, the GPU job downloads `reference_compare_region.mp4` from a
Hugging Face dataset before running `pytest -m ci_gpu`, then sets
`FLASHDREAMS_OMNIDREAMS_QUALITY_REFERENCE_CLIP`,
`FLASHDREAMS_OMNIDREAMS_QUALITY_EXAMPLE_DATA=1`, and
`FLASHDREAMS_OMNIDREAMS_QUALITY_TOTAL_BLOCKS=4`. The default dataset is
`nvidia/omni-dreams-samples`. Set the repository variable
`OMNIDREAMS_QUALITY_REFERENCE_REPO` if the references need to be read from a
different dataset. Set `OMNIDREAMS_QUALITY_REFERENCE_PATH` if the file path
changes.
The job uploads the generated comparison artifacts as a GitHub Actions artifact
named `omnidreams-quality-artifacts`.

## Shared environment knobs

`run_tests_docker.sh` reads these env vars
(`run_tests_local.sh` ignores them — it doesn't manage caches or images):

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASHDREAMS_TEST_IMAGE` | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` | Container image used for the run. System deps and uv are installed at container start. |
| `FLASHDREAMS_UV_CACHE_DIR` | `${HOME}/.cache/uv` | Host dir mounted to `/root/.cache/uv`. |
| `FLASHDREAMS_HF_CACHE_DIR` | `${HOME}/.cache/huggingface` | Host dir mounted to `/root/.cache/huggingface`. |
| `FLASHDREAMS_CACHE_DIR` | `${HOME}/.cache/flashdreams` | Host dir mounted to `/root/.cache/flashdreams`. |
| `FLASHDREAMS_TRITON_CACHE_DIR` | `${HOME}/.cache/triton` | Host dir mounted to `/root/.cache/triton`; persisted across runs to avoid recompiling Triton kernels (also exported as `TRITON_CACHE_DIR`). |

Each script also has a top-of-file usage block for the full set of CLI flags.
