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
