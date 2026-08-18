# Principle Agent Guide

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size, lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung, a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs (the platform is never the spec ideal, a clock drifts, a sensor reads off), anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

# FlashDreams Agent Guide

FlashDreams is a GPU-heavy inference and serving library for autoregressive video and world models. Default to inspection, docs, config checks, and CPU tests unless the user explicitly asks to run generation or GPU workflows.

Start here, then use the narrower docs for the task in front of you:

- `skills/` contains repo-authored Agent Skills. Use `skills/README.md` for opt-in setup and skill authoring rules.
- `README.md` covers user-facing setup, requirements, supported models, and first-run commands.
- `CONTRIBUTING.md` covers PR process, DCO sign-off, coding conventions, test markers, and dependency rules.
- `docs/README.md` covers Sphinx docs build and hosting. `tests/README.md` covers local, Docker, and CI-oriented test entry points.

## Repo Map

- `flashdreams/flashdreams/core/`: reusable numerical primitives, checkpoint loading, distributed helpers, attention, and I/O. Keep it model-agnostic.
- `flashdreams/flashdreams/infra/`: framework contracts and orchestration for configs, pipelines, encoders, decoders, diffusion models, schedulers, runners, profiling, and CUDA graph wrapping.
- `flashdreams/flashdreams/recipes/`: built-in reusable recipe code such as WAN, Cosmos, TAEHV, and template wiring.
- `flashdreams/flashdreams/configs/`, `plugins/`, and `scripts/`: runner registry, plugin discovery, and CLI entry points.
- `integrations/<name>/`: workspace-member model/plugin packages with their own configs, runners, tests, README files, and `pyproject.toml` entry points.
- `docs/source/`: Sphinx sources for quickstart, models, developer guides, API, and community docs.
- `tests/`: root test helpers plus package/integration tests. Ignore `.claude/worktrees/` when scanning the source tree; those are nested worktree artifacts, not the repo's current source.

## Skill Map

- `skills/flashdreams-integrations`: read before changing recipe/integration architecture, runner registration, pipeline wiring, or model boundaries.
- `skills/integrate-a-model`: read before porting a new external model or reproducing an integration workflow.
- `skills/profile-model-performance`: read before starting performance work on an existing model integration, demo, runner, or serving path; use it to map execution, add stage timings, and identify decode/model/cache/transfer/presentation bottlenecks.
- `skills/apply-inference-optimizations`: read before porting runtime speedups such as bounded K/V caches, overlap, compile, CUDA graphs, decoder layout changes, or presentation queue tuning into an integration.
- `skills/validate-performance-quality`: read before adding benchmark sweeps, quality comparisons, profiler probes, performance summaries, or docs for a performance change.
- `skills/flashdreams-postprocessing`: read before adding or modifying video post-processors, postprocess presets, `VideoPostprocessStream`, buffering/layout behavior, or runner postprocess wiring.
- `skills/python-docstring-style`: read before adding or polishing Python docstrings, field docstrings, module comments, or SPDX headers.
- `skills/maintaining-oss-state`: read before dependency, license, NOTICE, REUSE, or OSS-release collateral changes.
- When adding a new `skills/<skill-name>/SKILL.md`, update this section so agents can discover when to use it.

## Common Commands

```bash
uv sync --extra dev --extra runners
uv sync --package <integration> --extra dev
uv run flashdreams-run --help
uv run flashdreams-run <runner-name> --help
uv run flashdreams-run --no-instantiate <runner-name>
uv run --group lint pre-commit run -a
uv run pytest -m ci_cpu
uv run pytest -m "not manual"
./tests/run_tests_local.sh [target]
./tests/run_tests_docker.sh [target]
uv run --group docs sphinx-build -b html docs/source docs/_build/html
uv run --group docs sphinx-autobuild -E docs/source docs/_build/html --port 8000
```

Use `--no-instantiate` before GPU work to inspect the resolved runner config without constructing models or loading checkpoints.

## No-GPU Workflow

- Inspect available runners with `uv run flashdreams-run --help`, then inspect a specific runner with `uv run flashdreams-run --no-instantiate <runner-name>`.
- Prefer CPU checks first: config imports, checkpoint key-remap shape/bijection tests on CPU or meta tensors, docs builds, `pytest -m ci_cpu`, and static assertions about runner names and pipeline wiring.
- Avoid `ci_gpu`, generation, `torchrun`, Docker GPU tests, large Hugging Face downloads, rollout parity, CUDA graph, WebRTC runtime, and quality-regression tests on CPU-only hosts unless the user requests them or the test explicitly skips cleanly.

## Testing Guidance

Every pytest test must carry exactly one of `ci_cpu`, `ci_gpu`, or `manual`; `CONTRIBUTING.md` has the exact rules. Use module-level `pytestmark = pytest.mark.ci_cpu` for pure Python/metadata tests. Keep GPU, `libGL`/`cv2`, large-checkpoint, credential, and download-heavy checks out of `ci_cpu`.

## Boundaries

Keep dependency direction strict: `core` -> `infra` -> recipes/integrations. `core` and `infra` must not import from `integrations/`; expose a generic config slot or override hook instead of adding model-specific branches. Built-in reusable model pieces belong in `flashdreams/flashdreams/recipes/`; standalone plugin packages belong in `integrations/<name>/`.

## Known Pitfalls

- FlashDreams targets large NVIDIA GPUs; many real rollouts need 80 GB VRAM and CUDA-capable dependencies.
- Full `uv sync`/`uv run` can build CUDA packages such as Transformer Engine. For narrow work, use `uv sync --package <integration> --extra dev` when possible.
- `flashdreams-run --no-instantiate` resolves config only; it does not prove weights, CUDA execution, or parity.
- First GPU runs include compile/autotune warmup, so benchmark harnesses should discard warmup chunks.
- Docs CI mocks heavy GPU packages; local docs work should use the commands in `docs/README.md`.
- Plain `pytest` includes manual tests. Use `pytest -m ci_cpu` or `pytest -m "not manual"` for normal local validation.
- New or moved skills must pass the Agent Skills frontmatter checks in `tests/test_agent_skills.py`.

## Troubleshooting Links

- Setup and requirements: `README.md`, `docs/source/quickstart/installation.rst`
- CLI details: `docs/source/api/cli.rst`
- Integration/plugin layout: `docs/source/api/integrations.rst`
- New integrations: `docs/source/developer_guides/new_integration.rst`
- Docs and CPU autodoc: `docs/README.md`
- Tests and quality regressions: `tests/README.md`
- Security reports: `SECURITY.md`
