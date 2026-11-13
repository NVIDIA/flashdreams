# Torchao output-projection investigation (#625)

**Current result:** local compiler options resolve the measured torchao
compiled/eager discrepancy; the follow-up has 54/54 passing benchmark cases.
Performance still does not justify adoption for this projection. See
[the compatibility follow-up](#compatibility-follow-up-local-compiler-options)
for the diagnosis, required options, new measurements, and remaining limits.

Public evidence: [charts, measurement exports, diagnosis and validation logs](https://github.com/fusheng-ji/flashdreams/tree/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625).
The linked snapshot preserves the original results; machine hostnames are redacted.

## Scope and provenance

This is a component evaluation, not a model-quality or end-to-end throughput
claim. The experimental opt-in interface is limited to the shared accelerated
attention **output projection**, plus a reusable derived Linear. Q/K/V,
activation sharing, attention kernels, streaming K/V state, context parallelism,
model defaults, and mandatory dependencies are unchanged.

Base: NVIDIA/flashdreams `fd52f1a1f4076f86f3fcb86cf5a4425497804b65`.
At investigation time (2026-10-08), issue [#625](https://github.com/NVIDIA/flashdreams/issues/625)
was open, Priority-P2, unassigned, with no comments or development links.
The open-PR API returned 77 entries. Searching their titles and bodies, and
GitHub's PR search, found no torchao integration. PR #541 concerns Wan/LingBot
adapters; #639 concerns SageAttention; #589 concerns Cam2V timing. A historical
Self-Forcing parity patch uses torchao PerTensor quantization but does not
integrate the shared accelerated layer. Raw issue/main/PR snapshots are retained
under the local `artifacts/torchao-625/` directory.

## Capability map

Status below describes upstream documentation/source, **not local validation**.
Only the dynamic rowwise FP8 row is benchmarked. API names and implementation
observations are pinned to torchao 0.18.0; prototype paths are not promises of
stable compatibility.

| Capability | FlashDreams / torchao relationship | Upstream status and requirements | First-stage decision |
| --- | --- | --- | --- |
| FP8 dynamic activations + FP8 weights | Existing E4M3/E5M2 activation paths use explicit scales; torchao provides `Float8DynamicActivationFloat8WeightConfig` | Stable rowwise workflow, recommended H100/B200; use CUDA/BF16 aligned Linear and PyTorch kernels | Measure E4M3/PerRow only |
| FP8 weight-only | `Float8WeightOnlyConfig`; no direct current weight-only equivalent | Public API; v0.18 PyTorch implementation dequantizes weights before a high-precision matmul | Inventory only; packed storage does not imply FP8 GEMM speed |
| INT8 weight-only | `Int8WeightOnlyConfig`; existing FlashDreams INT8 path also quantizes activations | Stable workflow, upstream recommends A100 for BF16 activations | Inventory only |
| INT4 weight-only | `Int4WeightOnlyConfig`; new precision option for this layer | Stable workflow, packing/group constraints and kernel-dependent dependencies | Inventory only |
| FP8 activation + INT4 weight | `Float8DynamicActivationInt4WeightConfig` | Stable workflow listed for H100; packing and optional kernels must be checked per target | Inventory only; no B200 claim |
| Structured / block sparsity | Alternative capability, not equivalent to dense quantization | Semi-structured and block paths are kernel/hardware specific; pruning changes weights and needs quality validation | Inventory only; dense checkpoint cannot be declared equivalent |
| MXFP8 / MXFP4 | Additional block-scaled formats | Prototype inference workflows listed for B200; optional kernel dependencies | Deferred |
| NVFP4 | Additional precision format | Prototype inference workflow listed for B200; separate training status is irrelevant here | Deferred |
| Attention, RoPE, RMSNorm, fusion, K/V cache and streaming | Existing accelerated layer owns execution and state contracts | No evidence that the selected torchao Linear replaces these contracts | Retain FlashDreams ownership |
| Compile and explicit CUDA graphs | Reuse FlashDreams wrappers and persistent compiler-cache policy | Must test tensor-subclass tracing, specialization, capture and replay on the pinned stack | Measured component matrix below |

Sources: [workflow status](https://docs.pytorch.org/ao/main/workflows/index.html),
[quantized inference](https://docs.pytorch.org/ao/main/workflows/inference.html),
[v0.18.0 quantization API](https://github.com/pytorch/ao/blob/v0.18.0/torchao/quantization/quant_api.py),
[v0.18.0 FP8 tensor implementation](https://github.com/pytorch/ao/blob/v0.18.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py),
[release notes](https://github.com/pytorch/ao/releases/tag/v0.18.0),
[serialization guide](https://docs.pytorch.org/ao/main/eager_tutorials/serialization.html).
The older upstream version-compatibility issue did not yet list the final 0.18
release when checked, so successful imports and runtime checks are reported
separately from that table.

Checkpoint consequences are validated only for the selected dynamic FP8 adapter:
native weights are saved and derived execution tensors are rebuilt. For every
deferred weight-only, mixed, MX, and NV format, packed-layout serialization and
loading across versions/devices remain untested; an upstream quantized-checkpoint
workflow must not be assumed compatible with FlashDreams' nonpersistent buffers.
Sparse execution also requires a pruning/mask/repacking lifecycle and cannot be
reconstructed from a dense checkpoint without defining that transformation.

## Adapter and checkpoint findings

The original `NonPersistentLinear` registers weight/bias as nonpersistent
buffers. Its quantized subclass adds FP32 scales. Optimized attention keeps
checkpoint-native parameters, rebuilding execution weights after loading and
in `_apply`. LingBot also has model-owned rebuild/replacement paths; they are
outside this output-only change.

Torchao's public `quantize_` assigns a **Parameter**, even if the input Linear
used buffers. Applying it directly would change the checkpoint contract. The
adapter instead transforms a temporary Linear, verifies an actual E4M3
`Float8Tensor`, and registers its detached value as a nonpersistent buffer.
Checkpoint-native BF16 weights stay available for rebuilding and explicit
fallback. The source buffer aliases those weights; it does not duplicate their
storage. State-dictionary key sets and strict save/load round trips are tested
through an actual optimized attention owner, rather than a synthetic checkpoint
schema. No tensor-subclass serialization becomes a new checkpoint requirement.

CPU/FP32 and meta are staging states. Source movement/casting finishes before
quantization, compilation, and capture. Reloading requires fresh compilation
and graph reset/recapture: the existing wrapper does not detect replacement of
weight storage automatically. No hot reload guarantee is added.

The backend imports torchao only when materializing CUDA/BF16 execution weights,
and explicitly requires version 0.18.0. It is manually installed in an isolated
environment, with no dependency metadata/lockfile change. Missing packages,
unsupported policies, shapes and source weights raise errors; there is no hidden
BF16 substitution. `output_projection=None` selects the native-precision path.

### Zero-row failure found in the first GPU run

With the unmodified torchao config, all-zero activations produced NaNs both with
and without bias on B200. The original failure is retained in
`gpu-smoke-1073574.log`. In v0.18 `_choose_scale_float8`, scale is `max_abs / 448`
without an unconditional lower bound, followed by division by scale.

The public `activation_value_lb=float32.tiny * 448` option prevents this failure
and is part of the **measured** configuration. No upstream/private kernel is
patched. The public config has no equivalent weight lower-bound argument:
zero/underflow weight rows producing nonfinite quantized data are rejected at
preparation, with an explicit native-backend fallback message. This remains a
limitation, not a successful zero-weight compatibility claim.

## Methodology and reproduction

- B200 allocated through Slurm `overflow`, job name `test`, one GPU, 8 CPUs,
  64 GiB RAM, four-hour job limit. Smoke, benchmark and regression allocations
  are serialized using job dependencies; no existing user process is stopped.
- Isolated Python 3.12 environment: PyTorch 2.12.1+cu130, torchao 0.18.0,
  Triton 3.7.1. Exact package versions, cuDNN, driver and device metadata are
  retained in artifacts and each benchmark JSON.
- Fixed `M=4800, K=N=2048`, synthetic Gaussian BF16 inputs/source weights,
  no bias, seed 42.
  All three candidates generate the same tensors. Both FP8 paths include
  activation quantization. Existing FP8 uses per-output-channel weight scales
  and `Granularity.SLICE` inputs; torchao uses E4M3/PerRow, `TORCH` kernels,
  `use_fast_accum=False`, and `set_inductor_config=False`.
- Four modes: eager, `compile_module(dynamic=False)`, `CUDAGraphWrapper`, and
  compiled callable inside that wrapper. Compile mode is
  `max-autotune-no-cudagraphs`; `.drain` precedes capture.
- Three independent-process repetitions, five warmup calls and 50 measured
  calls per case. Compile modes use a fresh cache for each cold repetition,
  then a new process reusing that repetition's Inductor/Triton/CUDA cache.
- Synchronized wall time is the primary callable latency and includes graph
  input copies/output clones. CUDA event timing is a separate 50-call pass,
  avoiding event-instrumentation overhead in the headline wall-time samples.
- Preparation includes module construction, weight quantization, optional import,
  and first-use kernel initialization. Compile-wrapper creation and the first
  execution (including compilation/autotune) are separate fields. Capture-call
  time includes initial replay and output clone, not just CUDA capture.
- Weight storage is counted by unique underlying storage, including retained
  BF16 source weights, FP8 data, and scales. Allocation peaks are process-local
  PyTorch allocator measurements, not total board memory or a model estimate.
- Numerical metrics compare to the same BF16 `F.linear`: relative L2, MAE,
  maximum absolute error and cosine similarity. The existing E4M3 epsilon
  tolerance is reused as a component smoke check, not a model-quality gate.
  A separate eager/compiled comparison uses `rtol=atol=0.02`, and preserves
  relative L2 and mismatch fraction even when that elementwise check fails.

From the repository root, with an allocated CUDA device and a matching environment:

```bash
PYTHONPATH=flashdreams .venv-torchao/bin/python \
  flashdreams/benchmarks/accelerated/quantization/run_torchao_comparison.py \
  --output artifacts/torchao-625/run-001 --repeats 3
```

The runner refuses an existing manifest, saves failures as well as successes,
and preserves raw timing samples, environment metadata and source snapshots.
The first installation attempt using only the CUDA wheel index failed because
it lacked the required setuptools version. Installing the same PyTorch pin
through the repository's default PyPI index resolved this; no stack downgrade
was used.

## Original measurements: default compiler policy

**Original recommendation: do not adopt the default-compiler configuration for this projection.**
Retain the interface as a local experimental PoC, not a production-validated
backend. BF16 eager is fastest here, both FP8 implementations retain additional
weight storage, and torchao's default compiled path fails the chosen
cross-mode numerical check. This is a conclusion about this B200 software stack
and this shape, not about every torchao workflow or model.

The formal sweep was Slurm job **1073641** on `<B200-node>`:
PyTorch **2.12.1+cu130**, CUDA **13.0**, torchao **0.18.0**, Triton **3.7.1**,
cuDNN **92000**, driver **580.126.20**, B200 **SM100**. Its 54 processes produced
41 passes, 12 torchao compiled parity failures, and one preparation failure in
the existing FP8 baseline. That last failure was a Triton `ptxas` failure whose
error handler could no longer find its `/tmp/*.log`; the underlying assembler
error is therefore unknown. Recovery job **1073708** reran only that case on
the same node with fresh caches and a private `TMPDIR`, and passed. Original
failure evidence was retained. The tables use **54 valid measurements**, three
independent processes per mode/cache/backend cell, including this disclosed
replacement. Timing a failed parity case does not qualify that case for use.

### Steady latency and throughput

Values are medians across the three process-level statistics; ranges show the
three wall medians. The p90 columns are medians of process p90 values (nearest
rank in each set of 50 samples), not percentiles pooled across processes.
Compile rows below use **warm caches in fresh processes**; eager/graph rows use
fresh caches. Cold-cache steady samples also remain in the raw JSON. Runs were
serial in fixed order without explicitly locked GPU clocks. These short
component timings are not confidence intervals or sustained model measurements.

| Backend | Mode | Wall median [range], us | Wall p90, us | CUDA median / p90, us | M rows/s | Parity |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bf16 | eager | 42.54 [42.48, 42.59] | 44.26 | 38.78 / 39.94 | 112.8 | PASS |
| bf16 | compile | 72.62 [72.20, 73.14] | 77.33 | 70.30 / 74.72 | 66.1 | PASS |
| bf16 | graph | 52.81 [52.78, 52.91] | 55.69 | 48.75 / 49.57 | 90.9 | PASS |
| bf16 | compile_graph | 53.73 [53.47, 53.77] | 56.97 | 49.34 / 51.26 | 89.3 | PASS |
| flashdreams | eager | 64.66 [64.33, 65.01] | 69.96 | 62.75 / 66.66 | 74.2 | PASS |
| flashdreams | compile | 79.53 [79.32, 79.89] | 85.93 | 76.67 / 82.05 | 60.4 | PASS |
| flashdreams | graph | 55.86 [55.79, 56.09] | 59.17 | 51.33 / 52.38 | 85.9 | PASS |
| flashdreams | compile_graph | 56.69 [56.56, 58.75] | 59.56 | 51.20 / 52.19 | 84.7 | PASS |
| torchao | eager | 157.37 [156.99, 158.69] | 177.88 | 153.42 / 161.79 | 30.5 | PASS |
| torchao | compile | 80.28 [79.75, 94.75] | 88.57 | 77.79 / 82.11 | 59.8 | FAIL |
| torchao | graph | 127.30 [127.25, 127.75] | 129.32 | 122.72 / 124.29 | 37.7 | PASS |
| torchao | compile_graph | 57.31 [57.05, 59.94] | 59.29 | 52.62 / 53.60 | 83.8 | FAIL |

Torchao eager is about 3.70 times the BF16 eager latency. Compiled graph mode
reduces its overhead, but 57.31 us is still above BF16's 53.73 us and comparable
to existing FP8's 56.69 us, while failing parity. The existing FP8 implementation
also loses to BF16 for this component. Graph copies/clones and Python launch
costs are material at this scale; CUDA event spans include device idle gaps
between host submissions and must not be read as GEMM-only kernel times.

### Startup, separated from steady state

All values below are milliseconds, again medians across three processes.
Preparation includes weight conversion and first-use imports/kernel setup.
The first execution includes compilation/autotune for compile modes. Warmup is
five direct calls plus two wrapper warmup calls for graph modes. The final
column includes capture, the wrapper's first replay, and an output clone.

| Backend | Mode/cache | Prepare, ms | Compile wrapper, ms | First execution, ms | Warmup, ms | Capture+replay+clone, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | eager/fresh | 0.46 | 0.00 | 113.82 | 0.57 | 0.00 |
| bf16 | compile/cold | 0.41 | 1392.73 | 8334.72 | 0.40 | 0.00 |
| bf16 | compile/warm | 0.44 | 1361.47 | 5020.99 | 1.02 | 0.00 |
| bf16 | graph/fresh | 0.46 | 0.00 | 112.39 | 0.67 | 4.56 |
| bf16 | compile_graph/cold | 0.44 | 1435.53 | 8294.32 | 0.63 | 4.60 |
| bf16 | compile_graph/warm | 0.40 | 1401.90 | 4987.36 | 1.24 | 5.32 |
| flashdreams | eager/fresh | 1376.71 | 0.00 | 13.90 | 0.69 | 0.00 |
| flashdreams | compile/cold | 1365.92 | 1401.43 | 18554.99 | 0.56 | 0.00 |
| flashdreams | compile/warm | 623.81 | 1403.50 | 4924.56 | 0.91 | 0.00 |
| flashdreams | graph/fresh | 1358.16 | 0.00 | 14.35 | 0.95 | 4.79 |
| flashdreams | compile_graph/cold | 1426.26 | 1428.58 | 18576.79 | 0.85 | 4.84 |
| flashdreams | compile_graph/warm | 604.86 | 1435.16 | 4873.87 | 1.22 | 4.73 |
| torchao | eager/fresh | 709.30 | 0.00 | 14.57 | 1.00 | 0.00 |
| torchao | compile/cold | 718.78 | 1288.71 | 22347.41 | 0.74 | 0.00 |
| torchao | compile/warm | 723.83 | 1290.57 | 4933.17 | 0.92 | 0.00 |
| torchao | graph/fresh | 698.24 | 0.00 | 14.53 | 1.38 | 5.80 |
| torchao | compile_graph/cold | 701.16 | 1256.88 | 22044.35 | 1.09 | 5.77 |
| torchao | compile_graph/warm | 699.35 | 1275.95 | 4853.27 | 1.18 | 5.28 |

Warm-cache torchao processes recorded AOT and FX graph cache hits, despite an
upstream warning that `Float8Tensor` lacks `_stable_hash_for_caching`. First
execution still took about 4.9 seconds and retained autotuning work; cache reuse
does not make startup free. Every compiled fixed-shape case recorded one unique
Dynamo graph, with no observed graph breaks or recompilation. All captures and
replays in the valid measurement set executed successfully. The twelve failures
are numerical assertions after measurement, not compile or capture exceptions.

### Memory and retained weights

The same warm-cache/fresh grouping is used below. Peaks are absolute PyTorch
allocator readings, including input/source storage, workspaces, graph pools,
and returned tensors; they are not process RSS or total device usage. Metrics
are taken before the additional numerical-comparison buffers are created.
Reserved graph pools can remain large even after their temporary tensors die.

| Backend | Mode | Startup peak allocated, MiB | Steady peak allocated / reserved, MiB |
| --- | --- | ---: | ---: |
| bf16 | eager | 96.25 | 96.25 / 112.00 |
| bf16 | compile | 96.25 | 96.25 / 112.00 |
| bf16 | graph | 147.00 | 184.50 / 206.00 |
| bf16 | compile_graph | 147.00 | 184.50 / 206.00 |
| flashdreams | eager | 77.65 | 77.65 / 102.00 |
| flashdreams | compile | 77.65 | 77.65 / 102.00 |
| flashdreams | graph | 96.40 | 124.51 / 164.00 |
| flashdreams | compile_graph | 96.40 | 124.51 / 164.00 |
| torchao | eager | 172.90 | 173.40 / 262.00 |
| torchao | compile | 78.76 | 77.65 / 130.00 |
| torchao | graph | 191.65 | 124.51 / 258.00 |
| torchao | compile_graph | 96.40 | 124.51 / 164.00 |

| Backend | Retained BF16 source, MiB | FP8 data, MiB | FP32 scales, MiB | Deduplicated resident weights, MiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 8 | 0 | 0 | 8 |
| Existing FP8 | 8 | 4 | 0.0078125 | 12.0078125 |
| torchao FP8 | 8 | 4 | 0.0078125 | 12.0078125 |

Neither FP8 candidate reduces resident weight memory under the required
checkpoint/fallback contract. Allocator peaks can still differ because GEMM
workspaces and activation intermediates differ. Torchao eager's high temporary
allocation cost largely disappears with compilation, but that observation does
not resolve its numerical compatibility failure.

### Numerical compatibility

The synthetic input and source weight distribution is identical across all
cases. The error scale below is specific to those unnormalized Gaussian tensors.
Graph results match their corresponding eager/compiled mode. All valid FP8
measurements pass the existing relative-L2 smoke tolerance of E4M3 epsilon
(0.125); this does not establish task quality.

| Backend | Mode | Relative L2 vs BF16 | MAE | Max abs error | Cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| bf16 | eager | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| bf16 | compile | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| flashdreams | eager | 0.037498 | 1.349339 | 9.666016 | 0.999297 |
| flashdreams | compile | 0.037498 | 1.349339 | 9.666016 | 0.999297 |
| torchao | eager | 0.037490 | 1.348924 | 9.125000 | 0.999297 |
| torchao | compile | 0.037494 | 1.349111 | 9.416016 | 0.999297 |

Default compiled torchao differs from its own eager result by relative L2
**0.01152248**, maximum absolute difference **5.0**, and a mismatch fraction
**0.25651377** at `rtol=atol=0.02`. This repeats in both compile modes and cache
states across all three rounds. Its error against BF16 is almost unchanged;
the evidence does **not** show worse aggregate BF16 accuracy after compilation.
The unresolved issue is cross-mode consistency under the stated acceptance check.

A separate diagnostic, job **1073642** on B200 node `<B200-node>`, changed only
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS` in two fresh processes. The installed
PyTorch `_inductor/config.py` explains that Inductor ordinarily removes
intermediate BF16/FP16 downcast/upcast pairs during fusion. With this option set
to `1`, eager-relative L2 decreased to **0.00051730**, mismatch fraction to
**0.00076223**, and maximum absolute difference to **2.0**. The original
criterion **still fails**. This supports intermediate precision elision as a
major contributor, but does not identify the remaining differences. The
adapter does not change this global compiler policy, and this one-off diagnostic
is not included in the three-round timing tables.

Zero activations are separately tested, with exact finite zero/bias outputs
in eager and a compiled zero-input check after applying the public lower bound.
The benchmark handles a zero reference norm separately when computing relative
L2. Zero/underflow weight rows remain unsupported and explicitly rejected.

### Validation and artifact inventory

- CPU quantization/common regressions: **37 passed**, 44 deselected. Includes
  configuration validation, CPU/meta staging, no-parameter/empty-state contract,
  and blocking torchao imports to exercise the optional-dependency boundary.
- GPU attention-projection regressions, job **1073605**: **15 passed**, 428
  deselected. Includes the existing E4M3 projection matrix, real tensor subclass
  and payload checks, bias/no bias, 3D inputs, zero inputs, invalid width/dtype,
  strict checkpoint round-trip with `assign=True`, source reload/migration,
  explicit fallback, shape changes, graph reset/recapture, and retained outputs.
  The missing-package diagnostic is checked at CUDA materialization; CPU/meta
  staging deliberately does not resolve the optional package.
- Manual compiled lifecycle check, job **1073624**: **1 passed**. Zero input,
  shape specialization, fresh compilation after source replacement, graph
  recapture, and old output ownership pass for its small identity projection.
  This does not override the representative random-projection parity failures.
- Formal performance sweep plus the disclosed recovery: **42 passing and 12
  parity-failing measurements**. The initial zero-activation failure and an
  interrupted pilot sweep are retained, but excluded from final aggregates.
- Ruff lint/format, targeted `ty` checks, `git diff --check`, `uv lock --check`,
  and Sphinx HTML build with `-W --keep-going` passed. No dependency or lockfile
  update was needed. The only `pyproject.toml` change tells the type checker that
  the optional torchao import may be absent, following existing repository policy.

Artifacts are local and ignored under `artifacts/torchao-625/`:

- `sweep-1073641/`: original manifest, per-process pytest-benchmark JSON with all
  samples, full logs, environment freeze, GPU metadata, source patch and runner
  snapshot, cache directories, and original summaries.
- `recovery-1073708/result.json`, `gpu-recovery-1073708.log`: replacement baseline
  measurement; `gpu-recovery.sh` records its private temporary/cache setup.
- `combined-measurements.json`, `report-tables.txt`, `report_tables.py`: disclosed
  merged view and executable aggregation used for this report.
- `precision-1073642-{0,1}.json`, `precision_diagnostic.py`,
  `gpu-precision-check-1073642.log`, `inductor-precision-config.txt`: diagnostic
  outputs, executable reproducer, and installed-source explanation.
- `gpu-regression-1073605.log`, `gpu-compile-check-1073624.log`, `cpu.log`,
  `types.log`, `docs-final.log`, `slurm-accounting.txt`: validation evidence.
- `gpu-smoke-1073574.log`, `gpu-smoke-1073582.log`, `sweep-1073590/`,
  `initial-install-failure.txt`, `install.log`: original failures and recovery
  history. `issue-625.json`, `open-prs.json`, `upstream-main.json` retain triage.

The Slurm submission used the requested allocation; a minimal reproduction is:

```bash
ssh cluster 'sbatch --partition=overflow --job-name=test --gres=gpu:b200:1 \
  --cpus-per-task=8 --mem=64G --time=04:00:00 \
  --output=/absolute/repo/artifacts/torchao-625/benchmark-%j.log \
  /absolute/repo/artifacts/torchao-625/gpu-benchmark.sh'
```

That script sets `PYTHONPATH=flashdreams`, `OMP_NUM_THREADS=8`, and
`TORCHINDUCTOR_COMPILE_THREADS=8`, then invokes the runner shown above with a
fresh output directory. All GPU jobs used only their Slurm allocation and were
serialized; no model generation was run. No GitHub comment, push, or PR was made.

### Original adoption boundary and follow-up

Keep BF16 and the existing backend as the supported choices for this workload.
The local torchao interface remains useful for reproducing failures and testing
future upstream versions, but should not be advertised as ready for production.
The required activation-floor workaround, unsupported zero-weight rows,
version pin, compiler parity investigation and cache/deprecation warnings add
maintenance cost without a measured steady-state advantage here. No arbitrary
speedup cutoff is needed to reach this conclusion.

Before promoting or widening the backend, resolve the remaining compiled/eager
numerical differences with an explicit acceptance decision, establish the
weight-row policy through public APIs, and repeat the component matrix on the
intended production stack. Model-level quality and end-to-end benefit then
require a separate selected-model experiment using real activations/checkpoints,
explicit module inclusion/exclusion, and the project's quality protocol.
Q/K/V integration, other formats, and model generation remain outside this PoC.


## Compatibility follow-up: local compiler options

The follow-up fixes the measured compiled/eager discrepancy without changing
weights, the quantization config, tolerances, dependencies, or global compiler
numerics. The preceding measurements describe the **original default compiler
policy**; they remain valid evidence of that policy's failures.

### Controlled diagnosis

Job **1073871** ran four fresh processes on B200 node `<B200-node>`, using the
same pinned stack, `M=4800, K=N=2048`, and seed 42. Each process compared the
activation quantizer's FP8 data and scales separately, then ran eager and
compiled GEMM with identical quantized operands. It also substituted compiled
quantizer outputs into eager GEMM to isolate the source of final differences.

| Local compile options | Differing scales / 4800 | Differing FP8 values / 9830400 | Identical-operand GEMM | Full projection max error vs eager |
| --- | ---: | ---: | --- | ---: |
| Default | 4384 | 168333 | Exact | 5.0 |
| `emulate_precision_casts=True` only | 0 | 451 | Exact | 2.0 |
| `eager_numerics.division_rounding=True` only | 4384 | 168333 | Exact | 5.0 |
| Both | 0 | 0 | Exact | 0.0 |

The experiment identifies **two activation-quantization effects**. Inductor
first removes intermediate BF16 rounding, changing scales. Preserving those
casts restores scales but still leaves approximate FP32 division before the
FP8 cast. The generated kernel with both options uses `triton.language.div_rn`,
and FP8 data plus final outputs become exactly equal to eager. No GEMM
accumulation change is needed: identical quantized inputs already produce
identical outputs under every tested policy.

A preserved example is input `-1.5859375` with scale `0.0068359375` at row 526,
column 47. Their exact quotient is `-232`, halfway between FP8 values `-224`
and `-240`. Eager produces `-224`; the casts-only compiled quantizer produces
`-240`. The combined policy restores `-224`. The diagnostic JSON preserves
16 observed differences, not just a synthetic approximation of this boundary.
Diagnostic tolerance fractions are calculated after conversion to FP32; the
benchmark retains its original BF16 comparison and `rtol=atol=0.02` assertion.
Exact-equality results above are independent of that metric distinction.

### Explicit compiler interface

`compile_module` now accepts keyword-only `options: dict[str, Any] | None = None`.
Its default path is unchanged. For explicit options it copies the selected
mode's configuration, overlays caller options, and passes the merged dictionary
to `torch.compile`; PyTorch does not permit passing `mode` and `options` together.
No torchao-specific behavior is added to the shared helper.

```python
compiled = compile_module(
    prepared_projection,
    dynamic=False,
    options={
        "emulate_precision_casts": True,
        "eager_numerics.division_rounding": True,
    },
)
```

The options belong to the compiled callable, including its later specialization
and cache lookup. They do not modify the caller's dictionary or global numerical
settings. In the combined-policy diagnostic, a subsequent **default** compile
in the same process reproduced the original discrepancy (max error 5.0), while
both global settings remained false. This is deliberate isolation, not a claim
that the default compiler now matches eager. The benchmark exposes the opt-in
as `--torchao-eager-numerics`, records the applied dictionary in each JSON, and
leaves BF16/native FP8 compilation unchanged. All processes get private temporary
directories to avoid the previous shared-temporary-file failure.

### Follow-up regression checks

- **44 CPU tests passed**, including default compiler behavior, mode/option
  precedence, dynamic argument forwarding, dictionary preservation, and the
  existing quantization/common regressions.
- **6 manual GPU cases passed**: seeds 0, 1, 42 crossed with bias/no bias,
  random 3D inputs, exact zero/bias output, shape-triggered recompilation,
  graph drain/capture/replay, new source weights with new compiled instances,
  and retained output ownership. Eager/compiled checks still use 0.02/0.02;
  BF16-reference checks still use the existing 0.125 relative-L2 limit.
- The real optimized-attention output-projection owner passed a separate manual
  compiled lifecycle check: two source states, strict checkpoint save/load with
  `assign=True`, CPU/CUDA migration, fresh compilation and graph recapture,
  unchanged checkpoint key set, and old outputs retained across replacement.
- **15 GPU regression tests passed**, including existing native E4M3 projections
  and torchao bias, staging, missing-package, fallback, and checkpoint coverage.
- Ruff, targeted type checks, and the Sphinx build passed. No model generation
  or whole-model compile/quality test was run.

The GPU regression checks ran in job **1073874**, after the controlled diagnosis.
The baseline zero/underflow-weight rejection and explicit BF16 fallback are
unchanged. Compatibility on supported inputs does not establish performance
benefit or remove those input restrictions.
### Follow-up performance and final decision

Job **1073875** completed **54/54 passing processes**, three independent
repetitions per backend/mode/cache cell, on B200 node `<B200-node>`.
The runtime stack remains PyTorch 2.12.1+cu130, torchao 0.18.0, Triton 3.7.1,
CUDA 13.0, cuDNN 92000, and driver 580.126.20. There were no replacements or
failed cases in this follow-up sweep. Every backend/mode had zero measured
maximum difference against its own eager result. All fixed-shape compilations
recorded one Dynamo graph with no graph breaks, and all six warm-cache torchao
processes recorded AOT and FX cache hits while preserving the local policy.

The aggregation method is unchanged: median across three process medians/p90s,
with ranges of the process wall medians. Compile rows use new processes with
reused caches; graph/eager rows use fresh caches. Each process has 5 warmup and
50 measured calls, plus a separate 50-sample CUDA-event pass. The baseline and
native FP8 rows do not enable the torchao numerical options.

| Backend | Mode | Wall median [range], us | Wall p90, us | CUDA median / p90, us | M rows/s | Parity |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bf16 | eager | 42.32 [42.12, 42.43] | 43.69 | 38.88 / 39.55 | 113.4 | PASS |
| bf16 | compile | 65.76 [65.73, 86.63] | 70.75 | 63.38 / 66.66 | 73.0 | PASS |
| bf16 | graph | 53.03 [52.97, 53.42] | 55.37 | 48.50 / 49.38 | 90.5 | PASS |
| bf16 | compile_graph | 53.66 [53.66, 54.21] | 57.09 | 49.15 / 49.73 | 89.4 | PASS |
| flashdreams | eager | 64.36 [64.21, 64.77] | 70.08 | 61.87 / 65.38 | 74.6 | PASS |
| flashdreams | compile | 79.28 [78.75, 97.51] | 86.29 | 76.96 / 81.25 | 60.5 | PASS |
| flashdreams | graph | 55.81 [55.77, 55.92] | 58.67 | 51.09 / 51.74 | 86.0 | PASS |
| flashdreams | compile_graph | 56.43 [56.36, 58.45] | 59.43 | 51.15 / 52.42 | 85.1 | PASS |
| torchao | eager | 158.23 [157.50, 158.35] | 173.36 | 152.42 / 159.14 | 30.3 | PASS |
| torchao | compile | 84.77 [79.40, 99.10] | 107.12 | 82.45 / 103.55 | 56.6 | PASS |
| torchao | graph | 127.41 [127.18, 127.61] | 128.70 | 122.83 / 124.42 | 37.7 | PASS |
| torchao | compile_graph | 59.59 [59.11, 59.95] | 60.73 | 54.58 / 55.46 | 80.6 | PASS |

The new data separates compatibility from adoption: the local numerical policy
passes the component checks, but does not establish an advantage over the BF16
baseline for this shape. Comparing policies across the original and follow-up
runs is descriptive, not a controlled estimate of the options' performance
cost: the allocation/node and private temporary-directory policy changed.
The follow-up's baseline and candidate measurements share the same allocation.

Startup values below are milliseconds and include the same separated stages as
the original experiment. The first BF16 cold process had a 14.16-second compile
wrapper and a 42.14-second first execution; these are preserved, not dropped.
Three-process medians limit the influence of this initial startup outlier, but
neither compiler cache state nor these samples control OS filesystem caches.
Capture-call time still includes capture, first replay, and output clone.

| Backend | Mode/cache | Prepare, ms | Compile wrapper, ms | First execution, ms | Warmup, ms | Capture+replay+clone, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | eager/fresh | 0.40 | 0.00 | 110.93 | 0.53 | 0.00 |
| bf16 | compile/cold | 0.43 | 1298.25 | 12278.53 | 0.43 | 0.00 |
| bf16 | compile/warm | 0.35 | 1282.26 | 5025.88 | 0.84 | 0.00 |
| bf16 | graph/fresh | 0.41 | 0.00 | 111.28 | 0.70 | 13.90 |
| bf16 | compile_graph/cold | 0.41 | 1292.08 | 11353.99 | 0.64 | 5.10 |
| bf16 | compile_graph/warm | 0.40 | 1285.20 | 4901.61 | 1.21 | 5.14 |
| flashdreams | eager/fresh | 1732.14 | 0.00 | 13.36 | 0.70 | 0.00 |
| flashdreams | compile/cold | 1889.52 | 1317.50 | 33859.48 | 0.55 | 0.00 |
| flashdreams | compile/warm | 572.33 | 1307.93 | 4803.96 | 1.01 | 0.00 |
| flashdreams | graph/fresh | 1706.24 | 0.00 | 13.71 | 0.91 | 4.75 |
| flashdreams | compile_graph/cold | 1897.22 | 1259.79 | 33913.45 | 0.85 | 4.88 |
| flashdreams | compile_graph/warm | 568.86 | 1299.23 | 5204.86 | 1.27 | 6.73 |
| torchao | eager/fresh | 657.52 | 0.00 | 13.58 | 0.98 | 0.00 |
| torchao | compile/cold | 652.14 | 1154.41 | 50925.91 | 0.67 | 0.00 |
| torchao | compile/warm | 678.90 | 1181.58 | 4936.54 | 0.87 | 0.00 |
| torchao | graph/fresh | 656.12 | 0.00 | 13.65 | 1.40 | 6.04 |
| torchao | compile_graph/cold | 658.28 | 1171.18 | 50323.79 | 0.97 | 6.15 |
| torchao | compile_graph/warm | 659.64 | 1160.45 | 5016.93 | 0.88 | 22.20 |

Allocator memory remains distinct from retained weight payloads. Both FP8
implementations still retain 8 MiB of native source weights plus 4 MiB of FP8
data and 0.0078125 MiB of scales. The local compiler options do not remove that
12.0078125 MiB residency cost. The following peaks include allocator workspaces
and graph/output buffers; they are not model-memory estimates.

| Backend | Mode | Startup peak allocated, MiB | Steady peak allocated / reserved, MiB |
| --- | --- | ---: | ---: |
| bf16 | eager | 96.25 | 96.25 / 112.00 |
| bf16 | compile | 96.25 | 96.25 / 112.00 |
| bf16 | graph | 147.00 | 184.50 / 206.00 |
| bf16 | compile_graph | 147.00 | 184.50 / 206.00 |
| flashdreams | eager | 77.65 | 77.65 / 102.00 |
| flashdreams | compile | 77.65 | 77.65 / 102.00 |
| flashdreams | graph | 96.40 | 124.51 / 164.00 |
| flashdreams | compile_graph | 96.40 | 124.51 / 164.00 |
| torchao | eager | 172.90 | 173.40 / 262.00 |
| torchao | compile | 78.76 | 77.65 / 130.00 |
| torchao | graph | 191.65 | 124.51 / 258.00 |
| torchao | compile_graph | 185.40 | 124.51 / 164.00 |

Numerical errors against the original BF16 projection are below. The repaired
compiled torchao result now matches its eager error metrics, rather than merely
passing a relaxed tolerance. The FP8 component tolerance remains 0.125, and
all eager/compiled assertions remain `rtol=atol=0.02`.

| Backend | Mode | Relative L2 vs BF16 | MAE | Max abs error | Cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| bf16 | eager | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| bf16 | compile | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| flashdreams | eager | 0.037498 | 1.349339 | 9.666016 | 0.999297 |
| flashdreams | compile | 0.037498 | 1.349339 | 9.666016 | 0.999297 |
| torchao | eager | 0.037490 | 1.348924 | 9.125000 | 0.999297 |
| torchao | compile | 0.037490 | 1.348924 | 9.125000 | 0.999297 |

**Final decision:** the explicit local configuration is compatible with the
validated component cases, but adoption for this measured workload remains
**not recommended**. Keep it as an experimental opt-in. It retains extra source
weight storage and configuration/version maintenance, provides no demonstrated
steady-state benefit over BF16 here, and still rejects zero/underflow weight
rows. Default compilation without the options remains numerically divergent.
No claim is made for model-level quality, full attention compilation, other
shapes, or other devices; those require a separate selected-model evaluation.

### Follow-up reproduction and artifacts

```bash
PYTHONPATH=flashdreams .venv-torchao/bin/python \
  flashdreams/benchmarks/accelerated/quantization/run_torchao_comparison.py \
  --output artifacts/torchao-625/compatibility-v2/new-run \
  --repeats 3 --torchao-eager-numerics
```

Run this inside the requested Slurm allocation (`overflow`, job name `test`,
`--gres=gpu:b200:1 --cpus-per-task=8 --mem=64G --time=04:00:00`), with
`OMP_NUM_THREADS=8` and `TORCHINDUCTOR_COMPILE_THREADS=8`. GPU allocations were
serialized: diagnosis, validation, then a sweep dependent on successful
validation. No shared environment was changed or model generation performed.

All new evidence is under `artifacts/torchao-625/compatibility-v2/`:

- `diagnose.py`, `diagnose.sh`, `diagnostic-1073871/`: four-policy stage comparison,
  mismatch positions/values, full logs, and generated kernels/cache files.
- `attention_lifecycle.py`, `validate.sh`, `validation-1073874/`: executable real
  attention/checkpoint reproducer, six compiled regression cases, and GPU tests.
- `benchmark.sh`, `sweep-1073875/`: exact submission workload, per-process options,
  raw benchmark samples, metrics, source patch/runner snapshot, environment and
  GPU metadata, cache directories, and success/failure manifest.
- `report_tables.py`, `report-tables.txt`, `finalize_results.py`: reproducible
  aggregation and completeness/cache/numerical consistency checks.
- `cpu.log`, `types.log`, `docs.log`, `slurm-accounting.txt`, `final-review.patch`:
  validation, scheduler provenance, and final local review snapshot.

Original artifacts remain intact. No comment, push, or PR was published.
