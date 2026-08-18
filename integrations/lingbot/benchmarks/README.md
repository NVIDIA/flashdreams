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

# LingBot benchmarks

The LingBot benchmarks are manual, GPU-only pytest tests. The complete suite can
run on one sufficiently large GPU or with four-way context parallelism. The
single-GPU path is validated on a 256 GB NVIDIA GB300; smaller devices may run
out of memory during compile/autotune or cache setup. In the distributed path,
the 14B network has about 34.5 GiB of BF16 weights per rank before its KV cache,
activations, and compiler workspaces. The full-pipeline cases also require the
Hugging Face access and cache space documented in
[`integrations/lingbot/README.md`](../README.md).

Between benchmark cases, pytest synchronizes CUDA, resets compiler and CUDA-graph
state, collects Python objects, and empties the CUDA allocator cache. This prevents
one case from retaining memory needed by the next, but does not reduce a case's own
peak memory requirement.

Each benchmark layer runs three cases: the WAN/cuDNN reference, Triton FP8
projections with PyTorch cuDNN SDPA, and Triton FP8 projections with Triton
FlashAttention2 (FA2). Their stable labels are `wan_torch`, `triton_cudnn`, and
`triton_fa2`. Both Triton cases require compute capability 9.0 or newer and run
only on a single GPU because Triton attention does not support context
parallelism; cross-attention remains cuDNN in all three cases.

First sync the LingBot package and the workspace `test` dependency group,
which provides both `pytest` and `pytest-benchmark`:

```bash
uv sync --package flashdreams-lingbot --group test
```

Run the complete suite on one high-memory GPU:

```bash
uv run --package flashdreams-lingbot --group test pytest \
  integrations/lingbot/benchmarks \
  -p no:manual_marker -m manual --benchmark-only -v
```

Alternatively, run with four-way context parallelism when four usable CUDA
ordinals are visible in the same process namespace:

```bash
uv run --package flashdreams-lingbot --group test \
  torchrun --standalone --nproc_per_node=4 --no-python pytest \
  integrations/lingbot/benchmarks \
  -p no:manual_marker -m manual --benchmark-only -v
```

All four workers execute every WAN test because the DiT uses context-parallel
collectives. The tests align workers before every sample. Triton cases skip in
this distributed mode because that backend is single-GPU only. Once the command
is known to work, add `--local-ranks-filter=0` before `--no-python` to show
only rank 0's report; omit it while debugging because it also hides tracebacks
from failing nonzero ranks. If `torchrun` reports `invalid device ordinal`,
reduce `--nproc_per_node` or scope `CUDA_VISIBLE_DEVICES` to GPUs that are
usable together. Do not point multiple workers at one shared
`--benchmark-json` path; use a rank-specific path when retaining every rank's
raw report.

The block microbenchmark also fits on one large GPU:

```bash
uv run --package flashdreams-lingbot --group test pytest \
  integrations/lingbot/benchmarks/test_modules.py \
  -p no:manual_marker -m manual --benchmark-only -v
```

To select another benchmark layer, use one of these files:

- `test_modules.py` benchmarks only LingBot's integration-owned
  `CamCtrlBlock`. It deliberately does not add standalone benchmarks for the
  inherited Wan attention, MLP, normalization, encoder, or decoder modules.
  The whole-block timing necessarily includes the inherited Wan branches that
  the LingBot subclass executes.
- `test_network.py` benchmarks one steady-state evaluation of the complete
  random-initialized LingBot 14B camera-control DiT. It uses the CLI replay's
  352x640 geometry, compiled WAN or Triton self-attention, CUDA graph replay,
  and the shipped window15/sink3 cache layout; startup is excluded.
- `test_pipeline.py` separately benchmarks steady-state `generate` and
  `finalize` for
  `lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` at the CLI replay's
  352x640 geometry. This is an end-to-end measurement, so the recurring path
  includes its configured recipe components, but metadata identifies those
  stages and no reused component is presented as a LingBot module
  microbenchmark. At the measured AR indices, the Wan I2V VAE branch reuses
  its cached latent. Reported output FPS is generate-only; use the separate
  finalize result when evaluating the complete per-chunk lifecycle.

Keep `--group test` on both setup and run commands. A package-only environment
does not include the benchmark plugin. The benchmarks exclude setup, cache
fill, compile/autotune, and CUDA graph capture from measured rounds. When
publishing results, also record the exact command and commit, GPU/driver and
software stack, checkpoint identifiers, compiler-cache state, and fallback
warnings.
