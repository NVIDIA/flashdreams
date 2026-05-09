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

# `template` recipe

Minimal end-to-end recipe exercising every contract in
`flashdreams.infra`. Use as a reference when scaffolding a new recipe.

## What's exercised

- **Offline rollout** — `TEMPLATE_OFFLINE`, one AR step over the full
  temporal window.
- **Autoregressive streaming** — `TEMPLATE_AUTOREGRESSIVE`, multiple
  AR steps over a sliding `BlockKVCache`.
- **Classifier-free guidance** — opt-in by patching
  `guidance_scale > 1.0` via `derive_config`. The uncond branch gets an
  independent `network_cache_uncond`; `negative_context` is encoded
  only when CFG is on.
- **Context parallelism** along `L` — `cp_size` auto-detected from
  `torch.distributed.get_world_size()`, so `torchrun --nproc_per_node=N`
  is the single source of truth. Attention uses
  `flashdreams.core.attention.RingAttention` (fuses the KV gather with
  SDPA via an LSE merge).
- **Per-AR-step control input** — `TemplateControlEncoder` (a
  `StreamingEncoder`). Setting `encoder=None` exercises the no-control
  branch (`test_template_no_control`).
- **One-shot context encoder** — `TemplateTransformerConfig.context_encoder`
  (an `Encoder`; defaults to `NullEncoderConfig`, identity). Plug in a
  text or CLIP image encoder here.
- **Output decoding** — `TemplateDecoder` (a `StreamingDecoder` that
  ignores its empty cache; swap for a `StreamingVideoDecoder` subclass
  when the recipe needs spatial / temporal compression contracts).
- **Config derivation** — `TEMPLATE_AUTOREGRESSIVE` is a
  `derive_config` patch on top of `TEMPLATE_OFFLINE`, and
  `TEMPLATE_AUTOREGRESSIVE_COMPILED` derives from that.
- **`torch.compile` + `CUDAGraphWrapper`** — off by default; flip via
  `derive_config` patching `compile_network` / `use_cuda_graph` (or use
  the prebuilt `TEMPLATE_AUTOREGRESSIVE_COMPILED`). Covered by
  `test_template_compile_and_cudagraph_equivalence`.
- **Checkpoint loading** — `TemplateTransformerConfig.checkpoint_path`.
  `None` keeps the random init.

## Files

| Path | Purpose |
|---|---|
| `transformer/network.py` | 1-block DiT + `BlockKVCache` plumbing. |
| `transformer/__init__.py` | `Transformer` subclass, AR cache, config. |
| `encoder.py` | Control encoder (control channels → latent channels). |
| `decoder.py` | Decoder (latent channels → output channels). |
| `config.py` | Module-level literal pipeline configs + `TEMPLATE_CONFIGS`. |

## Entry point

```bash
uv run --extra dev pytest flashdreams/tests/test_template.py -v
# or as a script
uv run --extra dev python flashdreams/tests/test_template.py
# multi-GPU CP equivalence (after the single-GPU run has written the reference)
uv run --extra dev torchrun --nproc_per_node=2 -m pytest \
    flashdreams/tests/test_template.py::test_template_cp_equivalence -v
```

## Shape conventions

- Pre-patchify: `[B, C, T, H, W]` for both noisy latent and control.
- Post-patchify: `[B, L=T*H*W, C]`, CP-split to `[B, L/cp_size, C]`.
- Decoded output: `[B, C_out, T, H, W]`.
