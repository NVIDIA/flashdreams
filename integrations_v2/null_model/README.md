<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Null model

A deterministic CPU-safe pipeline used by v2 application tests. It emits `input + autoregressive_index` in `bcthw` layout.

This package supplies model components and config rather than an application slug.

## Tests

```bash
uv sync --package flashdreams-null-model
uv run --no-sync pytest integrations_v2/null_model/tests -m ci_cpu -v
```
