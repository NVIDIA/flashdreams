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

# `alpadreams`

Alpadreams integration package for `flashdreams`.

## Hugging Face org configuration

Alpadreams resolves public Omni Dreams assets from the `nvidia` Hugging Face
org by default:

- `nvidia/omni-dreams-models` for checkpoints.
- `nvidia/omni-dreams-samples` for bundled example data.

Set `HF_TOKEN` to a token with access to the selected org. To use the external
mirror instead, set `OMNI_DREAMS_HF_ORG` before running or importing
FlashDreams:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>
export OMNI_DREAMS_HF_ORG=nvidia-omni-dreams-lha
```

Internal S3-backed runs can still set `FLASHDREAMS_INTERNAL_STORAGE=1`, which
switches checkpoint and example-data URLs back to `s3://flashdreams`.
