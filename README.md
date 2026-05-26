<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/flashdreams-logo-horizontal-light.png">
    <img alt="FlashDreams" src="assets/logo/flashdreams-logo-horizontal.png" width="600">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://verbose-adventure-7plnn5m.pages.github.io/main/index.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
</p>

**FlashDreams** is a high-performance inference and serving library for
interactive autoregressive video and world models. It began as the optimized
runtime behind the [OmniDreams closed-loop demo for GTC 2026][omnidreams-blog]
and has grown into a general platform for real-time world-model applications
across gaming, autonomous vehicles, robotics, simulated or virtual
environments, and more.

[omnidreams-blog]: https://research.nvidia.com/labs/sil/projects/omnidreams-blog/

<p align="center">
  <img src="docs/source/_static/diagrams/compare-offline-online-video-model-v2.jpg"
       alt="Offline one-shot video inference compared with online autoregressive world-model serving."
       width="100%">
</p>


## Quickstart

The complete setup is in
[the installation guide](https://verbose-adventure-7plnn5m.pages.github.io/main/quickstart/installation.html).
Assuming `uv` is [installed](https://docs.astral.sh/uv/getting-started/installation), the shortest viable path is:

```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
uv sync --extra dev --extra runners
export HF_TOKEN=<your-hf-token>
uv run flashdreams-run --help
```

Then launch your first model by following
[the quickstart guide](https://verbose-adventure-7plnn5m.pages.github.io/main/quickstart/first_world_model.html).
For example, the offline Self-Forcing T2V quickstart command is:

```bash
uv run --project integrations/self_forcing \
    flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash \
    --total-blocks 7
```

You can also install FlashDreams as a library from PyPI:

```bash
pip install flashdreams
```

## Supported models

FlashDreams ships first-party integrations under
[`integrations/`](integrations/). Each model has a dedicated docs page with
runner slugs, multi-GPU commands, and (where available) profiling benchmarks.

| Model | Family |
| --- | --- |
| [Self-Forcing](https://verbose-adventure-7plnn5m.pages.github.io/main/models/self_forcing.html) | Streaming Wan2.1 T2V |
| [OmniDreams](https://verbose-adventure-7plnn5m.pages.github.io/main/models/omnidreams.html) | HDMap-conditioned driving world model |
| [LingBot-World](https://verbose-adventure-7plnn5m.pages.github.io/main/models/lingbot_world.html) | Camera-controllable I2V world model |
| [Wan2.1](https://verbose-adventure-7plnn5m.pages.github.io/main/models/wan21.html) | Bidirectional T2V / I2V |
| [Causal-Forcing](https://verbose-adventure-7plnn5m.pages.github.io/main/models/causal_forcing.html) | Streaming Wan2.1 T2V / I2V |
| [Causal Wan2.2](https://verbose-adventure-7plnn5m.pages.github.io/main/models/causal_wan22.html) | FastVideo Causal Wan 2.2 14B MoE T2V |
| [FlashVSR](https://verbose-adventure-7plnn5m.pages.github.io/main/models/flashvsr.html) | Streaming video super-resolution |
| [Cosmos-Predict2.5](https://verbose-adventure-7plnn5m.pages.github.io/main/models/cosmos_predict2.html) | Bidirectional T2V / I2V |

See [the model gallery](https://verbose-adventure-7plnn5m.pages.github.io/main/models/index.html) and
[the new method guide](https://verbose-adventure-7plnn5m.pages.github.io/main/developer_guides/new_integration.html)
to add your own.

## Developer guides

- [Inference pipeline overview](https://verbose-adventure-7plnn5m.pages.github.io/main/developer_guides/inference_pipeline_overview.html)
- [Config system](https://verbose-adventure-7plnn5m.pages.github.io/main/developer_guides/config_system.html)
- [Add a new method](https://verbose-adventure-7plnn5m.pages.github.io/main/developer_guides/new_integration.html)

For day-to-day development:

```bash
uv run pre-commit run -a
uv run pytest -m "not manual"
```

See [`DEV.md`](DEV.md) for repository-specific workflow notes.

## Contributing

Bug reports, feature requests, performance work, new integrations, and
documentation fixes are all welcome. The full process — DCO sign-off, PR
expectations, coding conventions, and local-build speedups — is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

FlashDreams is released under the [Apache License 2.0](LICENSE). Third-party
components and their licenses are listed in
[`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) and [`NOTICE`](NOTICE). The
repository is REUSE-compliant; see [`REUSE.toml`](REUSE.toml) and
[`LICENSES/`](LICENSES/).

## Citation

If FlashDreams is useful in your research or product, please cite the project:

```bibtex
@misc{flashdreams2026,
  title        = {FlashDreams: High-performance inference and serving for
                  interactive autoregressive video and world models},
  author       = {{NVIDIA Corporation}},
  year         = {2026},
  howpublished = {\url{https://github.com/NVIDIA/flashdreams}},
}
```
