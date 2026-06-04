<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/horizontal-dark.svg">
    <img alt="FlashDreams" src="assets/logo/horizontal-light.svg" width="600">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://nvidia.github.io/flashdreams/main/index.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
</p>

**FlashDreams** is a high-performance inference and serving library for
interactive autoregressive video and world models. It began as the optimized
runtime behind the [NVIDIA OmniDreams closed-loop demo for GTC 2026][omnidreams-blog]
and has grown into a general platform for real-time world-model applications
across gaming, autonomous vehicles, robotics, simulated or virtual
environments, and more.

[omnidreams-blog]: https://research.nvidia.com/labs/sil/projects/omnidreams-blog/

https://github.com/user-attachments/assets/2b000ce9-effe-4cc9-a227-5b4619413e4d

## System Requirements

- NVIDIA GPU with **80 GB VRAM or more** (e.g. H100 80GB), see notes below.
- NVIDIA driver from the **R580 series or newer** (compatible with CUDA 13.x)
- **CUDA 13.x** (PyTorch `2.11.0+cu130` and the `nvidia-*-cu13` libraries are
  resolved by `uv sync`. A system CUDA toolkit is needed only for the
  developer extras and is included in `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`)
- **Python >= 3.10**
- **PyTorch >= 2.11.0+cu130** (`>= 2.9` for bare PyPI library install)
- Linux x86-64 or arm64
- **100 GB+ free storage space** recommended for environment and model checkpoints.
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (optional, only for the container workflow)

> Development and testing were performed on GPUs with **80 GB of VRAM or more**.
> Inference can fail (out-of-memory) on consumer and even enthusiast GPUs.
> Per-model GPU and VRAM requirements are listed on each model page in
> [the model gallery](https://nvidia.github.io/flashdreams/main/models/index.html).

## Quickstart

The complete setup is in
[the installation guide](https://nvidia.github.io/flashdreams/main/quickstart/installation.html).
Assuming `uv` is [installed](https://docs.astral.sh/uv/getting-started/installation), the shortest viable path is:

```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
uv sync --extra runners
export HF_TOKEN=<your-hf-token>
uv run flashdreams-run --help
```

Note for developers/maintainers you would want to run `uv sync --extra dev --extra runners` instead.

Then launch your first model by following
[the quickstart guide](https://nvidia.github.io/flashdreams/main/quickstart/first_world_model.html).
For example, the offline Self-Forcing T2V quickstart command is:

```bash
uv run --project integrations/self_forcing \
    flashdreams-run self-forcing-wan2.1-t2v-1.3b \
    --total-blocks 7
```

You can also install FlashDreams as a library from PyPI:

```bash
pip install flashdreams
```

### Try the interactive driving demo

Drive a world model in real time with the OmniDreams `interactive-drive` demo. See the
**[interactive demo guide](https://nvidia.github.io/flashdreams/main/models/omnidreams.html#launch-the-interactive-demo)**.

## Supported models

FlashDreams ships first-party integrations under
[`integrations/`](integrations/). Each model has a dedicated docs page with
runner slugs, multi-GPU commands, and (where available) profiling benchmarks.

| Model | Family |
| --- | --- |
| [Self-Forcing](https://nvidia.github.io/flashdreams/main/models/self_forcing.html) | Streaming Wan2.1 T2V |
| [OmniDreams](https://nvidia.github.io/flashdreams/main/models/omnidreams.html) | HDMap-conditioned driving world model |
| [LingBot-World](https://nvidia.github.io/flashdreams/main/models/lingbot_world.html) | Camera-controllable I2V world model |
| [Wan2.1](https://nvidia.github.io/flashdreams/main/models/wan21.html) | Bidirectional T2V / I2V |
| [Causal-Forcing](https://nvidia.github.io/flashdreams/main/models/causal_forcing.html) | Streaming Wan2.1 T2V / I2V |
| [Causal Wan2.2](https://nvidia.github.io/flashdreams/main/models/causal_wan22.html) | FastVideo Causal Wan 2.2 14B MoE T2V |
| [FlashVSR](https://nvidia.github.io/flashdreams/main/models/flashvsr.html) | Streaming video super-resolution |
| [Cosmos-Predict2.5](https://nvidia.github.io/flashdreams/main/models/cosmos_predict2.html) | Bidirectional T2V / I2V |

See [the model gallery](https://nvidia.github.io/flashdreams/main/models/index.html) and
[the new method guide](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
to add your own.

## Developer guides

- [Inference pipeline overview](https://nvidia.github.io/flashdreams/main/developer_guides/inference_pipeline_overview.html)
- [Config system](https://nvidia.github.io/flashdreams/main/developer_guides/config_system.html)
- [Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)

For day-to-day development:

```bash
uv sync --extra dev --extra runners
uv run pre-commit run -a
uv run pytest -m "not manual"
```

See [`DEV.md`](DEV.md) for repository-specific workflow notes.

## Contributing

For how to contribute, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
New integrations, bug reports, feature requests, performance tuning, and
documentation edits are all welcome.

Use [GitHub Issues](https://github.com/NVIDIA/flashdreams/issues) to report defects or request improvements.

Join us on the [NVIDIA Omniverse Discord](https://discord.com/invite/nvidiaomniverse)
to share your results and take part in technical discussion! Channel: [`#flashdreams`](https://discord.gg/yTdHDqFP)

## Security

To report a potential security vulnerability, follow the coordinated
disclosure process in [`SECURITY.md`](SECURITY.md).

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
  author       = {{FlashDreams Contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/NVIDIA/flashdreams}},
}

@misc{nvidia2026omnidreams,
  title={OmniDreams: Real-Time Generative Closed-Loop Autonomous Vehicle Simulation Built on NVIDIA Cosmos},
  author={Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Wang, Jialiang and Lucas, James and Wu, Jay and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Michal and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Ren, Xuanchi and Lu, Yifan and Gojcic, Zan and Wang, Zian},
  year={2026},
  note={Technical report},
  howpublished={\url{https://research.nvidia.com/labs/sil/projects/omnidreams-blog/paper.pdf}}
}
```
