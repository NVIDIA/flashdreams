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


## Quickstart

The complete setup is in our
[quick-start guide](https://nvidia.github.io/flashdreams/main/quickstart/index.html#install).

## Checkout Our Demos! 🎉

Quick-start to our interactive driving demo is [here](https://nvidia.github.io/flashdreams/main/quickstart/index.html#run-your-first-model).

![](docs/source/_static/demo_clips/interactive-drive.avif)

Checkout our game-like crazy-robotaxi demo [here](https://nvidia.github.io/flashdreams/main/demos/crazy_robotaxi/).

![](docs/source/_static/demo_clips/crazy-robotaxi.avif)

## Developer guides

### For Demo Developers

Make demos using world-models with our [Demo API](https://nvidia.github.io/flashdreams/main/documentation/demo_api/).

### For Model Developers

Implement world-models with our [Inferencing API](https://nvidia.github.io/flashdreams/main/documentation/inferencing_api/).

## Contributing

For how to contribute, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
New integrations, bug reports, feature requests, performance tuning, and
documentation edits are all welcome.

Use [GitHub Issues](https://github.com/NVIDIA/flashdreams/issues) to report defects or request improvements.

Join us on the [NVIDIA Omniverse Discord](https://discord.com/invite/nvidiaomniverse)
to share your results and take part in technical discussion! Channel: [`#flashdreams`](https://discord.gg/cMt2mHm4aN)

## Security

To report a potential security vulnerability, follow the coordinated
disclosure process in [`SECURITY.md`](SECURITY.md).

## License

FlashDreams is released under the [Apache License 2.0](LICENSE). Third-party
components and their licenses are listed in
[`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) and [`NOTICE`](NOTICE). The
repository is REUSE-compliant; see [`REUSE.toml`](REUSE.toml) and
[`LICENSES/`](LICENSES/). The independently packaged
[Ludus renderer](integrations_v2/omnidreams/impl/ludus-renderer/LICENSE) retains its own Apache license file;
[upstream cudaraster terms](integrations_v2/omnidreams/impl/ludus-renderer/ludus_renderer/_cpp/cudaraster/LICENSE)
and per-file exceptions are described in the third-party notices.

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
  title         = {{NVIDIA} {OmniDreams}: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation},
  author        = {Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Wei, Fangyin and Ferroni, Francesco and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Lucas, James and Wu, Jay Zhangjie and Wang, Jialiang and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Micha{\l} and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Wu, Xindi and Ren, Xuanchi and Lu, Yifan and Zhang, Yuxuan and Gojcic, Zan and Wang, Zian},
  year          = {2026},
  eprint        = {2606.03159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.03159},
  url           = {https://arxiv.org/abs/2606.03159},
}
```

[omnidreams-blog]: https://research.nvidia.com/labs/sil/projects/omnidreams-blog/
