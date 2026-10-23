# LiDAR codec source and licenses

This package vendors LiDAR inference components from
[vLLM-Omni](https://github.com/MaciejBalaNV/vllm-omni/tree/8ef7df57219688835f3752accb2c3b2f444b4058/vllm_omni/diffusion/models/cosmos3/lidar_encoder),
branch `cosmos3_gtc_berlin_release`, revision
`8ef7df57219688835f3752accb2c3b2f444b4058`.

| Local file | Upstream file | Local changes |
| --- | --- | --- |
| `transformer_vae.py` | `transformer_vae.py` | Relative imports, CPU constructor shape arithmetic for meta-device inspection, inherited MIT attribution |
| `rope.py` | `rope3d.py` | None |
| `encoding.py` | `encoding.py` | None |
| `neighborhood_attention.py` | `neighborhood_attention.py` | None |

The upstream files retain vLLM-Omni contributor copyrights and Apache-2.0
identifiers. The complete [upstream Apache license](https://github.com/MaciejBalaNV/vllm-omni/blob/8ef7df57219688835f3752accb2c3b2f444b4058/LICENSE)
is included as [LICENSE.vllm-omni](LICENSE.vllm-omni).

The VAE credits k-diffusion's `image_transformer_v2.py`. Its inherited MIT
notice, Copyright (c) 2022 Katherine Crowson, is included as
[LICENSE.k-diffusion](LICENSE.k-diffusion), verified against the
[pinned upstream license](https://github.com/crowsonkb/k-diffusion/blob/e95849c23340e78c4bca5ef6cea6dd60611acad8/LICENSE).
The adapted VAE records `Apache-2.0 AND MIT`. The upstream snapshot has no
NOTICE file; the local [NOTICE](NOTICE) identifies this redistribution.
Both license texts, NOTICE and this README accompany the packaged codec.

`network.py` supplies the FlashDreams checkpoint and streaming interface.
It constructs the vendored encoder and decoder directly, preserves V1.1/V1.2
checkpoint layouts and bounded streaming caches, and uses compiled PyTorch
FlexAttention for spatial attention. The codec requires no vLLM runtime,
NATTEN or training checkout. Compatible model weights are supplied separately
under their own terms.

Semantic-map sampling and spinning-LiDAR projection are separate,
independently written components in the prepared-scene integration.
