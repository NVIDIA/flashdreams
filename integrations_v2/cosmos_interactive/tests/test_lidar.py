# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit tokenizer architecture and semantic ray-grid boundary checks."""

from types import SimpleNamespace

import pytest
import torch
from cosmos_interactive import lidar
from cosmos_interactive._lidar_codec import RangeProjection, network_config
from cosmos_interactive._lidar_codec.range_image import metric_lidar_to_network

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "version,levels,width,tokens", [("v1p1", 3, 1800, 904), ("v1p2", 4, 1808, 228)]
)
@pytest.mark.parametrize("patch_dim", [128, 512])
@pytest.mark.parametrize("contiguous_linears", [False, True])
def test_codec_version_preserves_semantic_rays_across_streaming_boundary(
    monkeypatch, version, levels, width, tokens, patch_dim, contiguous_linears
):
    observed = {}

    def backend(checkpoint, *, device, version):
        observed.update(network_config(version))
        projection = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(projection.weight)
        return SimpleNamespace(
            model=torch.nn.Sequential(projection),
            range_projection=RangeProjection(model_width=observed["resolution"][1]),
            device=device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(lidar, "LidarCodec", backend)
    # A misleading filename must never select the latent basis.
    tokenizer = lidar.LidarTokenizer.from_checkpoint(
        "v1p1.pt", device="cpu", version=version, contiguous_linears=contiguous_linears
    )
    prepared = tokenizer
    tokenizer = prepared.with_patch_dim(patch_dim)
    linear = tokenizer.backend.model[0]
    assert len(linear._forward_pre_hooks) == int(contiguous_linears)
    linear.register_forward_pre_hook(
        lambda _module, inputs: observed.update(contiguous=inputs[0].is_contiguous())
    )
    x = torch.arange(24).float().reshape(2, 4, 3).transpose(1, 2)
    assert not x.is_contiguous() and torch.equal(linear(x), x)
    assert observed["contiguous"] == contiguous_linears
    assert tokenizer.backend is prepared.backend
    assert prepared.geometry.patch == 2
    g = tokenizer.geometry
    assert g.patch_dim == patch_dim
    assert g.spatial_tokens == (g.height * g.width if patch_dim == 128 else tokens)
    assert len(observed["depths"]) == levels
    assert observed["resolution"] == (128, width)
    if version == "v1p2":
        assert observed["num_heads"] == (4, 4, 8, 8)
    b = tokenizer.backend
    b.min_range, b.max_range, b.validity_threshold = 5, 100, 0.5
    b.latent_mean, b.latent_std = 0, 1

    def encode(x, cache):
        observed["encoded_canvas"] = x.clone()
        return None, (torch.zeros(1, g.channels, 1, g.height, g.width), None), 1

    def decode(z, cache):
        return observed["encoded_canvas"], 1

    b.model = SimpleNamespace(
        _limit_streaming_cache=lambda cache, context, **kw: cache,
        encode_stream=encode,
        decode_stream=decode,
    )
    raw = torch.ones(1, 3, 1, 128, 1800)
    raw[:, 0] = torch.linspace(5, 100, 1800)
    raw[:, 1] = 0.25
    normalized, _ = metric_lidar_to_network(raw, min_range=5, max_range=100)
    latent = lidar.LidarEncoder(tokenizer)(raw)
    canvas = observed["encoded_canvas"]
    assert canvas.shape[-1] == width
    if version == "v1p2":
        assert torch.equal(canvas[..., :4], normalized[..., -4:])
        assert torch.equal(canvas[..., -4:], normalized[..., :4])
    torch.testing.assert_close(lidar.LidarDecoder(tokenizer)(latent), raw)
    values = torch.arange(latent.numel()).reshape_as(latent).float()
    assert torch.equal(tokenizer.unpatchify(tokenizer.patchify(values)), values)
    with pytest.raises(ValueError, match="semantic ray grid"):
        tokenizer.to_model_canvas(raw[..., :-1])
    with pytest.raises(ValueError, match="model canvas"):
        tokenizer.to_semantic_canvas(canvas[..., :-1])
    with pytest.raises(ValueError, match="version"):
        lidar.LidarTokenizer.from_checkpoint("anything.pt", version="guessed")  # ty: ignore[invalid-argument-type]  # Exercise invalid-version validation.


@pytest.mark.parametrize("patch_dim", [-128, 0, 127, 129, 256, 513])
def test_diffusion_patch_width_must_fit_square_latent_patches(patch_dim):
    tokenizer = lidar.LidarTokenizer(None)
    with pytest.raises(ValueError, match="square LiDAR diffusion patches"):
        tokenizer.with_patch_dim(patch_dim)
