# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Color correctors for the FlashVSR decoder tail.

Two implementations share the same ``(hq, lq, ...) -> hq'`` interface:

- :class:`_TorchColorCorrectorWavelet` -- pure-torch wavelet (default
  method) and AdaIN reference path; slow but portable, used for parity
  testing.
- :class:`_CudaColorCorrectorAdaIN` -- a hand-rolled CUDA AdaIN kernel
  (see ``csrc/color_corrector_adain_cuda.cu``) used in production.
  Wavelet is unsupported here.

:class:`FlashVSRColorCorrector` is the public dispatcher; it picks one
of the two implementations from the ``implementation`` knob (default
``"cuda"``).
"""

import hashlib
from importlib.resources import as_file, files
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load as _load_cuda_extension

_ADAIN_EPS = 1e-5
"""Variance floor used by the AdaIN normalization step (both backends).
Matches the legacy FlashVSR upsampler's literal; small enough that
unit-variance content is not visibly biased, large enough to clamp
near-zero variance in flat patches."""
_DISABLE_TORCH_COMPILE = getattr(
    getattr(torch, "compiler", None), "disable", lambda fn: fn
)
_ADAIN_CUDA_EXTENSION = None
_ADAIN_CUDA_EXTENSION_LOAD_ERROR: Optional[Exception] = None
ColorCorrectorImplementation = Literal["torch", "cuda"]
_ADAIN_CUDA_SOURCE_RESOURCE = (
    files("flashvsr").joinpath("csrc").joinpath("color_corrector_adain_cuda.cu")
)


def _load_adain_cuda_extension():
    global _ADAIN_CUDA_EXTENSION, _ADAIN_CUDA_EXTENSION_LOAD_ERROR
    if _ADAIN_CUDA_EXTENSION is not None:
        return _ADAIN_CUDA_EXTENSION
    if _ADAIN_CUDA_EXTENSION_LOAD_ERROR is not None:
        return None

    try:
        with as_file(_ADAIN_CUDA_SOURCE_RESOURCE) as source_path:
            # Hash the source bytes into the extension name so any edit to
            # ``color_corrector_adain_cuda.cu`` invalidates the cached ``.so``
            # under ``~/.cache/torch_extensions``. Without this suffix
            # ``torch.utils.cpp_extension.load`` would silently reuse a stale build
            # whenever the source contents change but the file timestamp doesn't
            # advance enough to trigger its content-only check.
            csrc_checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()[:8]
            _ADAIN_CUDA_EXTENSION = _load_cuda_extension(
                name=f"flashvsr_adain_cuda_{csrc_checksum}",
                sources=[str(source_path)],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
    except Exception as exc:
        _ADAIN_CUDA_EXTENSION_LOAD_ERROR = exc
        return None
    return _ADAIN_CUDA_EXTENSION


def _calc_mean_std(
    feat: torch.Tensor, eps: float = _ADAIN_EPS
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, "feat must be (N, C, H, W)"
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N and C must match"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _can_use_cuda_adain_5d(
    content_feat: torch.Tensor, style_feat: torch.Tensor
) -> bool:
    if not content_feat.is_cuda or not style_feat.is_cuda:
        return False
    if content_feat.shape != style_feat.shape or content_feat.dim() != 5:
        return False
    if content_feat.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if style_feat.dtype != content_feat.dtype:
        return False
    return content_feat.shape[-1] > 0 and content_feat.shape[-2] > 0


@_DISABLE_TORCH_COMPILE
def _adain_cuda_5d(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    if not _can_use_cuda_adain_5d(content_feat, style_feat):
        raise ValueError(
            "CUDA AdaIN requires same-shaped CUDA tensors with shape "
            "(B, 3, f, H, W) and dtype float16, bfloat16, or float32"
        )

    adain_cuda = _load_adain_cuda_extension()
    if adain_cuda is None:
        raise RuntimeError(
            "Native CUDA AdaIN extension is unavailable"
        ) from _ADAIN_CUDA_EXTENSION_LOAD_ERROR

    return adain_cuda.adain_forward_5d(
        content_feat,
        style_feat,
        float(clip_range[0]),
        float(clip_range[1]),
        _ADAIN_EPS,
    )


def adain_cuda_caps() -> dict:
    """Return the device capabilities the AdaIN CUDA extension is using.

    Keys: ``cooperative_launch`` (bool), ``persisting_l2`` (bool),
    ``num_sms`` (int), ``access_policy_max_window_size`` (int),
    ``persisting_l2_max_size`` (int). Empty dict if the extension is not
    available on this host.
    """
    ext = _load_adain_cuda_extension()
    if ext is None or not hasattr(ext, "caps"):
        return {}
    return ext.caps()


def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, "x must be (N, C, H, W)"
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(
        x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C
    )
    return out


def _wavelet_decompose(
    x: torch.Tensor, levels: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4, "x must be (N, C, H, W)"
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2**i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(
    content: torch.Tensor, style: torch.Tensor, levels: int = 5
) -> torch.Tensor:
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


class _TorchColorCorrectorWavelet(nn.Module):
    """Pure-torch reference: wavelet color transfer + AdaIN fallback.

    Pre-flattens the time axis so the underlying ``_wavelet_reconstruct`` /
    ``_adain`` helpers operate on 4-D ``(N, C, H, W)`` tensors. Used as
    the parity reference for :class:`_CudaColorCorrectorAdaIN`.
    """

    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        assert x.dim() == 5, "Input must be (B, C, f, H, W)"
        B, C, f, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * f, C, H, W)
        return y, B, f

    @staticmethod
    def _unflatten_time(y: torch.Tensor, B: int, f: int) -> torch.Tensor:
        BF, C, H, W = y.shape
        assert BF == B * f
        return y.reshape(B, f, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Color-correct ``hq_image`` toward ``lq_image`` over the time axis.

        Args:
            hq_image: HQ content ``[B, 3, f, H, W]``.
            lq_image: LQ reference ``[B, 3, f, H, W]`` (same shape as ``hq_image``).
            clip_range: Output clamp range; defaults to ``(-1.0, 1.0)``.
            method: ``"wavelet"`` (default) or ``"adain"``.
            chunk_size: Process the time axis in chunks of this many
                frames; ``None`` or ``>= f`` runs the whole clip in one
                shot.

        Returns:
            Corrected ``[B, 3, f, H, W]`` tensor in ``clip_range``.
        """
        assert hq_image.shape == lq_image.shape, (
            f"HQ and LQ shapes must match, but got {hq_image.shape} and {lq_image.shape}"
        )
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3, (
            "Input must be (B, 3, f, H, W)"
        )

        B, C, f, H, W = hq_image.shape
        if chunk_size is None or chunk_size >= f:
            hq4, B, f = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
                out4 = torch.clamp(out4, *clip_range)
            elif method == "adain":
                out4 = torch.clamp(_adain(hq4, lq4), *clip_range)
            else:
                raise ValueError(f"Unknown method: {method}")
            out = self._unflatten_time(out4, B, f)
            return out

        outs = []
        for start in range(0, f, chunk_size):
            end = min(start + chunk_size, f)
            hq_chunk = hq_image[:, :, start:end]
            lq_chunk = lq_image[:, :, start:end]
            hq4, B_, f_ = self._flatten_time(hq_chunk)
            lq4, _, _ = self._flatten_time(lq_chunk)
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
                out4 = torch.clamp(out4, *clip_range)
            elif method == "adain":
                out4 = torch.clamp(_adain(hq4, lq4), *clip_range)
            else:
                raise ValueError(f"Unknown method: {method}")
            out_chunk = self._unflatten_time(out4, B_, f_)
            outs.append(out_chunk)
        return torch.cat(outs, dim=2)


class _CudaColorCorrectorAdaIN(nn.Module):
    """Hand-rolled CUDA AdaIN kernel (no wavelet support).

    Dispatches to ``csrc/color_corrector_adain_cuda.cu`` once the
    extension is loaded. Inputs must be matching-shape, contiguous,
    ``(B, 3, f, H, W)`` tensors on CUDA. ``method != "adain"`` raises.
    """

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """AdaIN-correct ``hq_image`` toward ``lq_image`` on CUDA.

        Args:
            hq_image: HQ content ``[B, 3, f, H, W]``.
            lq_image: LQ reference ``[B, 3, f, H, W]`` (same shape as ``hq_image``).
            clip_range: Output clamp range; defaults to ``(-1.0, 1.0)``.
            method: Must be ``"adain"``; ``"wavelet"`` is not supported here.
            chunk_size: Unused; accepted only for interface parity with
                the torch backend.

        Raises:
            ValueError: ``method != "adain"``.
        """
        del chunk_size
        assert hq_image.shape == lq_image.shape, (
            f"HQ and LQ shapes must match, but got {hq_image.shape} and {lq_image.shape}"
        )
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3, (
            "Input must be (B, 3, f, H, W)"
        )
        if method != "adain":
            raise ValueError("CUDA color corrector only supports method='adain'")
        return _adain_cuda_5d(hq_image, lq_image, clip_range)


class FlashVSRColorCorrector(nn.Module):
    """Public color-corrector dispatcher (wavelet + AdaIN backends).

    The default ``implementation="cuda"`` is AdaIN-only; pick
    ``implementation="torch"`` for the wavelet reference. Forwards all
    ``forward`` kwargs verbatim to the chosen backend.
    """

    def __init__(
        self,
        levels: int = 5,
        implementation: ColorCorrectorImplementation = "cuda",
    ):
        super().__init__()
        if implementation == "torch":
            self.impl = _TorchColorCorrectorWavelet(levels=levels)
        elif implementation == "cuda":
            self.impl = _CudaColorCorrectorAdaIN()
        else:
            raise ValueError(
                f"Unknown color corrector implementation: {implementation}"
            )
        self.implementation: ColorCorrectorImplementation = implementation

    def forward(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Dispatch to the configured ``"torch"`` or ``"cuda"`` backend."""
        return self.impl(
            hq_image,
            lq_image,
            clip_range=clip_range,
            method=method,
            chunk_size=chunk_size,
        )
