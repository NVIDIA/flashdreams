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

"""Real-ESRGAN network architectures.

The RRDBNet and compact SRVGG definitions mirror the public Real-ESRGAN
and BasicSR architectures so upstream checkpoints can be loaded directly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Return the inverse pixel-shuffle transform for ``x``."""
    batch, channels, height, width = x.shape
    assert height % scale == 0 and width % scale == 0
    out_channels = channels * scale * scale
    height_out = height // scale
    width_out = width // scale
    x = x.view(batch, channels, height_out, scale, width_out, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(
        batch, out_channels, height_out, width_out
    )


def default_init_weights(module_list: list[nn.Module], scale: float = 1.0) -> None:
    """Initialize convolutional and linear weights in ``module_list``."""
    for module in module_list:
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight, a=0, mode="fan_in")
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.zero_()
            elif isinstance(child, nn.Linear):
                nn.init.kaiming_normal_(child.weight, a=0, mode="fan_in")
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.zero_()


class ResidualDenseBlock(nn.Module):
    """Residual dense block used by RRDBNet."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual dense block."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block used by RRDBNet."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual-in-residual block."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet used by RealESRGAN_x2plus and RealESRGAN_x4plus."""

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)
                for _ in range(num_block)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run RRDBNet on an RGB image batch in ``[0, 1]``."""
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2)))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2)))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style super-resolution network used by Real-ESRGAN v3."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._make_activation(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_activation(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"Unknown activation type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SRVGG on an RGB image batch in ``[0, 1]``."""
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base
