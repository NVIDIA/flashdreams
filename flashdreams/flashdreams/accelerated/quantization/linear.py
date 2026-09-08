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

"""Quantized nonpersistent linear transformation for accelerated inference."""

from collections.abc import Callable
from enum import Enum
from typing import overload

import torch
from torch import Tensor

from flashdreams.accelerated.common.non_persistent_linear import (
    NonPersistentLinear,
)
from flashdreams.accelerated.quantization.quantizer import (
    Granularity,
    dequantize,
    quantize,
)


class WeightGranularity(str, Enum):
    """Scale granularity for quantized linear weights."""

    PER_OUT_CHANNEL = "per_out_channel"
    """Use one scale for every output channel."""

    TENSOR = "tensor"
    """Use one scale for the complete weight tensor."""


class QuantizedNonPersistentLinear(NonPersistentLinear):
    """Apply a nonpersistent linear transformation with quantized weights.

    The layer quantizes and stores ``weight [O, I]`` during construction,
    where ``I`` is ``in_features`` and ``O`` is ``out_features``. At inference,
    it accepts activations ``x [..., I]`` and returns ``output [..., O]``. The
    leading activation dimensions may represent any combination of batch,
    sequence, or spatial dimensions.

    Examples:
        Construct a layer and let it quantize full-precision activations using
        one scale per ``I``-element slice::

            import torch

            from flashdreams.accelerated.quantization.linear import (
                QuantizedNonPersistentLinear,
                WeightGranularity,
            )
            from flashdreams.accelerated.quantization.quantizer import (
                Granularity,
                quantize,
            )

            weight = torch.randn(32, 64, device="cuda", dtype=torch.float16)
            bias = torch.randn(32, device="cuda", dtype=torch.float16)
            layer = QuantizedNonPersistentLinear(
                weight,
                bias,
                WeightGranularity.PER_OUT_CHANNEL,
                torch.float8_e4m3fn,
            )
            x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
            output = layer(x, Granularity.SLICE)
            assert output.shape == (2, 8, 32)

        Reuse prequantized activations and their scale to avoid quantizing
        ``x`` inside each call::

            quantized_x, x_scale = quantize(
                x,
                layer.dtype,
                Granularity.SLICE,
                axis=-1,
            )
            output = layer(quantized_x, x_scale, out_dtype=torch.float16)
            assert output.shape == (2, 8, 32)
    """

    dtype: torch.dtype
    """Quantized activation format required by prequantized inputs."""

    weight_scale: Tensor
    """FP32 weight scale shaped ``[O, 1]`` or ``[1, 1]``."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        granularity: WeightGranularity,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the transformation from existing tensors.

        Args:
            weight: Projection weight shaped ``[out_features, in_features]``.
            bias: Optional projection bias shaped ``[out_features]``.
            granularity: Scale granularity used to quantize ``weight``.
            dtype: Quantized activation format.

        Raises:
            ValueError: ``granularity`` or ``dtype`` is unsupported.
        """
        # Map the public weight modes onto ``quantize`` with ``axis=-1``:
        # ``[O, I] -> scale [O, 1]`` per output or ``scale [1, 1]`` per tensor.
        if granularity is WeightGranularity.PER_OUT_CHANNEL:
            quantizer_granularity = Granularity.SLICE
        elif granularity is WeightGranularity.TENSOR:
            quantizer_granularity = Granularity.TENSOR
        else:
            raise ValueError(f"unsupported weight granularity: {granularity}")

        # CUDA FP8 GEMM pairs E5M2 activations with an E4M3 weight operand.
        # All other formats use the same dtype for ``x`` and ``weight``.
        weight_dtype = torch.float8_e4m3fn if dtype is torch.float8_e5m2 else dtype
        quantized_weight, weight_scale = quantize(
            weight,
            weight_dtype,
            quantizer_granularity,
            axis=-1,
        )
        # Keep derived ``weight [O, I]`` and its scale out of ``state_dict``;
        # callers recreate both from the source weight when building the layer.
        super().__init__(quantized_weight.contiguous(), bias)
        self.dtype = dtype
        self.register_buffer(
            "weight_scale", weight_scale.contiguous(), persistent=False
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "QuantizedNonPersistentLinear":
        """Transform buffers without changing their quantization formats.

        Args:
            fn: Tensor transformation applied by :class:`torch.nn.Module`.
            recurse: Apply ``fn`` recursively to child modules.

        Returns:
            This module with transformed buffers.
        """
        weight = self.weight
        weight_scale = self.weight_scale
        module = super()._apply(fn, recurse=recurse)
        if self.weight.dtype is not weight.dtype:
            self.weight = weight.to(device=self.weight.device)
        if self.weight_scale.dtype is not weight_scale.dtype:
            self.weight_scale = weight_scale.to(device=self.weight_scale.device)
        return module

    @overload
    def forward(
        self,
        x: Tensor,
        scale_or_granularity: Tensor,
        out_dtype: torch.dtype = torch.float16,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        scale_or_granularity: Granularity,
        out_dtype: torch.dtype = torch.float16,
    ) -> Tensor: ...

    def forward(
        self,
        x: Tensor,
        scale_or_granularity: Tensor | Granularity,
        out_dtype: torch.dtype = torch.float16,
    ) -> Tensor:
        """Apply the quantized linear transformation.

        Args:
            x: Activations shaped ``[..., in_features]``. When a scale tensor
                is supplied, these must already use the layer's quantized
                ``dtype``; when a granularity is supplied, these are
                full-precision activations quantized inside this call.
            scale_or_granularity: For prequantized ``x``, an FP32 tensorwise
                scale shaped ``[1, ..., 1]`` or slice scale shaped
                ``[..., 1]``. For full-precision ``x``, the granularity used
                to produce one of those scale shapes.
            out_dtype: Data type of the projected activations. Defaults to
                ``torch.float16``.

        Returns:
            Projected activations shaped ``[..., out_features]``.

        Raises:
            ValueError: ``x`` or its scale does not match the layer contract.
        """
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {tuple(x.shape)}"
            )

        if isinstance(scale_or_granularity, Tensor):
            # Prequantized path: preserve ``x [..., I]`` and its existing
            # tensorwise ``scale [1, ..., 1]`` or slice ``scale [..., 1]``.
            if x.dtype is not self.dtype:
                raise ValueError(
                    f"expected quantized input dtype {self.dtype}, got {x.dtype}"
                )
            self._validate_scale(x, scale_or_granularity)
            quantized, scale = x, scale_or_granularity
        elif isinstance(scale_or_granularity, Granularity):
            # Dynamic path: quantization keeps ``x [..., I]`` unchanged in
            # shape and reduces either every dimension or only ``I`` for scale.
            if x.numel() == 0:
                return torch.empty(
                    (*x.shape[:-1], self.out_features),
                    device=x.device,
                    dtype=out_dtype,
                )
            quantized, scale = quantize(
                x,
                self.dtype,
                scale_or_granularity,
                axis=-1,
            )
        else:
            raise ValueError(
                "scale_or_granularity must be a scale tensor or Granularity"
            )

        return self._forward_quantized(quantized, scale, out_dtype)

    @staticmethod
    def _validate_scale(x: Tensor, scale: Tensor) -> None:
        """Validate a tensorwise ``[1, ..., 1]`` or slice ``[..., 1]`` scale."""
        tensor_shape = (1,) * x.ndim
        slice_shape = (*x.shape[:-1], 1)
        if scale.shape not in (tensor_shape, slice_shape):
            raise ValueError(
                f"expected scale shape {tensor_shape} or {slice_shape}, "
                f"got {tuple(scale.shape)}"
            )
        if scale.dtype is not torch.float32:
            raise ValueError(f"expected FP32 scale, got {scale.dtype}")
        if scale.device != x.device:
            raise ValueError(f"expected scale on device {x.device}, got {scale.device}")

    def _forward_quantized(
        self, x: Tensor, scale: Tensor, out_dtype: torch.dtype
    ) -> Tensor:
        """Project validated ``x [..., I]`` into ``output [..., O]``."""
        if x.numel() == 0:
            return torch.empty(
                (*x.shape[:-1], self.out_features),
                device=x.device,
                dtype=out_dtype,
            )

        # Collapse all leading dimensions into GEMM rows:
        # ``x [..., I] -> input_2d [R, I]``, where ``R = prod(x.shape[:-1])``.
        input_2d = x.reshape(-1, self.in_features).contiguous()
        # A slice scale ``[..., 1]`` follows the same collapse to ``[R, 1]``;
        # a tensor scale ``[1, ..., 1]`` becomes the scalar matrix ``[1, 1]``.
        input_scale = scale.reshape(-1, 1).contiguous()
        # Transpose weight scales from quantizer layout ``[O, 1]`` or
        # ``[1, 1]`` to GEMM output-column layout ``[1, O]`` or ``[1, 1]``.
        weight_scale = self.weight_scale.T.contiguous()

        if self.dtype is torch.int8:
            # Multiply ``[R, I] @ [I, O] -> int32 [R, O]``, then broadcast
            # activation scales down rows and weight scales across columns.
            input_rows = input_2d.shape[0]
            if input_rows <= 16:
                # CUDA ``_int_mm`` requires more than 16 rows. Zero-padding the
                # integer GEMM and slicing before dequantization preserves the
                # original activation scales and output exactly.
                input_2d = torch.cat(
                    (
                        input_2d,
                        input_2d.new_zeros(17 - input_rows, self.in_features),
                    )
                )
            output = dequantize(
                torch._int_mm(input_2d, self.weight.T)[:input_rows],
                input_scale,
                weight_scale,
                dtype=out_dtype,
            )
        else:
            # ``_scaled_mm`` accepts either two ``[1, 1]`` scales or rowwise
            # ``[R, 1]`` and ``[1, O]`` scales. Expand only the scalar side for
            # mixed granularities; repeated values preserve its tensor scale.
            if input_scale.numel() == 1 and weight_scale.numel() != 1:
                input_scale = input_scale.expand(input_2d.shape[0], 1).contiguous()
            elif weight_scale.numel() == 1 and input_scale.numel() != 1:
                weight_scale = weight_scale.expand(1, self.out_features).contiguous()

            rowwise_scaling = input_scale.numel() != 1 or weight_scale.numel() != 1
            # The CUDA rowwise path produces reliable high-precision ``[R, O]``
            # through BF16; cast after GEMM when the caller requests otherwise.
            scaled_out_dtype = torch.bfloat16 if rowwise_scaling else out_dtype
            # Multiply ``input_2d [R, I]`` by ``weight.T [I, O]`` and fuse both
            # dequantization scales into the resulting ``output [R, O]``.
            output = torch._scaled_mm(
                input_2d,
                self.weight.T,
                input_scale,
                weight_scale,
                out_dtype=scaled_out_dtype,
            )
            if output.dtype is not out_dtype:
                output = output.to(out_dtype)

        # Broadcast ``bias [O]`` over the ``R`` rows, then restore the original
        # leading dimensions: ``output [R, O] -> [..., O]``.
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=out_dtype)
        return output.reshape(*x.shape[:-1], self.out_features)
