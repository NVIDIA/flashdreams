# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native acceleration helpers for OmniDreams integrations."""

from .acceleration import (
    NativeAccelerationConfig,
    NativeAccelerationMode,
    NativeAccelerationUnavailable,
    NativeBackendSelection,
    require_extension_symbols,
    select_native_extension,
)
from .omnidreams_singleview import (
    build_info,
    load_extension,
    select_backend,
    sync_thirdparty,
    validate_thirdparty,
)
from .primitives import (
    NativePreparedTensor,
    NativePrepError,
    NativeTensorDescriptor,
    NativeTensorLayout,
    NativeTensorSpec,
    prepare_tensor_for_native,
)

__all__ = [
    "NativeAccelerationConfig",
    "NativeAccelerationMode",
    "NativeAccelerationUnavailable",
    "NativeBackendSelection",
    "NativePrepError",
    "NativePreparedTensor",
    "NativeTensorDescriptor",
    "NativeTensorLayout",
    "NativeTensorSpec",
    "build_info",
    "load_extension",
    "prepare_tensor_for_native",
    "require_extension_symbols",
    "select_backend",
    "select_native_extension",
    "sync_thirdparty",
    "validate_thirdparty",
]
