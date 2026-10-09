// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef OMNIDREAMS_NATIVE_HOST_DEVICE
#ifdef __CUDACC__
#define OMNIDREAMS_NATIVE_HOST_DEVICE __host__ __device__
#else
#define OMNIDREAMS_NATIVE_HOST_DEVICE
#endif
#endif

#ifndef OMNIDREAMS_NATIVE_DEVICE
#ifdef __CUDACC__
#define OMNIDREAMS_NATIVE_DEVICE __device__
#else
#define OMNIDREAMS_NATIVE_DEVICE
#endif
#endif
