<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Single-View Native

This directory contains native-source scaffolding for the OmniDreams single-view
integration. It hosts the managed third-party source manifest, synchronization
tooling, build helpers, and CUDA/C++ extension sources used by the
single-view-only native acceleration path.

This code is a work in progress. It is intended as an internal implementation
area for bringing up and validating native OmniDreams acceleration, not as a
stable user-facing API. Interfaces, file layout, and kernel coverage may change
as the single-view native path is developed.
