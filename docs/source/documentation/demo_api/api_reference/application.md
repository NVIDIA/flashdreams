---
title: 'Demo Application API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This page documents the public application protocols in `flashdreams.api_v2`
and the runtime-owned value types those protocols exchange. For implementation
workflows, start with [Create a demo](../guides/create_demo.md) or
[Integrate model adapter for a demo](../guides/integrate_model_adapter_for_a_demo.md).

## Application and session

::: flashdreams.api_v2.application.IApplication

::: flashdreams.api_v2.session.ISession

## Model and UI loops

::: flashdreams.api_v2.loop.IModelLoop

::: flashdreams.api_v2.loop.IUILoop

::: flashdreams.runtime_v2.imgui_ui_loop.ImGuiUILoop

## Session description

::: flashdreams.runtime_v2.session_desc.SessionDesc

::: flashdreams.runtime_v2.session_desc.BackpressureMode

::: flashdreams.runtime_v2.session_desc.PresentationMode
