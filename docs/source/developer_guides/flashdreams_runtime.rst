.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

FlashDreams runtime architecture
================================

The FlashDreams runtime connects interactive controls to an autoregressive
inference pipeline and routes the generated frames to an output destination.
This page defines the target architecture and the boundaries between its
components. It is a design contract rather than a reference for an existing
Python API.

Architecture overview
---------------------

.. figure:: /_static/diagrams/flashdreams-runtime.png
   :alt: FlashDreams runtime architecture and its input and output data flow.
   :align: center
   :width: 100%
   :class: zoomable

   The application owns the runtime components. Dashed arrows show data flow;
   solid lines with diamonds show ownership. Solid boxes are classes and
   dashed boxes are data.

.. image:: /_static/diagrams/flashdreams-runtime-data-flow.png
   :alt: FlashDreams user and global conditioning data flow into an inference session.
   :align: center
   :width: 70%
   :class: zoomable

Application layer
-----------------

``Application`` is the composition and lifecycle boundary for an interactive
FlashDreams runtime. It owns one ``InputSystem``, one ``InputMapping``, one
``OutputTarget``, and the main ``InferenceSession`` that runs the inference
pipeline. These components are passed to the application through dependency
injection. The application connects them, drives the runtime loop, and shuts
them down in a defined order.

Keeping orchestration in the application gives each child component a narrow
responsibility. Device handling stays out of model execution, model-specific
input conversion stays out of device handling, and presentation stays out of
the inference session.

InputSystem
~~~~~~~~~~~

``InputSystem`` owns the interaction with input devices and converts their
events into a canonical control representation. Raw input can come from many
sources, including keyboard events, a digital steering wheel, a controller
joystick, or Meta Quest hand tracking.

`Unity's Input System
<https://docs.unity3d.com/Packages/com.unity.inputsystem@1.20/manual/index.html>`_
provides similar concepts for configuring input devices and actions. In
FlashDreams, users can configure arbitrary key bindings through
``InputSystem``, which converts device signals into canonicalized user input.

.. admonition:: Example
   :class: note

   Either WASD or HJKL can be bound to movement directions and mapped into a
   2D character-movement vector.

.. admonition:: Queued events between pulls
   :class: note

   A call to ``InferenceSession.step()`` can take approximately 100--1000 ms
   because it runs a latent-diffusion step. ``InputSystem`` must therefore
   preserve every input change that occurs while inference is running rather
   than returning only the most recent device state. It queues all events since
   the previous ``InputSystem`` pull and returns them as an ordered list of
   timestamped, canonicalized user-input events.

   For example, assume positive *x* means right and positive *y* means forward.
   The user presses W at 5 ms, presses D at 10 ms, releases W at 50 ms, presses
   S at 60 ms, releases D at 70 ms, and releases S at 80 ms. The next pull
   returns the following canonical movement states:

   .. code-block:: text

      [
          ( 5 ms, vec2(0,  1)),  # W pressed
          (10 ms, vec2(1,  1)),  # D pressed; W remains pressed
          (50 ms, vec2(1,  0)),  # W released
          (60 ms, vec2(1, -1)),  # S pressed; D remains pressed
          (70 ms, vec2(0, -1)),  # D released
          (80 ms, vec2(0,  0)),  # S released
      ]

   The timestamps are the original event times, not the time at which the
   application eventually calls ``pull()``.

This layer handles device-facing concerns such as key bindings, dead zones,
axis conventions, and event sampling. Its output describes the user's intent
in a stable, device-independent form. It does not create model embeddings or
know how a particular inference pipeline represents conditioning.

InputMapping
~~~~~~~~~~~~

``InputMapping`` consumes the ordered list of timestamped, canonicalized
user-input events returned by ``InputSystem`` and produces the model-ready,
per-step inference conditioning expected by an ``InferenceSession``. Depending
on the model, this conversion can include embedding control values, rendering
a control representation, changing layouts, or assembling tensors.

.. admonition:: Example
   :class: note

   ``integrations/omnidreams`` represents canonicalized driving input as a
   floating-point steering-wheel angle and a floating-point paddle/brake value.
   Its ``InputMapping`` runs the vehicle-dynamics simulation, renders the
   resulting HD map with the Ludus renderer, and produces the rendered RGB HD
   map as the per-step user-input condition. The resulting tensor has shape
   ``(3, H, W)``.

The ``(3, H, W)`` output is specific to OmniDreams, not a universal
``InputMapping`` contract. Another ``InferenceSession`` might expect an image
embedding as its condition. In that case, ``InputMapping`` can use an image
encoder to encode the frame and return the resulting embedding instead.

This is the boundary between application-level control semantics and
model-specific conditioning. Replacing a keyboard with a controller should
usually affect the ``InputSystem``; replacing the model or its control encoder
should usually affect the ``InputMapping``.

OutputTarget
~~~~~~~~~~~~

``OutputTarget`` consumes the ``FrameStream`` produced by the inference
session. A target can present frames in a native window, send them to a video
encoder, publish them through a WebRTC host, or adapt them for another output
system.

The output target owns presentation and transport concerns. It must not be
responsible for interpreting user controls or advancing model inference.
Buffering and backpressure policies belong at this output boundary so that a
slow consumer does not silently redefine inference behavior.

InferenceSession
~~~~~~~~~~~~~~~~

``InferenceSession`` is the execution boundary for the main inference
pipeline. It accepts inference input, maintains the state required across
autoregressive steps, runs the pipeline, and exposes generated output as a
``FrameStream``.

The session receives model-ready data only. It does not poll devices,
canonicalize user intent, or present generated frames. After accepting global
inference conditioning, it retains the active global condition across later
steps until the application supplies an update or the session ends.

Input data flow
---------------

The input path deliberately separates physical device readings, semantic
controls, and model-ready conditioning. This separation allows devices and
models to evolve independently.

User conditioning
~~~~~~~~~~~~~~~~~

Raw user input
^^^^^^^^^^^^^^

**Raw user input** is a reading or event in the vocabulary of a physical input
source. Examples include:

* WASD key presses and releases;
* digital wheel input;
* controller joystick readings; and
* Meta Quest hand-tracking readings.

Raw values can depend on a particular device, driver, sampling rate, or key
binding. They are consumed by ``InputSystem`` and must not be passed directly
to ``InferenceSession``.

Canonicalized user input
^^^^^^^^^^^^^^^^^^^^^^^^

**Canonicalized user input** expresses user intent in the vocabulary of the
interaction, independent of the device that produced it. Typical structures
include:

* a 2D character-movement vector and a 2D camera-movement vector for
  character-control games;
* a floating-point wheel value and a floating-point paddle value for driving;
  and
* hand-tracking positions, correction vectors, or another agreed semantic hand
  control for Cosmos-style interaction models.

The exact structure depends on the interaction type and can evolve as its
semantics become clearer. The important invariant is that equivalent intent
from different devices has the same canonical representation.

Per-step inference conditioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Per-step inference conditioning** is the model-ready encoding of
canonicalized user input for one inference step. ``InputMapping`` produces it by
performing whatever embedding, rendering, or tensor conversion the selected
model requires.

This term distinguishes the changing user control for one step from global
conditioning, which normally remains stable across many steps.

Inference input
^^^^^^^^^^^^^^^

**Inference input** is the complete input delivered to ``InferenceSession``.
It is the runtime boundary object, not another name for a raw or canonicalized
control. It can carry:

* the per-step inference conditioning; and
* optional global inference conditioning.

The first inference step generally carries both. On later steps, the global
condition remains active inside the session, so the application normally sends
only new per-step inference conditioning. Omitting global conditioning means
"continue using the active global condition"; it must not mean "clear the
global condition."

Global conditioning
~~~~~~~~~~~~~~~~~~~

Global conditioning establishes the scene-level context for generation and can
contain model-specific data. Two of the most common examples are:

* a **global conditioning frame**, sometimes called an initial frame by a
  model; and
* a **global conditioning prompt**, containing the text description for the
  run.

The runtime uses *global conditioning frame* instead of *initial frame*
because the condition is not inherently limited to initialization. A future
runtime can replace it while a session is already running.

Raw and canonicalized global conditioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Raw global conditioning** is the application-facing representation. For
example, the prompt is text and the conditioning frame is image data.

**Canonicalized global conditioning** is the model-ready representation sent
through inference input as **global inference conditioning**. A text prompt is
typically converted into embedded tokens. A conditioning frame is
model-dependent: one model might convert it into CLIP embeddings, while
another might retain a frame or spatial representation such as an HD-map
condition.

For that reason, *canonicalized* is preferred over *embedded* for the combined
global condition. It does not incorrectly imply that every part of the global
condition must become an embedding.

Updating global conditioning during a run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although mid-run global-conditioning updates are not implemented yet, the
runtime contracts must leave room for them. An application could, for example,
submit a new conditioning frame and prompt to make an OmniDreams driving scene
transition suddenly to rainy weather.

A later inference input can therefore carry new global inference conditioning
alongside its per-step conditioning. The session then treats it as a change to
the active global context. When no update is present, the session reuses the
previous context. The exact effects on model history, caches, and transition
behavior are model-specific and must be defined by the corresponding pipeline;
the application-level contract must not assume that global conditioning is
initialization-only.

End-to-end runtime loop
-----------------------

At a conceptual level, one runtime iteration follows these steps:

#. ``Application`` pulls ``InputSystem`` for the events accumulated since the
   previous pull.
#. ``InputSystem`` returns an ordered list of timestamped, canonicalized
   user-input events.
#. ``InputMapping`` consumes this list of timestamped, canonicalized events and
   produces per-step inference conditioning.
#. ``Application`` packages that data as inference input, adding canonicalized
   global conditioning on the first step or whenever it changes.
#. ``InferenceSession`` advances the pipeline and emits generated frames
   through ``FrameStream``.
#. ``OutputTarget`` consumes the stream for display, encoding, transport, or
   another presentation path.

These boundaries are the central architectural constraint: input devices
produce semantic controls, the input map produces model-ready conditioning,
the inference session runs the model, and the output target delivers the
result.
