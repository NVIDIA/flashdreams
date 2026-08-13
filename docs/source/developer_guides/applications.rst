.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Application packages
====================

Application packages register a factory under ``flashdreams.applications``:

.. code-block:: toml

   [project.entry-points."flashdreams.applications"]
   my-model = "my_model:create_app"

The factory receives application-specific arguments and returns an object that
implements ``FlashDreamsApplication``:

.. code-block:: python

   def create_app(args):
       return MyApplication(args)

Applications initialize shared model state and create isolated sessions:

.. code-block:: python

   class MyApplication:
       default_io_handler = "local-window"

       def initialize(self, config):
           ...

       def create_session(self, launch_args):
           return MySession(...)

   class MySession:
       def generate(self, event, user_input):
           return frame_output

FlashDreams discovers an application and an IO handler, then passes both to the
core application runner. The runner owns application/session orchestration,
while the IO handler owns user-input and frame-output behavior.

IO packages register handler classes independently:

.. code-block:: toml

   [project.entry-points."flashdreams.io_handlers"]
   reactor = "reactor:ReactorIOHandler"

Adding an IO handler does not require changing a central mode enumeration or
application switch statement.

Invocation
----------

.. code-block:: bash

   flashdreams-run my-model local-window --model-flag value
   flashdreams-run my-model mp4 --output outputs/result.mp4
   flashdreams-run my-model webrtc --host 0.0.0.0 --port 8080

The optional positional value selects a registered IO handler. Each handler
consumes its own arguments; all remaining arguments are passed
unchanged to ``create_app``. When omitted, the application selects its default
IO handler.

Application and legacy runner names must not collide. Application entry points
are discovered without importing their modules, and only the selected factory
is loaded.
