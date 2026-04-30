Recipes
===================================

Concrete model implementations live under ``flashdreams.recipes``. Each
recipe wires a checkpoint family into the
:doc:`infra <infra>` abstractions and exposes the resulting
``StreamInferencePipelineConfig`` factories that the example launchers
consume.

.. note::

   Recipe modules import the heavy GPU stack (transformer-engine, CUDA
   ops) at import time, so this page shows them by *automodule* with
   ``:no-undoc-members:`` to keep the rendered API focused on the names
   that recipes actually expose. The launcher scripts in
   ``flashdreams/examples`` show end-to-end usage.

AlpaDreams
----------

.. automodule:: flashdreams.recipes.alpadreams
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.alpadreams.config
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.alpadreams.pipeline
   :members:
   :no-undoc-members:
   :show-inheritance:

Wan
---

.. automodule:: flashdreams.recipes.wan
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.wan.pipeline
   :members:
   :no-undoc-members:
   :show-inheritance:

Lingbot-World
-------------

.. automodule:: flashdreams.recipes.lingbot_world
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.lingbot_world.config
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.lingbot_world.pipeline
   :members:
   :no-undoc-members:
   :show-inheritance:

TAEHV
-----

.. automodule:: flashdreams.recipes.taehv
   :members:
   :no-undoc-members:
   :show-inheritance:
