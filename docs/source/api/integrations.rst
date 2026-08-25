Integrations
============

Model integrations are workspace packages rooted at
``integrations_v2/<model_name>/``. Each package owns the model implementation,
its canonical pipeline configs, and small bindings to reusable applications.

Layout
------

The required shape is::

   integrations_v2/<model_name>/
   ├── config.py
   ├── impl/                       # all model implementation
   │   ├── pipeline.py             # only when the model needs one
   │   └── transformer/
   ├── apps/
   │   └── <app_name>/
   │       └── adapter.py
   └── tests/

``config.py`` is the only Python module at the integration root and the single
source of model-specific defaults, pipeline configs, and hook factories. All
model implementation, integration-owned data, and vendored source stays under
``impl/``. Apart from ``apps/``, ``tests/``, ``README.md``, and
``pyproject.toml``, only an important integration-wide operational script may
sit beside ``config.py``.

Applications live under ``apps/<app_name>/`` and must not import a concrete
model package. When an app needs model behavior, it exposes a typed hook or
factory. The nested integration adapter imports those defaults and hooks only
from ``...config`` and passes them to the reusable application.

Application discovery
---------------------

An integration exposes a binding through its package manifest::

   [project.entry-points."flashdreams.applications_v2"]
   action2v-my-model = "my_model.apps.action2v.adapter:create_app"

The adapter should remain a bare construction boundary. Do not place UI,
input handling, launch orchestration, model selection, or checkpoint literals
in it.

Forbidden compatibility layers
------------------------------

Do not add integration-root ``runner.py``, ``launch.py``, ``runtime.py``,
``prepare.py``, or ``model_session.py`` shims. Move real model code into
``impl/`` and move complete demo behavior into the repository-level ``apps/``.

Tests
-----

Reusable application behavior is tested with the application. Integration
tests cover model config literals, adapter wiring, checkpoint transforms, and
model numerics. CPU-safe architecture tests verify that adapters stay minimal
and legacy runner modules do not return.
