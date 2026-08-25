Adding a model integration
==========================

Create a model package under ``integrations_v2/<model_name>/``. Do not create
a parallel package under the legacy ``integrations/`` tree.

1. Add the model implementation
-------------------------------

Keep concrete transformers, encoders, decoders, schedulers, and any custom
pipeline implementation under ``integrations_v2/<model_name>/impl/``. Reuse ``flashdreams.core`` and
``flashdreams.infra`` contracts; those framework packages must never import
the integration.

2. Define canonical configs
---------------------------

Create ``integrations_v2/<model_name>/config.py``. Define named pipeline
config literals and a dictionary keyed by each config's ``name``. Application
defaults and model-owned hooks also belong here, so adapters have exactly one
source for model configuration.

3. Reuse or create an application
---------------------------------

Complete demos belong under ``apps/<app_name>/``. The application must be
model-agnostic: represent model construction, checkpoint selection, input
loading, and backend creation as typed hooks or factories supplied by an
integration.

4. Add the model adapter
------------------------

Create ``integrations_v2/<model_name>/apps/<app_name>/adapter.py``. It should
only construct the reusable app using configuration imported from
``...config``::

   from shared_app import SharedApplication

   from ...config import APPLICATION_DEFAULTS, APPLICATION_HOOKS


   def create_app():
       return SharedApplication(
           defaults=APPLICATION_DEFAULTS,
           hooks=APPLICATION_HOOKS,
       )

Register it in the integration's ``pyproject.toml``::

   [project.entry-points."flashdreams.applications_v2"]
   action2v-my-model = "my_model.apps.action2v.adapter:create_app"

5. Verify the boundary
----------------------

Add CPU-safe tests that assert:

* every integration has a root ``config.py``;
* ``config.py`` is the only Python module at the integration root;
* model implementation lives under ``impl/``;
* the adapter imports model configuration only from ``...config``;
* complete application implementations live under ``apps/``;
* no integration-root runner or launch shim exists;
* the manifest entry point targets the nested adapter.

Run config imports and CPU tests before any GPU generation. Model rollout,
checkpoint parity, and quality tests retain their normal GPU or manual marker.
