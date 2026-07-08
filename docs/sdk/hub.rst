``biolm.hub``
=============

When using `biolm-hub <https://github.com/BioLM/biolm-hub>`_ gateways, the SDK can
discover models from a hub's OpenAPI spec and persist the gateway URL in
``~/.biolm/config.yaml``.

CLI setup
---------

.. code-block:: bash

   # In biolm-hub repo: bh serve
   biolm hub set http://127.0.0.1:8000
   biolm model list

SDK modules
-----------

- ``biolm.hub.config`` — read/write ``hub_api_url`` in ``~/.biolm/config.yaml``
- ``biolm.hub.discovery`` — list models from hub OpenAPI
- ``biolm.hub.catalog`` — bundled catalog helpers for hub model metadata

Example (discover models from a running hub):

.. code-block:: python

   from biolm.hub.discovery import list_models_from_openapi

   models = list_models_from_openapi("http://127.0.0.1:8000/api/v1")
   for m in models:
       print(m["model_slug"], m["actions"])

API
---

.. autofunction:: biolm.hub.discovery.list_models_from_openapi

See also
--------

- :doc:`../cli/hub` — CLI hub commands
- :doc:`../api-reference/biolm.hub` — full hub module reference
