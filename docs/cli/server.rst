biolm server
============

Run a local proxy for Modal-deployed BioLM model APIs.

Installation
------------

.. code-block:: bash

    pip install biolm[server]

Start the server
----------------

.. code-block:: bash

    biolm server start

    # optional
    biolm server start --host 0.0.0.0 --port 8787 --auth token

On start, the server prints ``BIOLM_BASE_API_URL`` and ``BIOLM_BASE_DOMAIN`` exports
for pointing the CLI/SDK at the local proxy.

Point the CLI/SDK at the server
-------------------------------

.. code-block:: bash

    export BIOLM_BASE_API_URL=http://127.0.0.1:8787/api/v3
    export BIOLM_BASE_DOMAIN=http://127.0.0.1:8787
    biolm model list
    biolm model run esm2-8m encode -i sequences.json

Environment variables
---------------------

Server-side (Modal credentials — never share with SDK users):

- ``MODAL_TOKEN_ID``
- ``MODAL_TOKEN_SECRET``

Server configuration:

- ``BIOLM_SERVER_HOST`` (default ``127.0.0.1``)
- ``BIOLM_SERVER_PORT`` (default ``8787``)
- ``BIOLM_SERVER_AUTH`` — ``none`` or ``token``
- ``BIOLM_SERVER_TOKEN`` — required when auth is ``token``
- ``BIOLM_SERVER_MODELS`` — comma-separated slugs (config fallback)
- ``BIOLM_SERVER_REFRESH_SECONDS`` — registry rescan interval (default ``60``)

Auth
----

Default on localhost: **no auth**.

Token mode:

.. code-block:: bash

    export BIOLM_SERVER_AUTH=token
    export BIOLM_SERVER_TOKEN=your-secret
    biolm server start --auth token

Clients send ``Authorization: Token your-secret``.

Catalog vs deployed models
--------------------------

- ``biolm model list`` — models **deployed on this server** (registry ∩ catalog)
- ``biolm model catalog`` — full OSS catalog (all deployable models)

Platform-only routes (protocols, OAuth, workspaces) return **501** on the local server.
Use the hosted BioLM platform for those features.

OSS integration
---------------

See :doc:`../server/oss-integration` for the contract expected from the Modal OSS deployment repo.

Commands
--------

.. click:: biolm.cli:server
   :prog: biolm server
   :nested: full
