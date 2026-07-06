biolm hub
=========

Connect the CLI and SDK to a running `biolm-hub <https://github.com/BioLM/biolm-hub>`_ gateway
(``bh serve`` locally or a deployed Modal gateway).

Prerequisites
-------------

Deploy and serve models from the `biolm-hub` repository:

.. code-block:: bash

    git clone https://github.com/BioLM/biolm-hub
    cd biolm-hub
    make install
    source .venv/bin/activate
    bh deploy esm2
    bh serve

Connect py-biolm
----------------

.. code-block:: bash

    biolm hub set
    biolm hub status
    biolm model list
    biolm model run esm2-8m encode -i sequences.json

By default ``biolm hub set`` points at ``http://127.0.0.1:8000`` and saves
``hub_api_url`` to ``~/.biolm/config.yaml``.

.. code-block:: bash

    biolm hub set http://127.0.0.1:8000
    biolm hub set https://your-workspace--biolm-gateway-web.modal.run

Disconnect
----------

.. code-block:: bash

    biolm hub unset

After unset, model inference uses the hosted platform (``biolm.ai``) unless
``BIOLM_BASE_API_URL`` is set in the environment.

Hybrid workflow
---------------

Platform login and protocols stay on ``biolm.ai``; only model inference uses the hub:

.. code-block:: bash

    bh serve
    biolm hub set
    biolm login
    biolm model run esm2-8m encode -i seq.json

Environment override
--------------------

``BIOLM_BASE_API_URL`` in the environment takes precedence over ``~/.biolm/config.yaml``.

.. code-block:: bash

    export BIOLM_BASE_API_URL=http://127.0.0.1:8000/api/v1

Catalog UI
----------

Browse models and run inference in the browser at ``http://127.0.0.1:8000/catalog``
while ``bh serve`` is running.
