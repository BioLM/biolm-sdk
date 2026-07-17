``biolm hub``
=============

Connect the CLI and SDK to a running `biolm-hub <https://github.com/BioLM/biolm-hub>`_ gateway
(``bh serve`` locally or a deployed Modal gateway). Platform login and protocols still use
``biolm.ai``; only model inference is routed through the hub.

**Prerequisites.** Deploy and serve models from the `biolm-hub` repository:

.. code-block:: bash

    git clone https://github.com/BioLM/biolm-hub
    cd biolm-hub
    make install
    source .venv/bin/activate
    bh deploy esm2
    bh serve

**Typical workflow:**

.. code-block:: bash

    bh serve
    biolm hub set
    biolm account login
    biolm model run esm2-8m encode -i seq.json

By default ``biolm hub set`` points at ``http://127.0.0.1:8000`` and saves ``hub_api_url`` to
``~/.biolm/config.yaml``. ``BIOLM_BASE_API_URL`` in the environment takes precedence over the
saved config. Browse models in the browser at ``http://127.0.0.1:8000/catalog`` while
``bh serve`` is running.

.. click:: biolm.cli:hub
   :prog: biolm hub
   :nested: full
