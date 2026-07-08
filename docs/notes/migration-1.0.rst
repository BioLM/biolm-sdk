Migration to biolm-sdk 1.0
==========================

biolm-sdk 1.0 is the new home for the BioLM Python client. Install with
``pip install biolm-sdk``; import and use ``biolm`` in code.

Install
-------

.. code-block:: bash

    pip install biolm-sdk
    pip install "biolm-sdk[pipeline]"  # pipeline features

Package and CLI
---------------

+-------------------------+-----------------------------+
| Before                  | After                       |
+=========================+=============================+
| ``pip install biolmai`` | ``pip install biolm-sdk``   |
| ``biolmai`` CLI         | ``biolm`` CLI (primary)     |
| ``import biolmai``      | ``import biolm``            |
+-------------------------+-----------------------------+

The ``biolmai`` import and CLI still work but emit deprecation warnings.

Repository
----------

The SDK lives at `biolm-sdk <https://github.com/BioLM/biolm-sdk>`_ on GitHub.
PyPI package name: ``biolm-sdk``.

Environment variables
---------------------

Canonical names use the ``BIOLM_`` prefix. Legacy ``BIOLMAI_*`` names still work:

+----------------------+----------------------+
| Canonical            | Legacy (deprecated)  |
+======================+======================+
| ``BIOLM_TOKEN``      | ``BIOLMAI_TOKEN``    |
| ``BIOLM_BASE_DOMAIN`` | ``BIOLMAI_BASE_DOMAIN`` |
| ``BIOLM_BASE_API_URL`` | ``BIOLMAI_BASE_API_URL`` |
| ``BIOLM_LOCAL``      | ``BIOLMAI_LOCAL``    |
| ``BIOLM_THREADS``    | ``BIOLMAI_THREADS``  |
+----------------------+----------------------+

``BIOLM_BASE_API_URL`` overrides **model inference and model list/catalog** only.
``BIOLM_BASE_DOMAIN`` controls the **platform** (OAuth, auth, hosted UI). For the
common hybrid workflow—login on ``biolm.ai``, run models through ``bh serve``—use
``biolm hub set`` or set ``BIOLM_BASE_API_URL``.

Credentials path
----------------

Canonical location: ``~/.biolm/credentials``. Legacy ``~/.biolmai/credentials`` is
still read (with a deprecation warning) if the canonical file does not exist.
New writes always go to ``~/.biolm/``.

Pipeline cache (default DuckDB home): ``~/.biolm/pipelines/`` (legacy:
``~/.biolmai/pipelines/`` with the same read-fallback behavior).

Hub gateway config: ``~/.biolm/config.yaml`` (unchanged).

biolm-hub
---------

See :doc:`../cli/hub`. Run ``bh serve`` in the biolm-hub repo, then ``biolm hub set``.

Terminal colors
---------------

The CLI auto-detects dark vs light terminals. If text is hard to read:

- ``export BIOLM_CLI_THEME=dark`` or ``light``
- ``biolm --color ...`` to force color on
- ``biolm --no-color ...`` or ``NO_COLOR=1`` for plain output
