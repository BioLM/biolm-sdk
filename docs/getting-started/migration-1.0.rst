Migration to biolm 1.0
======================

biolm 1.0 renames the package from ``biolmai`` to ``biolm`` and adds ``biolm server``.

Install
-------

.. code-block:: bash

    pip install biolm
    pip install biolm[server]   # local Modal proxy
    pip install biolm[pipeline] # pipeline features

Package and CLI
---------------

+------------------+---------------------------+
| Before           | After                     |
+==================+===========================+
| ``pip install biolmai`` | ``pip install biolm`` |
| ``biolmai`` CLI  | ``biolm`` CLI (primary)   |
| ``import biolmai`` | ``import biolm``        |
+------------------+---------------------------+

The ``biolmai`` import and CLI still work but emit deprecation warnings.

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
common hybrid workflow—login on ``biolm.ai``, run models through ``biolm server``—set
only ``BIOLM_BASE_API_URL``.

Credentials path
----------------

Credentials remain at ``~/.biolmai/credentials`` for backward compatibility.

Local server
------------

See :doc:`../cli/server` and :doc:`../server/oss-integration`.

Terminal colors
---------------

The CLI auto-detects dark vs light terminals. If text is hard to read:

- ``export BIOLM_CLI_THEME=dark`` or ``light``
- ``biolm --color ...`` to force color on
- ``biolm --no-color ...`` or ``NO_COLOR=1`` for plain output
