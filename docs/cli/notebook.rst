``biolm notebook``
==================

Start and stop a **local** JupyterLab session with BioLM credentials and the
`jupyterlab-biolm <https://github.com/BioLM/jupyterlab-biolm>`_ extension
wired for the SDK. Platform-hosted sandboxes are not part of this command yet —
``--local`` is currently required on ``start`` and ``stop``.

**Install** the notebook extra (JupyterLab + jupyterlab-biolm):

.. code-block:: bash

    pip install 'biolm-sdk[notebook]'

**Typical workflow:**

.. code-block:: bash

    biolm account login                 # optional if BIOLM_TOKEN is already set
    biolm notebook start --local        # foreground Lab on port 8888
    # …or…
    biolm notebook start --local -d     # detach; prints URL
    biolm notebook stop --local

Session metadata (pid, port, URL) is stored in ``~/.biolm/notebook-local.json``
so ``stop`` can find the process. Only one tracked local session is allowed at
a time.

For a walkthrough of auth, Lab features, and troubleshooting, see
:doc:`../guide/local-jupyterlab`.

.. click:: biolm.cli:notebook
   :prog: biolm notebook
   :nested: full
