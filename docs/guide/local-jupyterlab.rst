.. _local-jupyterlab:

=========================
Local JupyterLab sessions
=========================

Notebooks are where most BioLM exploration happens — try a model, load a
FASTA, plot a distribution, adjust, repeat. Rather than hand-wire auth and
extensions yourself, ``biolm-sdk`` ships a launcher that starts a local
JupyterLab session with BioLM credentials and the jupyterlab-biolm extension
already in place. This guide covers that local launcher — starting,
stopping, and what you get inside Lab. It does not cover platform-hosted
sandboxes; ``--local`` is currently the only supported target.

Install
=======

The notebook extra pulls in JupyterLab and jupyterlab-biolm alongside the
SDK:

.. code-block:: bash

    pip install 'biolm-sdk[notebook]'

If you start a session without these installed, the CLI reports which package
is missing and prints an install command pinned to the active interpreter.

Starting a session
==================

Run the launcher in the foreground to keep JupyterLab attached to your
terminal:

.. code-block:: bash

    biolm notebook start --local

This resolves your BioLM credentials, sets a light CLI theme for Lab's
terminal (see :ref:`auth-and-env` below), and starts ``jupyter lab`` on port
8888 by default. Press Ctrl+C to shut it down — the launcher answers
JupyterLab's shutdown prompt for you, so one Ctrl+C is enough.

To free the terminal, start the session detached instead. ``-d`` (or
``--detach``) spawns JupyterLab in the background and prints the URL:

.. code-block:: bash

    biolm notebook start --local -d

``--local`` is required. It names the launch target; today it is the only
target the CLI supports. Platform-hosted sandboxes may follow later.

Useful options
--------------

.. code-block:: bash

    biolm notebook start --local --port 8890
    biolm notebook start --local --dir ./notebooks
    biolm notebook start --local --no-browser

- ``--port`` — bind to a port other than 8888.
- ``--dir`` — set JupyterLab's root directory.
- ``--no-browser`` — skip opening a browser tab (handy on a remote host or in
  CI).

Anything after a bare ``--`` is passed through to ``jupyter lab``:

.. code-block:: bash

    biolm notebook start --local --no-browser -- --ServerApp.token=''

Only one local session at a time
--------------------------------

The launcher refuses a second ``start`` while a session is already tracked:

.. code-block:: text

    A local notebook is already running (pid 41213, http://127.0.0.1:8888/lab).
    Stop it with: biolm notebook stop --local

Stop the running session before starting another.

Stopping a session
==================

.. code-block:: bash

    biolm notebook stop --local

This stops the process the CLI is tracking, whether it was started in the
foreground or detached. If no session is tracked, the command reports that
instead of guessing which process to kill.

Session metadata
================

Each launch writes its pid, port, and URL to
``~/.biolm/notebook-local.json``. ``biolm notebook stop --local`` reads this
file to find the process and clears it when the session ends. You do not need
to edit the file; it is plain JSON if you want to inspect it.

.. _auth-and-env:

Auth and environment
====================

The launcher normalizes ``BIOLM_TOKEN``, ``BIOLMAI_TOKEN``, and
``BIOLM_API_KEY`` before starting Lab: whichever one you have set fills in
the others, so the SDK and the jupyterlab-biolm extension agree on the same
token. If none are set, the extension's Settings panel can authenticate from
inside Lab, and credentials from ``biolm account login`` (or
``~/.biolm/credentials``) still work — see :doc:`authentication`.

The launcher also seeds ``BIOLM_CLI_THEME=light`` when you have not set it, so
``biolm`` output stays readable in JupyterLab's light terminal. A value you
set yourself is left alone.

What you get in Lab
===================

Once JupyterLab opens, jupyterlab-biolm adds a BioLM sidebar with:

- a model browser for the catalog,
- ready-to-run code snippets for common calls,
- a Settings panel for wiring a token in-Lab.

Inside a notebook, the SDK behaves as it does anywhere else — the same
``Model`` calls and file loaders from :doc:`sequence-and-structure-data` and
:doc:`running-inference` work once a token is available from the environment,
``~/.biolm/credentials``, or the sidebar Settings panel.

``jupyterlab-mlflow`` is not part of the notebook extra. Install it separately
if you want MLflow alongside BioLM notebooks.

Troubleshooting
===============

If ``biolm notebook start --local`` reports missing dependencies, install the
notebook extra for the interpreter you are actually running. The error includes
a copy-pasteable ``pip install`` command pinned to that interpreter — important
when ``jupyter`` on your ``PATH`` belongs to a different environment than the
one where you installed ``biolm-sdk[notebook]``.

Where to go next
================

- :doc:`authentication` — how ``BIOLM_TOKEN`` and ``biolm account login`` work.
- :doc:`sequence-and-structure-data` — load files into ``items`` in a notebook.
- :doc:`running-inference` — call a model from Python or the CLI.
- :doc:`../cli/notebook` — the complete ``biolm notebook`` command reference.
