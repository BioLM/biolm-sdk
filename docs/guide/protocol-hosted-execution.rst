.. _protocol-hosted-execution:

==========================================
Running Protocols on the Platform
==========================================

Once a protocol is registered on BioLM, you submit **inputs** against its
**slug** — not the local YAML path. Input keys must match the protocol's
``inputs`` section. Author and validate YAML locally first; see
:doc:`protocol-authoring`.


Discover and submit (CLI)
==================================

.. code-block:: bash

   biolm protocol list --search design
   biolm protocol run my-protocol-slug -i inputs.json
   biolm protocol run my-protocol-slug -i inputs.json --wait

Without ``--wait``, the command returns a run ID immediately. Use that ID to
monitor or manage the run:

.. code-block:: bash

   biolm protocol status ALY_123
   biolm protocol wait ALY_123
   biolm protocol results ALY_123 --output results.json
   biolm protocol download ALY_123 --output-dir results/
   biolm protocol cancel ALY_123

Use ``--format json`` for machine-readable list, run, status, wait, cancel, and
results output.


One-liner: :func:`biolm.run_protocol`
======================================

:func:`~biolm.run_protocol` submits a run, blocks until it finishes, and
returns the results dict:

.. code-block:: python

    from biolm import run_protocol

    results = run_protocol(
        "my-protocol-slug",               # platform slug
        inputs={                          # keys match the protocol's `inputs`
            "sequences": ["MKTAYIAKQRQGHQAMAEIKQ"],
        },
        run_name="protocol-demo",         # optional label
        timeout=3600.0,                   # seconds to wait
        show_progress=True,
    )

    print(results)

Authentication comes from ``BIOLM_TOKEN``, an explicit ``api_key=`` argument,
or saved OAuth credentials from ``biolm account login``. See
:doc:`authentication`.


Full control: :class:`~biolm.ProtocolClient`
==============================================

When you need progress polling, cancellation, downloads, or a DataFrame, submit
with :class:`~biolm.protocols.ProtocolClient` and drive the returned
:class:`~biolm.protocols.ProtocolRun`:

.. code-block:: python

    from biolm import ProtocolClient

    client = ProtocolClient()  # BIOLM_TOKEN or saved OAuth credentials

    run = client.submit(
        "my-protocol-slug",
        inputs={"sequences": ["MKTAYIAKQRQGHQAMAEIKQ"]},
        run_name="protocol-demo",
    )
    print("submitted:", run.run_id, run.status)

    # Stream status until terminal, or poll a snapshot yourself
    run.wait(timeout=3600.0, show_progress=True)
    snapshot = run.progress()   # dict: status, progress_pct, ...

    # Collect results
    detail = run.results()      # full run detail dict
    df = run.to_dataframe()     # download CSV zip -> pandas DataFrame
    path = run.download(output_dir="results", file_type="csv")

    # Cancel a running job if needed
    # run.cancel()

To reconnect to a run started elsewhere, use its ID:

.. code-block:: python

    run = client.get_run("run-abc123")
    run.wait()

.. note::

   ``run.to_dataframe()`` requires ``pandas``, and ``run.wait()`` uses
   ``websockets`` for live telemetry. Install them if they are not already in
   your environment.


ProtocolClient vs. :func:`run_protocol`
========================================

Reach for :func:`~biolm.run_protocol` when you want a single blocking call and
only care about the final results dict — scripts, notebooks, and quick jobs.

Reach for :class:`~biolm.ProtocolClient` / :class:`~biolm.ProtocolRun` when you
need any of: submitting without blocking, polling ``progress()`` for a
dashboard, cancelling a run, downloading artifacts, converting results to a
DataFrame, or reconnecting to an existing ``run_id``. Under the hood
``run_protocol`` is exactly ``client.submit(...).wait(...)`` followed by
``run.results()``.


Logging results to MLflow
==================================

The ``biolm protocol log`` command pushes a run's results into MLflow using the
protocol's ``outputs`` configuration (params, metrics, tags, aggregates, and
artifacts):

.. code-block:: bash

    biolm protocol log results.jsonl --outputs my_protocol.yaml \
        --account acme --workspace lab --protocol my-protocol-slug

MLflow support is optional and ships behind an extra. If it is not installed the
command reports *MLflow Not Available*; install it with:

.. code-block:: bash

    pip install biolm-sdk[mlflow]

See :doc:`../cli/protocol` for the complete command reference and options.


Troubleshooting hosted runs
==================================

**Authentication.** A missing token raises ``ValueError`` on the first API call.
Set ``BIOLM_TOKEN`` (or run ``biolm account login``), or pass ``api_key=`` to
:class:`~biolm.ProtocolClient`. See :doc:`authentication`.

**Run failures.** A run that fails or is cancelled raises
:class:`~biolm.ProtocolRunError` from ``wait()``. An unknown slug or version
raises :class:`~biolm.ProtocolNotFoundError` (a subclass). Catch them to react:

.. code-block:: python

    from biolm import run_protocol, ProtocolRunError, ProtocolNotFoundError

    try:
        results = run_protocol("my-protocol-slug", inputs={...})
    except ProtocolNotFoundError:
        print("Check the protocol slug/version.")
    except ProtocolRunError as exc:
        print("Run failed:", exc)

**Timeouts.** If a run exceeds ``timeout`` seconds, ``wait()`` raises
``TimeoutError`` after refreshing status — the run keeps going on the platform.
Reconnect later with ``client.get_run(run_id)`` and call ``wait()`` again, or
raise the ``timeout`` for long jobs.


See also
==================================

- :doc:`protocol-workflows` — local vs hosted overview
- :doc:`protocol-local-execution` — run YAML locally instead
- :doc:`../sdk/protocols` — Python API reference
