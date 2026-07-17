.. _protocol-workflows:

==========================================
Protocol Workflows
==========================================

*Orchestrating molecular design workflows*

Protocols let you describe a multi-step molecular design job — chained model
calls, gather/aggregation steps, filters, and structured outputs — as a single
declarative YAML file. The platform runs the whole graph for you, handling
concurrency, dependency ordering, and result collection, so you submit inputs
and collect a results table instead of wiring API calls together by hand.

This guide walks the full lifecycle: **author → validate → submit → wait →
results**. Authoring and validation happen locally against your YAML; submission
and monitoring happen on the platform through the CLI or Python SDK.

.. contents::
   :local:
   :depth: 1


What a protocol is
==================================

A protocol is a YAML document with a handful of top-level sections:

- ``inputs`` — named parameters the caller supplies at submission time.
- ``tasks`` — the steps of the graph (model calls, ``gather`` aggregations,
  transforms), wired together with ``depends_on`` and ``${{ ... }}`` template
  expressions.
- ``outputs`` — rules that shape the final merged results table (what to log,
  how to filter, order, and limit rows).

For the full field reference, see :doc:`../yaml/protocol-schema`.

Once authored, a protocol is registered on the platform and referenced by its
**slug** (for example ``my-protocol-slug``). You never submit the local YAML
file to run it — you validate the YAML locally, then submit inputs against the
registered slug.


Scaffold and author
==================================

Start from a blank template or a bundled example:

.. code-block:: bash

    # Blank template
    biolm protocol init my_protocol.yaml

    # From a bundled example (if examples/ ships with your install)
    biolm protocol init my_protocol.yaml --example antibody_design

    # See what examples ship with the SDK
    biolm protocol init --list-examples

Inspect any local file (or a registered protocol by ID) as a formatted report:

.. code-block:: bash

    biolm protocol show my_protocol.yaml


Validate
==================================

Validation is fully local and catches the errors that would otherwise fail a
run: YAML syntax, schema compliance, unknown task references, circular
dependencies, and malformed template expressions.

From the CLI:

.. code-block:: bash

    biolm protocol validate my_protocol.yaml

    # Machine-readable output for CI
    biolm protocol validate my_protocol.yaml --json

The command exits non-zero when validation fails, so it drops straight into a
pre-commit hook or CI step.

From Python, :meth:`biolm.Protocol.validate` returns a result object you can
inspect programmatically:

.. code-block:: python

    from biolm import Protocol

    result = Protocol.validate("my_protocol.yaml")

    if result.is_valid:
        print("OK:", result.statistics)  # task_count, input_count, ...
    else:
        for err in result.errors:
            # error_type is one of: syntax, schema, semantic
            print(f"[{err.error_type}] {err.path}: {err.message}")
        for warning in result.warnings:
            print("warning:", warning)

Each error carries a ``message``, a JSONPath-like ``path``, and an
``error_type``, so you can surface exactly where a protocol is malformed.


Submit, wait, and collect results
==================================

Once the platform has a registered protocol, run it from the CLI or Python.
Submission uses the platform **slug**, not the local YAML path. Input keys must
match the protocol's ``inputs`` section.

From the CLI
------------

Discover protocols, submit a JSON input object, and optionally wait for the
result:

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

The one-liner: :func:`biolm.run_protocol`
-------------------------------------------

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

Full control: :class:`~biolm.ProtocolClient` and :class:`~biolm.ProtocolRun`
----------------------------------------------------------------------------

When you need progress polling, cancellation, downloads, or a DataFrame, submit
with :class:`~biolm.ProtocolClient` and drive the returned
:class:`~biolm.ProtocolRun`:

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


ProtocolClient vs. run_protocol()
==================================

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


Troubleshooting
==================================

**Validation errors.** ``Protocol.validate()`` (and ``biolm protocol validate``)
report an ``error_type`` per error:

- ``syntax`` — the YAML itself failed to parse. Fix indentation/quoting.
- ``schema`` — a field is missing, misnamed, or the wrong type. Check it against
  :doc:`../yaml/protocol-schema`.
- ``semantic`` — a ``depends_on``/``from``/``foreach`` reference points at an
  unknown task or input, a **circular dependency** exists between tasks, or a
  ``${{ ... }}`` **template expression** is empty or has unbalanced braces.

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

- :doc:`../guide/workflows-overview` — how protocols fit alongside the local pipeline framework
- :doc:`managing-datasets` — upload and download platform datasets for protocol inputs
- :doc:`../yaml/protocol-schema` — the full protocol YAML field reference
- :doc:`../sdk/protocols` — Python API reference for protocol classes
- :doc:`../cli/protocol` — CLI command reference, including ``log`` and MLflow
