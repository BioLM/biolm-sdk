.. _protocol-authoring:

==========================================
Authoring and Validating Protocols
==========================================

A protocol is a YAML document with a handful of top-level sections:

- ``inputs`` — named parameters the caller supplies at submission time.
- ``tasks`` — the steps of the graph (model calls, ``gather`` aggregations,
  transforms), wired together with ``depends_on`` and ``${{ ... }}`` template
  expressions.
- ``outputs`` — optional rules that shape the final results table (filter,
  order, limit rows for logging or downstream use).

For the full field reference, see :doc:`../yaml/protocol-schema`.

Once authored, a protocol can be registered on the platform and referenced by
its **slug** (for example ``my-protocol-slug``). Hosted submission uses that
slug with JSON inputs — not the local YAML file. Validate YAML locally before
registering or running locally.


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


Troubleshooting validation
==================================

``Protocol.validate()`` (and ``biolm protocol validate``) report an
``error_type`` per error:

- ``syntax`` — the YAML itself failed to parse. Fix indentation/quoting.
- ``schema`` — a field is missing, misnamed, or the wrong type. Check it against
  :doc:`../yaml/protocol-schema`.
- ``semantic`` — a ``depends_on``/``from``/``foreach`` reference points at an
  unknown task or input, a **circular dependency** exists between tasks, or a
  ``${{ ... }}`` **template expression** is empty or has unbalanced braces.


See also
==================================

- :doc:`protocol-workflows` — choose local vs hosted execution
- :doc:`protocol-local-execution` — run validated YAML locally
- :doc:`protocol-hosted-execution` — submit to the platform after registration
