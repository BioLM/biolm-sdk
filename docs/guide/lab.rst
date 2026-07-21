.. _lab-lltp:

=========================
Lab-in-the-Loop (LLTP)
=========================

**Lab-in-the-Loop Protocol (LLTP)** is an open protocol for sending molecular
designs to wet-lab providers, checking job status on demand, and pulling
structured results back into your computational workflow. BioLM's LLTP stack
keeps the wire protocol, language SDKs, and vendor connectors in separate
repositories; **biolm-sdk** adds project compose (``lltp.yaml``), local run
state, CLI/Python orchestration, and a SeqFrame bridge so design tables can
round-trip with lab jobs.

What LLTP is
============

A typical loop:

1. You have candidate sequences (often as a :doc:`SeqFrame <seqframe>`).
2. You submit an order to a lab connector (expression, DNA synthesis, etc.).
3. You poll status when you care — LLTP does not block waiting for the bench.
4. When results are ready, you pull them and join them back onto your table.

biolm-sdk does **not** reimplement the protocol. It shepherds work: resolve
``lltp.yaml``, call installed connectors, record runs under ``.biolm/lltp/``,
and convert between SeqFrame and LLTP payloads.

Upstream projects
=================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Piece
     - Repository
   * - Protocol spec
     - `BioLM/lltp <https://github.com/BioLM/lltp>`_
   * - Python SDK
     - `BioLM/lltp-py <https://github.com/BioLM/lltp-py>`_
   * - JavaScript/TypeScript SDK
     - `BioLM/lltp-js <https://github.com/BioLM/lltp-js>`_
   * - Vendor connectors (Adaptyv, Twist, …)
     - `BioLM/lltp-connectors <https://github.com/BioLM/lltp-connectors>`_

Install
=======

Install the biolm-sdk lab extra, SeqFrame (for convert/merge), and at least one
connector. Connectors are not yet on PyPI; install from GitHub while iterating.

.. code-block:: console

    $ pip install "biolm-sdk[lltp]"
    $ pip install "biolm-sdk[seqframe]"
    $ pip install "adaptyv-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=adaptyv-lltp/src/py"
    $ pip install "twist-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=twist-lltp/src/py"

Auth is BYOK via environment variables and/or ``lltp.yaml`` (env wins). Field
reference: :doc:`../yaml/lltp-schema`.

Project config
==============

Create a starter config:

.. code-block:: console

    $ biolm lab init

That writes ``lltp.yaml`` in the current project. Discovery walks parents until
it finds the file. Named **experiments** are presets (connector +
``service_id`` + options) so submit calls stay short.

Example:

.. code-block:: yaml

    version: 1
    default_connector: adaptyv
    connectors:
      adaptyv:
        auth:
          token_env: ADAPTYV_API_TOKEN
        defaults:
          service_id: adaptyv-lltp.expression-v1
      twist:
        auth:
          token_env: TWIST_END_USER_TOKEN
        defaults:
          service_id: twist-lltp.dna-synthesis-v1
          wait_for_scoring: false
    experiments:
      express:
        connector: adaptyv
        service_id: adaptyv-lltp.expression-v1
      synthesize:
        connector: twist
        service_id: twist-lltp.dna-synthesis-v1
        wait_for_scoring: false

Each run is one JSON file under ``.biolm/lltp/<run_id>.json``. Full field docs:
:doc:`../yaml/lltp-schema`.

Python API
==========

.. code-block:: python

    from biolm import SeqFrame
    from biolm.lab import submit, status, confirm, results

    sf = SeqFrame.from_fasta("candidates.fasta")
    run = submit(sf, experiment="express")
    info = status(run.run_id)          # on-demand poll; no blocking wait
    # when quote approval is required:
    # confirm(run.run_id)
    sf_out = results(run.run_id)
    sf2 = sf.lab.merge(sf_out)         # join on id ↔ entity.entity_id

SeqFrame bridge
===============

``sf.lab`` only converts — it does not call vendor HTTP:

- ``sf.lab.to_lltp(service_id=...)`` → connector order payload
- ``SeqFrame.lab.from_lltp(dataset)`` → SeqFrame from ``to_lltp_result``
- ``sf.lab.merge(other)`` → join results on ``id``

Orchestration (submit / status / confirm / results) is ``biolm.lab`` or the
CLI below.

CLI
===

.. code-block:: console

    $ biolm lab submit candidates.parquet --experiment express
    $ biolm lab status <run_id>
    $ biolm lab confirm <run_id>
    $ biolm lab results <run_id> -o results.parquet
    $ biolm lab list

See also
========

- :doc:`seqframe` — sequence tables that feed and absorb lab results
- :doc:`../yaml/lltp-schema` — ``lltp.yaml`` field reference
- `LLTP spec <https://github.com/BioLM/lltp>`_ — wire protocol and schemas
