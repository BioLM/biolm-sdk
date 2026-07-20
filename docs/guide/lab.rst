.. _lab-lltp:

=========================
Lab-in-the-Loop (LLTP)
=========================

*Submit designs to lab providers, check status on demand, and pull results back
into a SeqFrame.*

Install the protocol helper and (separately) a connector:

.. code-block:: console

    $ pip install "biolm-sdk[lltp]"
    $ pip install "biolm-sdk[seqframe]"   # for SeqFrame convert / merge
    $ pip install "adaptyv-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=adaptyv-lltp/src/py"
    $ pip install "twist-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=twist-lltp/src/py"

Connectors are not yet published to PyPI; install from GitHub while iterating.
Auth is BYOK via environment variables and/or ``lltp.yaml`` (env wins).


Project config
==============

Create a starter config:

.. code-block:: console

    $ biolm lab init

Example ``lltp.yaml``:

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

Runs are stored as one JSON file each under ``.biolm/lltp/<run_id>.json``.


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


CLI
===

.. code-block:: console

    $ biolm lab submit candidates.parquet --experiment express
    $ biolm lab status <run_id>
    $ biolm lab confirm <run_id>
    $ biolm lab results <run_id> -o results.parquet
    $ biolm lab list

See also :doc:`seqframe` and the
`LLTP spec <https://github.com/lltprotocol/lltp>`_.
