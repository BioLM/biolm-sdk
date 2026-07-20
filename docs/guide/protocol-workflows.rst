.. _protocol-workflows:

==========================================
Protocol Workflows
==========================================

*Orchestrating molecular design workflows*

Protocols let you describe a multi-step molecular design job — chained model
calls, gather/aggregation steps, filters, and structured outputs — as a single
declarative YAML file. You author and validate the YAML locally, then either
run it on your machine (Local Protocol Profile) or submit inputs to a
registered slug on the BioLM platform.

This section is split into focused guides:

.. contents::
   :local:
   :depth: 1


Choose local vs hosted
==================================

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Local execution
     - Hosted execution
   * - Where it runs
     - Your machine (``biolm[pipeline]``)
     - BioLM platform servers
   * - Entry points
     - ``biolm protocol run-local``, :meth:`Protocol.execute`
     - ``biolm protocol run SLUG``, :func:`run_protocol`
   * - Input format
     - ``--input key=value`` or Python ``dict``
     - JSON file keyed to protocol ``inputs``
   * - Feature set
     - ApiTask DAG, ``response_mapping``, ``outputs[]`` selection
     - Full protocol (gather, foreach, task-output expressions, …)
   * - Best for
     - Fast iteration, offline dev, CI smoke tests
     - Long runs, sharing by slug, full platform features

See :doc:`protocol-local-profile` for the exact local feature matrix.


Guides in this section
==================================

- :doc:`protocol-authoring` — scaffold, inspect, and validate YAML
- :doc:`protocol-local-execution` — ``run-local``, :meth:`Protocol.execute`, outputs selection
- :doc:`protocol-hosted-execution` — submit, wait, download, MLflow logging
- :doc:`protocol-local-profile` — Local Protocol Profile v1 reference


See also
==================================

- :doc:`workflows-overview` — protocols vs pipelines vs model one-offs
- :doc:`managing-datasets` — platform datasets for protocol inputs
- :doc:`../yaml/protocol-schema` — full protocol YAML field reference
- :doc:`../sdk/protocols` — Python API reference
- :doc:`../cli/protocol` — CLI command reference
