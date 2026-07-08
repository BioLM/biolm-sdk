:orphan:

.. _sdk-index:

SDK Reference
=============

Reference for the ``biolm`` Python package (``pip install biolm-sdk``). For onboarding and client usage patterns, see :sdklink:`Python SDK overview <../../intro/sdk-overview.html>`.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Module
     - Primary symbols
     - Description
   * - :sdklink:`biolm.models <../models.html>`
     - ``biolm()``, ``Model``
     - High-level model inference and embeddings.
   * - :sdklink:`biolm.protocols <../protocols.html>`
     - ``Protocol``, ``ProtocolClient``, ``run_protocol()``
     - Protocol YAML validation and programmatic runs.
   * - :sdklink:`biolm.pipeline <../pipeline.html>`
     - ``GenerativePipeline``, config types
     - Multi-stage protein design with DuckDB caching.
   * - :sdklink:`biolm.workspaces <../workspaces.html>`
     - ``Workspace``
     - Workspace management (Python SDK upcoming).
   * - :sdklink:`biolm.volumes <../volumes.html>`
     - ``Volume``
     - Volume storage (Python SDK upcoming).
   * - :sdklink:`biolm.io <../io.html>`
     - ``load_fasta``, ``to_csv``, …
     - FASTA, CSV, JSON, and PDB file helpers.
   * - :sdklink:`biolm.finetune <../finetune.html>`
     - ``Finetune``
     - XGBoost and DSM finetuning workflows.
   * - :sdklink:`biolm.hub <../hub.html>`
     - ``list_models_from_openapi``, hub config
     - biolm-hub gateway discovery and configuration.

Full module index: :sdklink:`biolm package <../../api-reference/biolm.html>`.
