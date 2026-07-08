:orphan:

.. _sdk-index:

SDK Reference
=============

Reference for the ``biolm`` Python package (``pip install biolm-sdk``). For onboarding and client usage patterns, see :doc:`../intro/sdk-overview`.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Module
     - Primary symbols
     - Description
   * - :doc:`models`
     - ``biolm()``, ``Model``
     - High-level model inference and embeddings.
   * - :doc:`protocols`
     - ``Protocol``, ``ProtocolClient``, ``run_protocol()``
     - Protocol YAML validation and programmatic runs.
   * - :doc:`pipeline`
     - ``GenerativePipeline``, config types
     - Multi-stage protein design with DuckDB caching.
   * - :doc:`workspaces`
     - ``Workspace``
     - Workspace management (Python SDK upcoming).
   * - :doc:`volumes`
     - ``Volume``
     - Volume storage (Python SDK upcoming).
   * - :doc:`io`
     - ``load_fasta``, ``to_csv``, …
     - FASTA, CSV, JSON, and PDB file helpers.
   * - :doc:`finetune`
     - ``Finetune``
     - XGBoost and DSM finetuning workflows.
   * - :doc:`hub`
     - ``list_models_from_openapi``, hub config
     - biolm-hub gateway discovery and configuration.

Full module index: :doc:`../api-reference/biolm`.
