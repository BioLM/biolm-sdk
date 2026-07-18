:orphan:

.. _sdk-index:

SDK Reference
=============

Reference for the ``biolm`` Python package (``pip install biolm-sdk``). **Product APIs**
below are the recommended entry points. For HTTP clients, legacy helpers, and deprecated
imports, see :sdklink:`biolm.core <../core.html>`.

Product APIs
------------

Inference
~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`Model <../models.html>`
     - Encode, predict, generate, and lookup with a bound model interface.
   * - :sdklink:`get_example, list_models <../models.html>`
     - Model catalog browsing and copy-paste example generation (``biolm.models.examples``).

Protocols
~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`Protocol <../protocols.html>`
     - Load, validate, and inspect protocol YAML locally.
   * - :sdklink:`run_protocol() <../protocols.html>`
     - Submit a protocol run and block until results are ready.
   * - :sdklink:`ProtocolClient <../protocols.html>`
     - Submit, track, and download protocol runs programmatically.

Pipelines
~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`GenerativePipeline <../pipeline.html>`
     - Multi-stage protein design with DuckDB caching and resumability.
   * - :sdklink:`Pipeline configs <../pipeline.html>`
     - ``ScoringProtocolConfig``, ``DirectGenerationConfig``, saturation mutagenesis, and related types.

Platform
~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`PlatformClient, Workspace <../workspaces.html>`
     - Supported account/environment context, organization, environment, and budget APIs.
   * - :sdklink:`Volume <../volumes.html>`
     - Deprecated compatibility placeholder; Modal volumes are runtime-only.
   * - :sdklink:`Finetune <../finetune.html>`
     - XGBoost and DSM finetuning workflows.

Datasets
~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`DatasetClient, Dataset <../../api-reference/biolm.datasets.html>`
     - Local dataset inventory (``dataset.yaml``); optional push/pull backends.

SeqFrame
~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`SeqFrame <seqframe.html>`
     - Sequence-centric Parquet/DuckDB dataframe (optional ``biolm-sdk[seqframe]``); opens datasets labeled ``type: seqframe``.

Utilities
~~~~~~~~~

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`biolm.io <../io.html>`
     - FASTA, CSV, JSON, and PDB file helpers.
   * - :sdklink:`biolm.hub <../hub.html>`
     - biolm-hub gateway discovery and configuration.

Core client (advanced)
----------------------

Prefer product APIs above. Use :sdklink:`biolm.core <../core.html>` when you need direct
HTTP control, async clients, or legacy imports.

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`biolm() <../core.html>`
     - Legacy one-shot sync wrapper (re-exported from ``biolm``).
   * - :sdklink:`BioLMApi <../core.html>`
     - Sync HTTP client with schema access and manual batching.
   * - :sdklink:`BioLMApiClient <../core.html>`
     - Async HTTP client; ``await`` its methods.
   * - :sdklink:`biolm.core.legacy <../../api-reference/biolm.core.legacy.html>`
     - Deprecated; do not use in new code.

.. _plugins-optional:

Plugins (optional)
------------------

Optional third-party backends under ``biolm.plugins``. Install extras as needed
(``pip install biolm-sdk[mlflow]``). Plugins are not imported from the top-level
``biolm`` package.

.. list-table::
   :header-rows: 0
   :widths: 28 72

   * - :sdklink:`biolm.plugins.mlflow <../../api-reference/biolm.plugins.mlflow.html>`
     - Protocol result logging and optional dataset push/pull backend.
   * - :sdklink:`protocol log <../cli/protocol.html>`
     - CLI: log protocol run results to MLflow.
   * - :sdklink:`dataset <../cli/dataset.html>`
     - CLI: local datasets (create, list, add) and push/pull backends.

Full module index: :sdklink:`biolm package <../../api-reference/biolm.html>`.
