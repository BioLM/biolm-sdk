:orphan:

.. _cli-index:

CLI
===

Command-line interface for **biolm-sdk** (``biolm``). Authenticate once, then manage workspaces,
run models, execute protocols, and work with MLflow-backed datasets from the terminal.

.. _cli-index-authentication:

Authentication
--------------

Sign in, check status, and manage saved credentials.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm login <../login.html#biolm-login>`
     - Log in with OAuth (PKCE); credentials are saved to ``~/.biolm/credentials`` for later commands.
   * - :clicmd:`biolm logout <../logout.html#biolm-logout>`
     - Remove saved credentials so the CLI no longer has platform access until you log in again.
   * - :clicmd:`biolm status <../status.html#biolm-status>`
     - Show auth state, environment variables, model API URL, hub mode, and credential path.
   * - :clicmd:`biolm version <../../reference/cli.html#biolm-version>`
     - Print the installed ``biolm`` package version.

.. _cli-index-models:

Models
------

Discover models and run inference from the terminal (hosted API or a connected biolm-hub).

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm model list <../model.html#biolm-model-list>`
     - Browse the model catalog with filters, sorting, and table/JSON/YAML/CSV export.
   * - :clicmd:`biolm model catalog <../model.html#biolm-model-catalog>`
     - List the full open-source deployable catalog (or hub OpenAPI routes when connected).
   * - :clicmd:`biolm model show <../model.html#biolm-model-show>`
     - Inspect a model's metadata, actions, and optional JSON schemas for each action.
   * - :clicmd:`biolm model run <../model.html#biolm-model-run>`
     - Run encode, predict, generate, or lookup on FASTA, CSV, PDB, or JSON input.
   * - :clicmd:`biolm model example <../model.html#biolm-model-example>`
     - Generate copy-pasteable Python SDK examples for a model and action.

.. _cli-index-protocols:

Protocols
---------

Define, validate, and execute multi-step workflows described in YAML.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm protocol list <../protocol.html#biolm-protocol-list>`
     - List platform protocols (listing UI coming soon; use ``show``/``validate`` for local YAML).
   * - :clicmd:`biolm protocol show <../protocol.html#biolm-protocol-show>`
     - Render a readable report from a YAML file or a protocol ID on the platform.
   * - :clicmd:`biolm protocol run <../protocol.html#biolm-protocol-run>`
     - Execute a protocol YAML file locally (requires ``biolm[pipeline]``; use ``--input key=value``, ``--json``, ``--output-dir``).
   * - :clicmd:`biolm protocol validate <../protocol.html#biolm-protocol-validate>`
     - Validate YAML syntax, JSON schema, task graph, and template expressions.
   * - :clicmd:`biolm protocol init <../protocol.html#biolm-protocol-init>`
     - Scaffold a blank or example-based protocol YAML file.
   * - :clicmd:`biolm protocol log <../protocol.html#biolm-protocol-log>`
     - Push protocol run results to MLflow using the outputs section of the protocol.

.. _cli-index-workspaces:

Workspaces
----------

Create and manage BioLM workspaces that scope projects and protocol runs.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm workspace list <../workspace.html#biolm-workspace-list>`
     - List workspaces you can access with names, IDs, and basic metadata.
   * - :clicmd:`biolm workspace show <../workspace.html#biolm-workspace-show>`
     - Show details for a workspace by ID, or the current workspace when omitted.
   * - :clicmd:`biolm workspace create <../workspace.html#biolm-workspace-create>`
     - Create a new workspace for organizing projects and protocol runs.
   * - :clicmd:`biolm workspace delete <../workspace.html#biolm-workspace-delete>`
     - Permanently delete a workspace and its associated resources.

.. _cli-index-datasets:

Datasets
--------

Upload, download, and inspect MLflow-backed datasets on the platform.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm dataset list <../dataset.html#biolm-dataset-list>`
     - List datasets in your MLflow experiment with optional JSON or CSV export.
   * - :clicmd:`biolm dataset show <../dataset.html#biolm-dataset-show>`
     - Show metadata, tags, metrics, and artifact listings for a dataset by ID.
   * - :clicmd:`biolm dataset upload <../dataset.html#biolm-dataset-upload>`
     - Upload a file or directory to a dataset (creates the dataset run if needed).
   * - :clicmd:`biolm dataset download <../dataset.html#biolm-dataset-download>`
     - Download all artifacts, or a single artifact path, to a local directory.

.. _cli-index-hub:

Hub
---

Route model inference through a local or self-hosted `biolm-hub <https://github.com/BioLM/biolm-hub>`_ gateway.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm hub set <../hub.html#biolm-hub-set>`
     - Save a gateway URL so ``biolm model`` uses local or self-hosted inference.
   * - :clicmd:`biolm hub status <../hub.html#biolm-hub-status>`
     - Show saved hub config, active model API URL, and live gateway health.
   * - :clicmd:`biolm hub unset <../hub.html#biolm-hub-unset>`
     - Clear saved hub settings and revert model calls to the hosted biolm.ai API.
