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
     - List accessible registered protocols, with search, pagination, and JSON output.
   * - :clicmd:`biolm protocol show <../protocol.html#biolm-protocol-show>`
     - Render a readable report from a YAML file or a protocol ID on the platform.
   * - :clicmd:`biolm protocol run <../protocol.html#biolm-protocol-run>`
     - Submit inputs to a registered protocol slug and optionally wait for results.
   * - :clicmd:`biolm protocol validate <../protocol.html#biolm-protocol-validate>`
     - Validate YAML syntax, JSON schema, task graph, and template expressions.
   * - :clicmd:`biolm protocol init <../protocol.html#biolm-protocol-init>`
     - Scaffold a blank or example-based protocol YAML file.
   * - :clicmd:`biolm protocol log <../protocol.html#biolm-protocol-log>`
     - Push protocol run results to MLflow using the outputs section of the protocol.

Use ``status`` and ``wait`` to monitor an existing run, ``cancel`` to request
cancellation, ``results`` to inspect its final JSON, and ``download`` to fetch
CSV or JSONL result archives.

.. _cli-index-workspaces:

Workspaces
----------

Manage immutable account/environment identities addressed as
``{account}/{environment}``.

.. list-table::
   :header-rows: 0
   :widths: 28 72
   :class: cli-command-table

   * - :clicmd:`biolm workspace list <../workspace.html#biolm-workspace-list>`
     - List accessible personal and organization workspace paths and IDs.
   * - :clicmd:`biolm workspace show <../workspace.html#biolm-workspace-show>`
     - Show the current workspace, or resolve an exact path when one is provided.
   * - :clicmd:`biolm workspace switch <../workspace.html#biolm-workspace-switch>`
     - Switch the active account and environment to an exact workspace path.
   * - :clicmd:`biolm workspace create <../workspace.html#biolm-workspace-create>`
     - Create an environment in the active account or the account selected by ``--account``.

There is no workspace delete command or platform delete endpoint.

Organizations and budgets
-------------------------

Organization commands manage accounts available to the authenticated user:
``biolm org list`` lists organizations, ``biolm org show`` inspects one,
``biolm org create`` creates one, and ``biolm org invite`` invites a member
with an organization role.

Budget commands operate on the active account context. ``biolm budget show``
displays budget and usage fields, while ``biolm budget set`` sets a
nonnegative account budget.

Monthly usage
-------------

``biolm usage show`` displays the effective account, selected month, usage
amounts, and charges grouped by model. It defaults to the current month and
personal account. Use ``--year`` and ``--month`` for another month,
``--environment-id`` to filter an environment, ``--account`` for an
organization or personal account, and ``--format json`` for the unmodified API
response. The platform exposes neither a billing-history list nor a stable
live-activity API through this command.

API keys
--------

``biolm apikey create`` creates an API key for the active account, or the
account named by ``--account``. The token is shown only once, so store it
immediately. ``biolm apikey delete`` revokes a key by its full token or its
eight-character prefix. There is no list command because the platform exposes
no API-key listing endpoint.

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
