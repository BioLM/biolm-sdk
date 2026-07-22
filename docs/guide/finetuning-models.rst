.. _finetuning-models:

================
Model finetuning
================

*Adapting BioLMs to your own data.*

The catalog models get you a long way, but sometimes the right model is one
trained on *your* sequences and *your* labels. The SDK exposes BioLM's managed
finetuning service through :class:`biolm.finetune.Finetune`, a small
Python-only client that launches training runs on BioLM infrastructure and
tracks them to completion. There is no CLI for finetuning — it lives entirely
in Python — and it covers two families of job: **XGBoost** models trained on
BioLM embeddings, and **DSM** language-model finetuning across three stages.

Finetuning is a gated feature. Every call authenticates exactly like the rest
of the SDK (``BIOLM_TOKEN`` or ``api_key=``; see :doc:`authentication`), and if
your account is not enabled for finetuning the service returns HTTP ``402`` and
the client raises a :class:`PermissionError` telling you to contact BioLM. A
missing or invalid token raises the same error for ``401``.

.. contents::
   :local:
   :depth: 1


How the client is shaped
=========================

:class:`~biolm.finetune.Finetune` is a collection of classmethods that each
return a plain ``dict`` — the JSON the service sends back, typically including a
``run_id`` you use to track progress. Every launch method is keyword-only, and
every training argument that accepts data (``train_data``, ``paired_data``,
``seed_sequences``, …) takes one of two shapes:

- a **list of row dicts** — ``[{"sequence": "MK...", "label": 1}, ...]``, or
- a **raw CSV string** — the full contents of a CSV file, sent inline.

Both are transmitted as JSON in the request body, so there is no file upload
step: build your table in memory (or read a CSV off disk into a string) and pass
it straight in. Because the whole client is Python, the natural workflow is to
prepare data with pandas or plain lists, launch a run, capture the returned
``run_id``, and poll it — all in the same script or notebook. An optional
``run_name`` labels the job for later, and ``environment_id`` pins it to a
specific compute environment when your account has more than one.


Training XGBoost on embeddings
==============================

:meth:`~biolm.finetune.Finetune.xgboost` trains a gradient-boosted model on
embeddings drawn from one or more catalog models. You supply labeled data, name
the embedding model(s) to featurize with, and pick a task type:

.. code-block:: python

    from biolm.finetune import Finetune

    run = Finetune.xgboost(
        train_data=[
            {"sequence": "MSILVTRPSPAGEEL", "label": 1},
            {"sequence": "MKTAYIAKQRQISFV", "label": 0},
            # ...
        ],
        embedding_models=["esm2-8m"],
        task_type="classification",
        target_column="label",
        text_column="sequence",
    )
    run_id = run["run_id"]

``target_column`` and ``text_column`` name the label and sequence fields in your
rows (defaulting to ``"label"`` and ``"sequence"``). ``task_type`` is
``"classification"`` or ``"regression"``. Beyond that you can tune the booster
directly — ``n_estimators``, ``max_depth``, ``learning_rate``, ``n_splits`` for
cross-validation, and ``seed`` — or turn on Ray Tune hyperparameter search with
``hyperopt=True`` and ``hyperopt_n_trials``. Optional ``test_data`` and
``validation_data`` follow the same row/CSV shape as ``train_data``. For
antibody workflows set ``antibody_mode=True`` and name the ``heavy_column`` and
``light_column``.


DSM finetuning: stage 1, stage 2, and RL
=========================================

DSM (diffusion sequence model) finetuning runs in stages that build on one
another: stage 1 adapts the base language model to your sequence distribution,
stage 2 specializes it on paired data, and the RL stage steers generation toward
a measurable objective. You can stop after any stage — a stage-1 checkpoint is
useful on its own — but stage 2 and RL expect the artifacts the earlier stages
produce.

**Stage 1** — :meth:`~biolm.finetune.Finetune.dsm_stage1` — is a single-chain
masked-LM finetune over your sequences:

.. code-block:: python

    run = Finetune.dsm_stage1(
        train_data=my_sequences_csv,      # list of dicts or CSV string
        sequence_col="sequence",
        max_steps=50000,
    )

**Stage 2** — :meth:`~biolm.finetune.Finetune.dsm_stage2` — continues from a
stage-1 checkpoint to learn paired, multichain structure (for example antibody
heavy/light pairs). Pass the checkpoint identifier the stage-1 run produced:

.. code-block:: python

    run = Finetune.dsm_stage2(
        stage1_checkpoint="<checkpoint-id>",
        paired_data=my_pairs,             # rows with heavy_col / light_col
        heavy_col="heavy",
        light_col="light",
    )

**RL** — :meth:`~biolm.finetune.Finetune.dsm_rl` — optimizes sequences against
an oracle using reinforcement learning, seeded with sequences you provide:

.. code-block:: python

    run = Finetune.dsm_rl(
        seed_sequences=["MSILVTRPSPAGEEL", "MKTAYIAKQRQISFV"],
        oracle_type="esmc",
        stability_objective="thermostability",
        algorithm="ppo",
        num_episodes=100,
    )

Each stage exposes its own training knobs — learning rate, batch size, gradient
accumulation, step counts, ``fp16``, and so on — with sensible defaults, so you
only override what you need.


Tracking a run to completion
=============================

Launches return immediately; training happens asynchronously on BioLM's side.
The simplest way to wait is :meth:`~biolm.finetune.Finetune.wait`, which polls
until the run reaches a terminal state and then returns its full detail:

.. code-block:: python

    result = Finetune.wait(run_id, poll_interval=15, timeout=3600)

The terminal states are held in :data:`~biolm.finetune.TERMINAL_STATUSES` —
``succeeded``, ``failed``, ``cancelled``, and ``error``. For finer control,
:meth:`~biolm.finetune.Finetune.progress` returns lightweight status,
:meth:`~biolm.finetune.Finetune.get_run` returns full detail and results,
:meth:`~biolm.finetune.Finetune.list_runs` paginates your history (filter by
``dag`` or ``status``), and :meth:`~biolm.finetune.Finetune.cancel` stops an
in-flight run.


Async variants
==============

Every launch and tracking method has an ``_async`` twin —
``xgboost_async``, ``dsm_stage1_async``, ``get_run_async``, and so on — that you
``await`` inside an event loop. The synchronous methods are thin wrappers that
run the coroutine for you, so use the plain names in scripts and notebooks and
the ``_async`` names when you are already in async code (see
:doc:`client-interfaces`).


Declarative builds (BioLM definition)
=====================================

Besides calling :class:`~biolm.finetune.Finetune` from Python, you can describe
an XGBoost-on-embeddings adaptation as a **recipe** YAML — a Dockerfile-like
blueprint — and compile it with ``biolm model build``. Build writes a locked
**package** under ``~/.biolm/models/<name>/<tag>/`` with a shouty ``BioLM``
manifest (YAML without a ``.yaml`` suffix). The recipe file is not modified;
think dbt source SQL versus compiled SQL in ``target/``.

Minimal recipe::

    schema_version: 1
    name: antibody-binder-clf
    from: esm2-8m
    layers:
      - type: embedding_head
        task: classification
        data: ./data/binders.csv

Build it::

    biolm model build ./models/antibody-binder-clf.yaml
    biolm model build ./models/antibody-binder-clf.yaml --tag v1

Or from Python::

    from biolm.models import build_model

    pkg = build_model("models/antibody-binder-clf.yaml", tag="v1")
    print(pkg.path)  # ~/.biolm/models/antibody-binder-clf/v1

``data`` must be a local CSV path (resolved relative to the recipe). v0 supports
exactly one ``embedding_head`` layer and maps it to
:meth:`~biolm.finetune.Finetune.xgboost`. The package records lineage
(``run_id``, resolved data path) and serving ``actions`` (``encode`` +
``predict``) for later Hub or MLflow consumers.


Where to go next
================

- :doc:`what-are-biolms` — the models and actions you can featurize and finetune from.
- :doc:`choosing-models` — pick the right embedding model for an XGBoost run.
- :doc:`sequence-and-structure-data` — prepare labeled sequences for training.
- :doc:`authentication` — set ``BIOLM_TOKEN`` and confirm finetuning is enabled.
- :doc:`client-interfaces` — sync vs. async, and when to reach for the ``_async`` methods.
- :doc:`../sdk/finetune` — the full :class:`~biolm.finetune.Finetune` API reference.
- :doc:`../cli/usage/models` — ``biolm model build`` and other model CLI commands.
