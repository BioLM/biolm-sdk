.. _managing-datasets:

=================
Managing datasets
=================

*Local inventory for protocol results, finetuning inputs, and related artifacts.*

A **dataset** in the BioLM SDK is a self-describing directory: a ``dataset.yaml``
plus whatever files you put beside it (usually under ``data/``). Datasets are
discovered under ``~/.biolm/datasets`` and ``./.biolm/datasets`` (and optional
extra roots). They are addressable by id so other workflows can refer to them
later; typed openers (for example SeqFrame) are a follow-up.

Optional backends such as MLflow only extend ``push`` / ``pull``. Local create,
list, show, and add work with no extras installed.

.. contents::
   :local:
   :depth: 1


Layout and discovery
====================

.. code-block:: text

    ~/.biolm/datasets/finetuning-v1/
    ├── dataset.yaml
    └── data/
        └── train.csv

``biolm dataset create`` writes that layout for you. ``biolm dataset init PATH``
drops a ``dataset.yaml`` into an existing directory without moving files. Schema
details live in :doc:`../yaml/dataset-schema`.


Creating and adopting datasets
==============================

.. code-block:: bash

    biolm dataset create finetuning-v1 --tag finetune
    biolm dataset init ./training-data --id finetuning-v1 --tag finetune

Add files:

.. code-block:: bash

    biolm dataset add finetuning-v1 train.csv
    biolm dataset add finetuning-v1 ./more-data --recursive


Listing and inspecting
======================

.. code-block:: bash

    biolm dataset list
    biolm dataset list --type files --tag finetune
    biolm dataset show finetuning-v1
    biolm dataset show ./training-data


Python API
==========

.. code-block:: python

    from biolm.datasets import DatasetClient

    client = DatasetClient()
    ds = client.create("finetuning-v1", tags=["finetune"])
    ds.add("train.csv")
    for item in client.list(tag="finetune"):
        print(item.id, item.path)


Push and pull (optional backends)
=================================

Sync with a remote backend when you need shared storage. MLflow requires the
optional extra and platform login:

.. code-block:: bash

    pip install "biolm-sdk[mlflow]"
    biolm account login

    biolm dataset push finetuning-v1 --backend mlflow
    biolm dataset pull finetuning-v1 --backend mlflow
    biolm dataset pull finetuning-v1 --backend mlflow --path ./my-copy

``pull`` defaults to ``~/.biolm/datasets/<id>/``.


Workspaces, datasets, and volumes
=================================

- A **workspace** is account/environment context (``{account}/{environment}``).
- A **dataset** is a local (or pushed) bag of files with ``dataset.yaml``.
- A **runtime volume** is server-side storage for Jupyter/protocol runs — not a
  local SDK storage API.

Use ``biolm workspace`` for platform context and ``biolm dataset`` for local
artifact inventory (with optional remote sync).


Dataset next steps
==================

- :doc:`../yaml/dataset-schema` — ``dataset.yaml`` field reference.
- :doc:`sequence-and-structure-data` — reading sequences and structures from files.
- :doc:`protocol-workflows` — protocol inputs/outputs that pair with datasets.
- :doc:`../cli/dataset` — full ``biolm dataset`` command reference.
