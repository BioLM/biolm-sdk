.. _managing-datasets:

=================
Managing datasets
=================

*Working with biological data.*

Once your data outgrows a single file on one laptop, you want it stored somewhere
durable, versioned, and shared with your team. The BioLM platform provides exactly
that: **platform datasets**, backed by `MLflow <https://mlflow.org/>`_ runs and
artifacts hosted on ``biolm.ai``. You push files up, list what is there, inspect a
dataset's metadata and contents, and pull artifacts back down — all through the
``biolm dataset`` command group. This is the counterpart to :doc:`sequence-and-structure-data`,
which is about reading sequences and structures from local files; here the data lives
on the platform rather than on disk.

Under the hood, each dataset is a single MLflow run tagged as a dataset, and the
files you upload are stored as that run's artifacts. You never have to touch MLflow
directly — the ``biolm dataset`` commands wrap all of that — but knowing the mapping
explains the vocabulary you will see in ``show`` output (run IDs, tags, params, and
metrics all come straight from the backing run).

.. contents::
   :local:
   :depth: 1


Before you start
================

Dataset commands live behind an optional extra, so install it and authenticate first:

.. code-block:: bash

    pip install "biolm-sdk[mlflow]"
    biolm login

The ``[mlflow]`` extra pulls in the MLflow client the commands are built on; without
it every ``biolm dataset`` call stops with a clear "MLflow Not Available" message.
``biolm login`` establishes the platform credentials the commands reuse — see
:doc:`authentication` for the full login flow. All commands talk to the hosted
tracking server at ``https://mlflow.biolm.ai/`` by default; override it per command
with ``--mlflow-uri`` if you run your own MLflow.

Datasets are organized under MLflow *experiments*. Yours default to
``{username}/datasets``, so you never have to name an experiment for everyday use.
Pass ``--experiment`` on any command to read from or write to a different one.


Listing what you have
=====================

Start by seeing what already exists:

.. code-block:: bash

    biolm dataset list

This prints a table of your datasets — dataset ID, name, status, and artifact count —
drawn from the runs tagged as datasets in ``{username}/datasets``. For scripting or
archival, switch the shape with ``--format json`` (or ``csv``) and send it to a file
with ``-o``:

.. code-block:: bash

    biolm dataset list --format json -o my-datasets.json

To look inside one dataset, use its ID:

.. code-block:: bash

    biolm dataset show my-dataset-123

``show`` reports the full picture: the underlying MLflow run ID, tags, params,
metrics, and a listing of every artifact with its size. Like ``list``, it accepts
``--format json`` and ``-o`` when you want the metadata as structured output rather
than a formatted panel.


Uploading files
==============

Uploading attaches one or more files to a dataset, creating the dataset on the fly
if the ID does not exist yet:

.. code-block:: bash

    biolm dataset upload my-dataset-123 data.csv

The first argument is the dataset ID you choose; the second is the local file to
push. Upload a whole directory tree with ``--recursive``:

.. code-block:: bash

    biolm dataset upload my-dataset-123 ./training_data --recursive

Give a new dataset a human-readable label with ``--name`` — it is stored as the
MLflow run name and shown by ``list`` and ``show``:

.. code-block:: bash

    biolm dataset upload my-dataset-123 data.csv --name "Training set v1"

Uploading to an ID that already exists appends the new artifacts to that dataset
rather than replacing it, so you can accumulate files across several calls.


Downloading artifacts
====================

Pulling a dataset back down is the mirror image of upload:

.. code-block:: bash

    biolm dataset download my-dataset-123

With no output path, artifacts land in the current directory; pass one to redirect
them:

.. code-block:: bash

    biolm dataset download my-dataset-123 ./downloads

By default every artifact in the dataset is fetched. When you only need one file
from a larger dataset, name it with ``--artifact-path`` — the same path you saw in
``biolm dataset show``:

.. code-block:: bash

    biolm dataset download my-dataset-123 ./downloads --artifact-path model.pkl

The command creates the destination directory if it does not exist and reports the
run ID it pulled from when it finishes. If the dataset ID cannot be found in the
experiment, download stops with a "Dataset Not Found" error rather than writing an
empty directory, so a typo fails fast instead of looking like an empty result.


A note on workspaces
===================

You may see a ``biolm workspace`` command group alongside ``biolm dataset``. It is
currently a stub and does not yet manage real remote storage, so reach for
``biolm dataset`` for anything you actually need to persist on the platform today.


Where to go next
===============

- :doc:`sequence-and-structure-data` — read sequences and structures from local
  FASTA, CSV, and PDB files (the local-data counterpart to this page).
- :doc:`protocol-workflows` — run platform protocols, whose inputs and outputs pair
  naturally with datasets.
- :doc:`authentication` — set up the ``biolm login`` credentials these commands rely on.
- :doc:`../cli/dataset` — the complete ``biolm dataset`` command reference.
