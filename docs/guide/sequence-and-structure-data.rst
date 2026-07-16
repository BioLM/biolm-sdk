.. _sequence-and-structure-data:

===================================
Working with biological data
===================================

Real inputs rarely start as a Python string typed into a call. They live in
files — a FASTA of candidate sequences, a CSV exported from a spreadsheet, a
JSONL dump from an earlier run, a PDB downloaded from the RCSB. This guide is
about the step *before* inference: turning those files into the ``items`` a
model expects. Once you know *which* model and action you want (see
:doc:`running-inference`), the only remaining question is how to get your data
into the right shape — and the :mod:`biolm.io` helpers exist for exactly that.

From files to ``items``
=======================

Every model action takes ``items``: a value, a list, or a list of dicts. As
covered in :doc:`how-biolms-work`, a bare string is normalized into a dict under
the model's input field, but for anything loaded from disk you usually want the
dicts directly, so ids and metadata ride along with each sequence.

``biolm.io`` gives you one loader per format — ``load_fasta``, ``load_csv``,
``load_json``, ``load_pdb`` — and each returns a plain ``list[dict]`` you can
hand straight to a :class:`~biolm.models.Model`:

.. code-block:: python

    from biolm import Model
    from biolm.io import load_fasta

    items = load_fasta("candidates.fasta")
    model = Model("esm2-8m")
    results = model.encode(items=items)

Every loader accepts either a path (``str`` or ``pathlib.Path``) or an open
file-like object, so the same code works with files on disk, uploads, or
in-memory buffers. Each also has a matching writer — ``to_fasta``, ``to_csv``,
``to_json``, ``to_pdb`` — for sending results back out.

FASTA
=====

``load_fasta`` parses single- and multi-line sequences and returns one dict per
record, each with ``sequence``, ``id``, and a ``metadata`` dict. The id comes
from the header; pipe- or space-separated header fields land in ``metadata``
(a trailing description is stored under ``"description"``):

.. code-block:: python

    items = load_fasta("candidates.fasta")
    # {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'metadata': {}}

Records without a usable id are numbered ``sequence_1``, ``sequence_2``, and so
on, so you can always trace a result back to its input. FASTA is the natural
choice for plain batches of sequences.

CSV
===

``load_csv`` reads a header row and returns one dict per data row, with column
names as keys. Values stay as strings — no type inference — so a numeric column
comes back as ``"0.95"``, not ``0.95``. Pass ``sequence_key`` to assert that a
particular column exists and fail early if it does not:

.. code-block:: python

    items = load_csv("library.csv", sequence_key="sequence")
    # {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'score': '0.95'}

Because every column is preserved on the dict, CSV is convenient when each
sequence carries extra fields you want to keep alongside the API call.

JSON and JSONL
==============

``load_json`` accepts three shapes and always returns a list of dicts:

- a **single object** → a one-item list,
- a **JSON array** of objects → the list as-is,
- **JSONL** (one JSON object per line) → one item per line.

It also unwraps a request envelope: a top-level ``{"items": [...]}`` or
``{"query": [...]}`` returns the inner array, so a saved request body round-trips
cleanly. Passing ``"-"`` reads from standard input.

.. code-block:: python

    items = load_json("payload.json")     # object, array, or JSONL — all work

JSON is the format to reach for when items need arbitrary per-item fields
(nested ``params``, structured metadata) that FASTA and CSV cannot express.

PDB structures
==============

Structure-aware models — ESMFold's neighbours, inverse-folding models like
ProteinMPNN and AntiFold — take a ``pdb`` field rather than a sequence.
``load_pdb`` reads a PDB file and returns ``[{"pdb": "..."}]``. Multi-model
files (those with ``MODEL`` / ``ENDMDL`` records) split into one item per model,
which is exactly the list an inverse-folding call wants:

.. code-block:: python

    from biolm.io import load_pdb
    from biolm import Model

    structures = load_pdb("backbone.pdb")   # [{"pdb": "..."}, ...]
    model = Model("protein-mpnn")
    designs = model.generate(items=structures)

If you are building a design workflow rather than a single call, the pipeline
framework can read backbones straight from a file or an upstream folding stage —
see :doc:`structure-conditioned-generation`.

Saving results back to files
============================

The writers mirror the loaders and take the same target types — a path, a
file-like object, or ``"-"`` for stdout. Since a list of results is just a list
of dicts, exporting is one call:

.. code-block:: python

    from biolm.io import to_csv

    to_csv(results, "encoded.csv")          # fieldnames inferred from row 0

``to_fasta`` accepts a ``sequence_key`` when your sequence lives under a
non-default key; ``to_json`` writes a JSON array by default and JSONL when you
pass ``jsonl=True`` or use a ``.jsonl`` extension; ``to_pdb`` concatenates
multiple structures into one file. If instead you want the *client* to stream
results to disk as they arrive — better for large jobs than collecting
everything in memory — use ``output="disk"`` on the action itself, described in
:ref:`disk-output`.

How this connects to the CLI
============================

These helpers are the Python counterpart of the CLI's file handling. When you
run ``biolm model run -i data.fasta``, the command detects the format from the
extension and loads it the same way ``load_fasta`` does; ``-o results.csv``
writes it back like ``to_csv``. So a shell one-liner and the loader/writer pair
are two doors to the same conversion — reach for the CLI when your data already
sits in a file and you want results straight to disk, and for :mod:`biolm.io`
when you need to inspect or reshape items in Python first.

Where to go next
================

- :doc:`running-inference` — invoke a model once you have your ``items`` ready.
- :doc:`managing-datasets` — store and share files on the platform.
- :doc:`how-biolms-work` — how ``items`` and ``type`` are normalized per model.
- :doc:`structure-conditioned-generation` — feed PDB backbones into design pipelines.
- :doc:`client-interfaces` — sync vs. async clients and :ref:`disk-output`.
- :doc:`../sdk/io` — the full ``biolm.io`` reference, with round-trip examples.
