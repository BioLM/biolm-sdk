BioLM Definition Schema Reference
=================================

A **recipe** is a Dockerfile-like YAML blueprint for an adapted model. Compile it
with ``biolm model build`` (or :func:`biolm.models.build_model`); build writes a
locked **package** under ``~/.biolm/models/<name>/<tag>/`` with a shouty
``BioLM`` manifest (YAML, no ``.yaml`` suffix). The recipe file is not modified.

v0 supports exactly one ``embedding_head`` layer, which maps to
:meth:`~biolm.finetune.Finetune.xgboost`.

See :doc:`../guide/finetuning-models` for the workflow and
:doc:`../cli/usage/models` for CLI flags.

Recipe (input)
--------------

Minimal example
~~~~~~~~~~~~~~~

.. code-block:: yaml

    schema_version: 1
    name: antibody-binder-clf
    from: esm2-8m
    layers:
      - type: embedding_head
        task: classification
        data: ./data/binders.csv

Recommended example
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    schema_version: 1
    name: antibody-binder-clf
    description: Classify binder vs non-binder from ESM2 embeddings
    from: esm2-8m
    layers:
      - type: embedding_head
        task: classification
        data: ./data/binders.csv
        embedding_models:
          - esm2-8m
        target_column: label
        text_column: sequence
    actions:
      encode:
        input: sequence
      predict:
        input: sequence
        task: classification

Top-level fields
~~~~~~~~~~~~~~~~

**Required**

- ``name`` — package name (slug); may be overridden with ``--name`` / ``name=``.
- ``from`` — base model slug used for embeddings (e.g. ``esm2-8m``).
- ``layers`` — list with **exactly one** layer in v0.

**Optional**

- ``schema_version`` — integer; default ``1``.
- ``description`` — human-readable context; copied into the package when set.
- ``actions`` — optional overrides merged into default serving actions. When
  present, must include both ``encode`` and ``predict`` keys.

Layer (``embedding_head``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Required**

- ``type`` — must be ``embedding_head``.
- ``data`` — local filesystem path to a training CSV. Relative paths resolve
  against the recipe file’s directory.

**Optional**

- ``task`` — ``classification`` (default) or ``regression``.
- ``embedding_models`` — list of model slugs; default is ``[from]``.
- ``target_column`` — label column name; default ``label``.
- ``text_column`` — sequence column name; default ``sequence``.

Training CSV
~~~~~~~~~~~~

Must exist at build time. Column names must match ``text_column`` /
``target_column``. Content is passed to
:meth:`~biolm.finetune.Finetune.xgboost` as ``train_data``.

Package (``BioLM`` manifest)
----------------------------

Build layout::

    ~/.biolm/models/<name>/<tag>/
    ├── BioLM
    └── artifacts/          # only with --bundle
        └── <basename>

Example locked manifest (shape; values vary by run):

.. code-block:: yaml

    schema_version: 1
    name: antibody-binder-clf
    tag: latest
    description: Classify binder vs non-binder from ESM2 embeddings
    from:
      slug: esm2-8m
      load: lazy
    layers:
      - type: embedding_head
        task: classification
        data:
          path: /abs/path/to/binders.csv
        embedding_models:
          - esm2-8m
        target_column: label
        text_column: sequence
        run_id: "<finetune-run-id>"
        artifact:
          load: preload
          uri: https://example.com/model.joblib
          # path: only when bundled
        # metrics: ...   # when present on the finetune result
    actions:
      encode:
        input: sequence
        schema: biolm.encode.v1
      predict:
        input: sequence
        task: classification
        schema: biolm.predict.v1
    built:
      at: "2026-07-21T18:00:00Z"
      status: locked
      recipe_path: /abs/path/to/antibody-binder-clf.yaml

Package fields
~~~~~~~~~~~~~~

- ``from`` — object with ``slug`` (base model) and ``load`` (``lazy``: resolve
  the base model at serve time).
- ``layers[0].data`` — object with absolute ``path`` to the training CSV used
  at build.
- ``layers[0].run_id`` — finetune run that produced the head.
- ``layers[0].artifact`` — head weights policy:

  - ``load`` — ``preload`` (default for the head).
  - ``uri`` — remote or local URI when known (from the finetune result or
    ``--artifact``).
  - ``path`` — absolute path under ``artifacts/`` when built with ``--bundle``.

- ``actions`` — serving contract for MLflow / Modal consumers (via
  ``mlflow-biolm``). Defaults always include ``encode`` and ``predict`` with
  schema refs ``biolm.encode.v1`` / ``biolm.predict.v1``. Recipe ``actions``
  merge on top of those defaults.
- ``built`` — provenance: UTC timestamp, ``status: locked``, absolute
  ``recipe_path``.

Bundle and export
-----------------

- ``biolm model build … --bundle`` downloads the head into ``artifacts/`` and
  sets ``artifact.path``. Requires a URI from the finetune result or an
  explicit ``--artifact`` path/URL.
- ``biolm model export-mlflow name:tag -o ./mlflow-model`` (requires
  ``mlflow-biolm``) turns a package into an MLflow model directory.

Python
------

.. code-block:: python

    from biolm.models import build_model, load_recipe, resolve_package

    recipe = load_recipe("models/antibody-binder-clf.yaml")
    pkg = build_model("models/antibody-binder-clf.yaml", tag="v1", bundle=True,
                      artifact="./head.joblib")
    print(pkg.path, pkg.manifest["actions"])
    print(resolve_package("antibody-binder-clf:v1"))

What is not in v0
-----------------

- Multiple layers, non-``embedding_head`` types, or LoRA / full fine-tune recipes.
- Dataset IDs or Hub URIs as ``data`` (local CSV path only).
- Editing the recipe in place; rebuild to refresh the package for a given tag.
