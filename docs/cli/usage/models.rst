Models
======

Use the ``biolm model`` commands to list models, show details, run models,
generate examples, **build** BioLM definition recipes into locked local
packages, and **export** those packages to MLflow. See :doc:`../model` for the
command reference.

Build a recipe (XGBoost ``embedding_head``) into ``~/.biolm/models/<name>/<tag>/BioLM``::

    biolm model build ./models/antibody-binder-clf.yaml
    biolm model build ./models/antibody-binder-clf.yaml --tag v1 --name my-clf
    biolm model build ./recipe.yaml --bundle --artifact ./head.joblib

Export for MLflow / Modal (requires ``pip install mlflow-biolm``)::

    biolm model export-mlflow antibody-binder-clf:latest -o ./mlflow-model

See :doc:`../../guide/finetuning-models` for the recipe format and how it relates
to :class:`~biolm.finetune.Finetune`.
