Models
======

Use the ``biolm model`` commands to list models, show details, run models,
generate examples, and **build** BioLM definition recipes into locked local
packages. See :doc:`../model` for the command reference.

Build a recipe (XGBoost ``embedding_head``) into ``~/.biolm/models/<name>/<tag>/BioLM``::

    biolm model build ./models/antibody-binder-clf.yaml
    biolm model build ./models/antibody-binder-clf.yaml --tag v1 --name my-clf

See :doc:`../../guide/finetuning-models` for the recipe format and how it relates
to :class:`~biolm.finetune.Finetune`.
