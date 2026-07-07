``biolm model``
=================

Work with BioLM models.

Usage
-----

The model commands allow you to explore available models, view model details, run models, and generate SDK usage examples.

Examples
--------

List all available models:

.. code-block:: bash

   biolm model list

Filter models by capabilities:

.. code-block:: bash

   biolm model list --filter encoder=true
   biolm model list --sort model_name
   biolm model list --format json --output models.json

Show details for a specific model:

.. code-block:: bash

   biolm model show esm2-8m
   biolm model show esmfold --include-schemas

Run a model:

.. code-block:: bash

   biolm model run esm2-8m encode -i sequences.fasta -o embeddings.json
   biolm model run esmfold predict -i data.csv --params '{"temperature": 0.7}'
   biolm model run esm2-8m encode -i large.fasta --progress

Generate SDK usage examples:

.. code-block:: bash

   biolm model example
   biolm model example esm2-8m
   biolm model example esm2-8m --action encode
   biolm model example esm2-8m --output example.py

Command Reference
-----------------

.. click:: biolm.cli:model
   :prog: biolm model
   :show-nested:
