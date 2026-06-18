``biolmai model``
=================

Work with BioLM models.

Usage
-----

The model commands allow you to explore available models, view model details, run models, and generate SDK usage examples.

Examples
--------

List all available models:

.. code-block:: bash

   biolmai model list

Filter models by capabilities:

.. code-block:: bash

   biolmai model list --filter encoder=true
   biolmai model list --sort model_name
   biolmai model list --format json --output models.json

Show details for a specific model:

.. code-block:: bash

   biolmai model show esm2-8m
   biolmai model show esmfold --include-schemas

Run a model:

.. code-block:: bash

   biolmai model run esm2-8m encode -i sequences.fasta -o embeddings.json
   biolmai model run esmfold predict -i data.csv --params '{"temperature": 0.7}'
   biolmai model run esm2-8m encode -i large.fasta --progress

Generate SDK usage examples:

.. code-block:: bash

   biolmai model example
   biolmai model example esm2-8m
   biolmai model example esm2-8m --action encode
   biolmai model example esm2-8m --output example.py

Command Reference
-----------------

.. click:: biolm.cli:model
   :prog: biolmai model
   :show-nested:
