Protocols
=========

Use ``biolm protocol`` to author and validate local YAML, execute supported
protocols locally, or run protocols already registered on the BioLM platform.

Validate local YAML before registration or local execution:

.. code-block:: bash

   biolm protocol validate design.yaml

Run a Local Protocol Profile–compatible YAML on your machine
(requires ``pip install "biolm-sdk[pipeline]"``):

.. code-block:: bash

   biolm protocol run-local design.yaml --input sequence=MKLLIV
   biolm protocol run-local design.yaml --input sequences='["MKLLIV"]' --json

Discover registered protocols, then submit JSON inputs against a protocol slug:

.. code-block:: bash

   biolm protocol list --search design
   biolm protocol run my-protocol-slug -i inputs.json
   biolm protocol run my-protocol-slug -i inputs.json --wait

The hosted submit command prints a run ID unless ``--wait`` is present. Reconnect to
that run from another shell:

.. code-block:: bash

   biolm protocol status ALY_123
   biolm protocol wait ALY_123
   biolm protocol results ALY_123 --output results.json
   biolm protocol download ALY_123 --output-dir results/
   biolm protocol cancel ALY_123

Use ``--format json`` for machine-readable list, run, status, wait, cancel, and
results output. Authentication accepts ``BIOLM_TOKEN`` or saved OAuth
credentials from ``biolm account login``.

See :doc:`../../sdk/protocols` and :doc:`../../guide/protocol-local-profile`
for local vs hosted details, and :doc:`../protocol` for the complete command reference.
