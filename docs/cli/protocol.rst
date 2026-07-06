``biolm protocol``
====================

Work with protocols (workflow YAML).

Usage
-----

List, show, run, validate, or initialize protocols.

.. code-block:: bash

   biolm protocol list
   biolm protocol show PROTOCOL_SOURCE
   biolm protocol run protocol.yaml
   biolm protocol validate protocol.yaml
   biolm protocol init --example EXAMPLE
   biolm protocol log results.json --outputs outputs.yaml --account ACCT --workspace WS --protocol PROTO

Command Reference
-----------------

.. click:: biolm.cli:protocol
   :prog: biolm protocol
   :show-nested:
