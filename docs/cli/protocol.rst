``biolmai protocol``
====================

Work with protocols (workflow YAML).

Usage
-----

List, show, run, validate, or initialize protocols.

.. code-block:: bash

   biolmai protocol list
   biolmai protocol show PROTOCOL_SOURCE
   biolmai protocol run protocol.yaml
   biolmai protocol validate protocol.yaml
   biolmai protocol init --example EXAMPLE
   biolmai protocol log results.json --outputs outputs.yaml --account ACCT --workspace WS --protocol PROTO

Command Reference
-----------------

.. click:: biolm.cli:protocol
   :prog: biolmai protocol
   :show-nested:
