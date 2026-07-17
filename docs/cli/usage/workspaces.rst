Workspace command usage
=======================

Workspaces are account/environment identities addressed as
``{account}/{environment}``, such as ``acme/research``.

Use ``biolm workspace list`` to enumerate paths, ``show [PATH]`` to inspect the
current or named workspace, ``switch PATH`` to change active context, and
``create NAME [--account ACCOUNT]`` to create an environment. There is no
workspace delete command. Use ``biolm org`` for organization membership and
``biolm budget`` for the active account budget.

Manage API keys with ``biolm apikey``:

.. code-block:: bash

   biolm apikey create
   biolm apikey create --account acme
   biolm apikey delete <full-token-or-8-char-prefix> --yes

``create`` prints the token once; store it immediately. Pass ``--account`` to
own the key with an organization or your personal account. ``delete`` revokes a
key by full token or eight-character prefix. The platform has no API-key
listing endpoint, so there is no ``apikey list`` command.

See :doc:`../workspace` for the generated workspace command reference.
