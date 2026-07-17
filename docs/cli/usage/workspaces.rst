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

Inspect monthly usage with ``biolm usage show``:

.. code-block:: bash

   biolm usage show
   biolm usage show --year 2026 --month 7
   biolm usage show --account acme --environment-id 31
   biolm usage show --format json

The command defaults to the current month and personal account. It reports the
effective account and environment filter returned by the server, total usage,
environment usage, and charges grouped by model. JSON output preserves the API
response. Live activity and billing-history listing are not exposed.

See :doc:`../workspace` for the generated workspace command reference.
