Workspace command usage
=======================

Workspaces are account/environment identities addressed as
``{account}/{environment}``, such as ``acme/research``.

Use ``biolm workspace list`` to enumerate paths, ``show [PATH]`` to inspect the
current or named workspace, ``switch PATH`` to change active context, and
``create NAME [--account ACCOUNT]`` to create an environment. There is no
workspace delete command. Use ``biolm org`` for organization membership and
``biolm budget`` for the active account budget.

See :doc:`../workspace` for the generated workspace command reference.
