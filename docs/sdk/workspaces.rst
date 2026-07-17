``biolm.workspaces``
====================

A :class:`~biolm.platform.Workspace` is an immutable account and environment
identity. Its canonical path is ``{account}/{environment}``, for example
``acme/research``. Account and environment IDs remain the authoritative
identifiers; the path is the readable form used by the SDK and CLI.

Use :class:`~biolm.platform.PlatformClient` to manage platform context. The
``biolm.workspaces`` module re-exports ``PlatformClient`` and ``Workspace`` for
compatibility.

Workspace API
-------------

``PlatformClient`` provides:

- ``list_workspaces()`` — list personal and organization workspaces
- ``current_workspace()`` — inspect the active account/environment context
- ``get_workspace(path)`` — resolve an exact ``account/environment`` path
- ``switch_workspace(path)`` — change the active account and environment
- ``create_workspace(name, account=None)`` — create an environment in the
  current account or a named account

Creating a workspace creates an environment under an account. The platform has
no workspace delete endpoint.

Organizations, environments, and budgets
-----------------------------------------

The same client exposes the underlying platform resources:

- Organizations: ``list_organizations()``, ``get_organization()``,
  ``create_organization()``, and ``invite_to_organization()``
- Environments: ``list_environments()`` and ``create_environment()``
- Active-account budgets: ``get_budget()`` and ``set_budget()``
- API keys: ``create_api_key(account=None)`` and
  ``delete_api_key(token_or_prefix)``

``create_api_key()`` returns the one-time token secret and owns the key with the
active account, or the account named by ``account``. The secret is not stored by
the SDK. ``delete_api_key()`` revokes a key by full token or eight-character
prefix. The platform has no API-key listing endpoint.

Session-scoped usage
--------------------

Account context is session-scoped. Reuse one client for related operations, or
use it as a context manager:

.. code-block:: python

   from biolm import PlatformClient

   with PlatformClient() as platform:
       current = platform.current_workspace()
       workspaces = platform.list_workspaces()
       target = platform.get_workspace("acme/research")
       platform.switch_workspace(target)
       created = platform.create_workspace("experiments", account="acme")

The client handles OAuth/token credentials and persists the session cookies
needed for account-context switches. Run ``biolm login`` once when using OAuth;
application code does not need to copy or manage those cookies.

Platform API reference
----------------------

.. autoclass:: biolm.platform.Workspace
   :members:
   :undoc-members:

.. autoclass:: biolm.platform.PlatformClient
   :members: list_workspaces, current_workspace, get_workspace, switch_workspace, create_workspace, list_organizations, get_organization, create_organization, invite_to_organization, list_environments, create_environment, get_budget, set_budget, create_api_key, delete_api_key

Workspace documentation
-----------------------

- :doc:`../cli/workspace` — workspace CLI
- :doc:`../api-reference/biolm` — ``biolm.platform`` module reference
