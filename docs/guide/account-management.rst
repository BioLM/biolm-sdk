.. _account-management:

==================
Account management
==================

*Who you are, what you are connected to, and how you spend.*

Everything you do on the BioLM platform happens inside an **account context**: a
personal or organization account paired with an environment. This guide ties the
pieces together — identity, workspaces, organizations, budgets, usage, and API
keys — and shows the canonical ``biolm account`` commands alongside the
:class:`~biolm.platform.PlatformClient` methods that back them. For the initial
login flow, see :doc:`authentication`.

.. contents::
   :local:
   :depth: 1


Prerequisites
=============

Account commands require credentials. Set ``BIOLM_TOKEN`` or log in once:

.. code-block:: bash

    biolm account login

Short aliases such as ``biolm login`` and ``biolm logout`` still work, but this
guide uses the canonical ``biolm account`` forms.


Who am I connected to?
======================

Two top-level commands answer the everyday questions. ``biolm status`` reports
*what you are connected to* — configured endpoints, authentication state, and the
active account and workspace when available. It stays useful even when you are
logged out or a service is unreachable, degrading to an ``unavailable`` marker
rather than failing.

.. code-block:: bash

    biolm status

``biolm whoami`` answers *who is authenticated* — your username, email, and the
active personal or organization account and environment.

.. code-block:: bash

    biolm whoami
    biolm whoami --format json

From Python, the same identity is available through
:meth:`~biolm.platform.PlatformClient.get_current_user`:

.. code-block:: python

    from biolm import PlatformClient

    with PlatformClient() as platform:
        user = platform.get_current_user()
        context = platform.current_workspace()


Workspaces and account context
==============================

A **workspace** is an immutable account and environment identity, addressed as
``{account}/{environment}`` (for example ``acme/research``). Workspace commands
list, inspect, switch, and create these identities:

.. code-block:: bash

    biolm workspace list
    biolm workspace show
    biolm workspace show acme/research
    biolm workspace switch acme/research
    biolm workspace create experiments --account acme

Switching a workspace changes the active account context that later commands and
SDK calls use. There is no workspace delete command; the platform exposes no
delete endpoint. See :doc:`../cli/usage/workspaces` for the command reference and
:doc:`../sdk/workspaces` for the SDK equivalents.


Organizations
=============

Organization commands manage the accounts available to you. They accept an exact
organization **name or slug** — you never need a numeric ID:

.. code-block:: bash

    biolm account org list
    biolm account org show "Acme Labs"
    biolm account org show acme
    biolm account org invite acme person@example.com --role member

Create organizations in the BioLM console; the CLI does not expose organization
creation. When a name and a slug match different organizations, the SDK reports
an ambiguous identifier instead of guessing.


Budgets
=======

Budget commands operate on the active account context. Show the current budget,
or set a nonnegative amount:

.. code-block:: bash

    biolm account budget
    biolm account budget set 250

The SDK exposes :meth:`~biolm.platform.PlatformClient.get_budget` and
:meth:`~biolm.platform.PlatformClient.set_budget`.


Reviewing monthly usage
=======================

``biolm account usage`` reports the effective account, selected month, usage
amounts, and charges grouped by model. It defaults to the current month and your
personal account:

.. code-block:: bash

    biolm account usage
    biolm account usage --year 2026 --month 7
    biolm account usage --account acme --environment-id 31
    biolm account usage --format json

Pass ``--account`` to scope an organization or personal account within the same
session. JSON output preserves the raw API response. The platform exposes
neither a billing-history list nor a live-activity API through this command.


Managing API keys
=================

API keys authenticate application code without an interactive login. Create a key
for the active account or a named account, and revoke it by full token or
eight-character prefix:

.. code-block:: bash

    biolm account api-key create
    biolm account api-key create --account acme
    biolm account api-key delete <full-token-or-8-char-prefix> --yes

The token is shown **only once** on creation, so store it immediately; the SDK
never persists it. There is no ``list`` command because the platform exposes no
API-key listing endpoint. Set a created token as ``BIOLM_TOKEN`` to use it for
subsequent requests — see :doc:`authentication`.


Using the SDK directly
======================

Every command above maps to a :class:`~biolm.platform.PlatformClient` method.
Account context is session-scoped, so reuse one client for related operations:

.. code-block:: python

    from biolm import PlatformClient

    with PlatformClient() as platform:
        platform.switch_workspace("acme/research")
        budget = platform.get_budget()
        usage = platform.get_usage_summary(year=2026, month=7, account="acme")
        key = platform.create_api_key(account="acme")  # token shown once

See :doc:`../sdk/workspaces` for the full platform API reference.


Related documentation
=====================

- :doc:`authentication` — logging in and credential storage
- :doc:`../cli/account` — generated ``biolm account`` command reference
- :doc:`../cli/usage/workspaces` — workspace command usage
- :doc:`../sdk/workspaces` — ``PlatformClient`` and ``Workspace`` reference
