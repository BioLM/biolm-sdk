.. highlight:: shell

==============
Authentication
==============

To authenticate API requests made by the Python client,
two methods can be used:

1. set an environment-variable, which is permanent, meaning
   re-authentication will not be necessary
2. or log in using the CLI, which will save an access and refresh
   token that will expire after a period of inactivity.


Environment variable authentication
-----------------------------------

Obtain an API token from your BioLM User page,
then set the environment variable :code:`BIOLM_TOKEN`.
The legacy name :code:`BIOLMAI_TOKEN` still works but emits a deprecation warning.

.. note::

   Ensure you replace the example API token with your own.

.. code-block:: shell

    export BIOLM_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b

For Bash
^^^^^^^^

.. code-block:: shell

    echo "export BIOLM_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.bash_profile && source ~/.bash_profile

For Zsh
^^^^^^^

.. code-block:: shell

    echo "export BIOLM_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.zshrc && source ~/.zshrc

For Python
^^^^^^^^^^

.. code-block:: python

    import os
    os.environ['BIOLM_TOKEN'] = '9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b'

Alternatively, with :code:`biolm-sdk` installed, in your Terminal run :code:`biolm login`.

CLI authentication
------------------

.. code-block:: bash

    biolm login

The command will:

- Check for existing valid credentials
- If credentials are missing or invalid, open a browser for OAuth authorization
- Save credentials to :code:`~/.biolm/credentials`

.. code-block:: bash

    Login succeeded! Credentials saved to ~/.biolm/credentials

.. note::

   If you have credentials from an older install at ``~/.biolmai/credentials``, they
   are still read with a deprecation warning. New logins write to ``~/.biolm/credentials``.

.. code-block:: bash

    biolm login

If already logged in:

.. code-block:: text

    Credentials location: ~/.biolm/credentials
    Run `biolm status` to view your authentication status.

Custom OAuth client ID:

.. code-block:: bash

    biolm login --client-id YOUR_CLIENT_ID

Or set the ``BIOLM_OAUTH_CLIENT_ID`` environment variable (legacy: ``BIOLMAI_OAUTH_CLIENT_ID``).

Custom OAuth scopes (supported: read, write, introspection):

.. code-block:: bash

    biolm login --scope "read write"

Examples
--------

Login with default settings:

.. code-block:: bash

    biolm login

Login with custom client ID:

.. code-block:: bash

    biolm login --client-id abc123xyz

Login with custom scope:

.. code-block:: bash

    biolm login --scope "read write"

Credentials file format
-----------------------

Credentials are saved to :code:`~/.biolm/credentials` in JSON format with:

- ``access`` — short-lived access token
- ``refresh`` — refresh token for renewing access
- OAuth metadata (``token_url``, ``client_id``, etc.) when using ``biolm login``

See :doc:`migration-1.0` for the full list of renamed environment variables.
