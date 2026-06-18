``biolmai login``
=================

Login to BioLM using OAuth 2.0 with PKCE.

Usage
-----

Authenticate with your BioLM account:

.. code-block:: bash

   biolmai login

The command will:

- Check for existing valid credentials
- If credentials are missing or invalid, open a browser for OAuth authorization
- Save credentials to :code:`~/.biolmai/credentials`

Options
-------

You can specify a custom OAuth client ID:

.. code-block:: bash

   biolmai login --client-id your-client-id

Or set the ``BIOLMAI_OAUTH_CLIENT_ID`` environment variable.

You can also specify custom OAuth scopes (supported: read, write, introspection):

.. code-block:: bash

   biolmai login --scope "read write"

Examples
--------

Login with default settings:

.. code-block:: bash

   biolmai login

Login with custom client ID:

.. code-block:: bash

   biolmai login --client-id abc123xyz

Login with custom scope (read and write for API access):

.. code-block:: bash

   biolmai login --scope "read write"

Command Reference
-----------------

.. click:: biolm.cli:login
   :prog: biolmai login
