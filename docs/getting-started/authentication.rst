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
then use it to set the environment variable :code:`BIOLMAI_TOKEN`.
For examples, see below.

.. note::

   Ensure you replace the example API token with your own.

.. code-block:: shell

    export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b

For Bash
^^^^^^^^

.. code-block:: shell

    echo "export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.bash_profile && source ~/.bash_profile

For Zsh
^^^^^^^

.. code-block:: shell

    echo "export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.zshrc && source ~/.zshrc

For Python
^^^^^^^^^^

.. code-block:: python

    import os
    os.environ['BIOLMAI_TOKEN'] = '9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b'


CLI authentication
------------------

Alternatively, with the :code:`biolm-sdk` package installed, in your Terminal run :code:`biolm login`.
This uses OAuth 2.0 to authenticate via your browser.

OAuth Login
^^^^^^^^^^^

The login command uses OAuth 2.0 with PKCE (Proof Key for Code Exchange) for secure authentication.
It checks for existing credentials first, and only opens a browser if credentials are missing or invalid.

.. code-block:: shell

    $ biolm login

    Starting OAuth login...
    A browser window will open for authorization.
    Opened browser for authorization...
    Waiting for authorization...
    
    Login succeeded! Credentials saved to ~/.biolm/credentials

If you already have valid credentials, the command will inform you:

.. code-block:: shell

    $ biolm login

    Valid credentials found. You are already logged in.
    Credentials location: ~/.biolm/credentials
    Run `biolm status` to view your authentication status.

Login Options
^^^^^^^^^^^^^

You can specify a custom OAuth client ID and scope:

.. code-block:: shell

    # Specify a custom OAuth client ID
    $ biolm login --client-id YOUR_CLIENT_ID

    # Or set it via environment variable
    $ export BIOLMAI_OAUTH_CLIENT_ID=YOUR_CLIENT_ID
    $ biolm login

    # Specify custom scope (supported: read, write, introspection)
    $ biolm login --scope "read write"

OAuth Configuration
^^^^^^^^^^^^^^^^^^^

The OAuth login uses a fixed redirect URI: :code:`http://127.0.0.1:8765/callback`
(or :code:`http://localhost:8765/callback`). This port must be available on your
machine. If port 8765 is in use, you'll need to close the application using it
or configure a different redirect URI.

.. note::

   If using a custom OAuth client ID, ensure the redirect URI is registered in
   your BioLM OAuth Application (Console → OAuth Apps). For flexible port support,
   register :code:`http://127.0.0.1/callback` (no port) per RFC 8252.

The callback server may use HTTPS with a self-signed certificate when the
:code:`cryptography` package is installed. Your browser will display a security
warning about the certificate - this is expected and safe for localhost. You can
proceed by clicking "Advanced" and then "Proceed to 127.0.0.1" (or similar).

.. note::

   HTTPS support requires the :code:`cryptography` package. If it's not installed,
   the CLI will fall back to HTTP and display a warning. Install it with:
   :code:`pip install cryptography`

Credentials are saved to :code:`~/.biolm/credentials` in JSON format with:

- :code:`access`: Access token
- :code:`refresh`: Refresh token
- :code:`expires_at`: Token expiration timestamp
- :code:`token_url`: OAuth token endpoint
- :code:`client_id`: OAuth client ID

Legacy Username/Password Login
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   The legacy username/password login method is deprecated. Use OAuth login instead.
   The legacy method does not work with social logins (Google, GitHub, etc.).
