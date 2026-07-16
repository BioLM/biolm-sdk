.. _biolm-hub:

===================================
Running inference through BioLM Hub
===================================

Every guide so far has assumed your model calls go to ``biolm.ai``. `biolm-hub
<https://github.com/BioLM/biolm-hub>`_ lets you change *where the inference runs*
without changing how you write code. You point the CLI and SDK at a hub gateway —
running locally with ``bh serve`` or deployed to your own infrastructure — and the
same ``Model`` calls, ``biolm model run`` commands, and discovery helpers resolve
against that gateway instead. This is useful when you want models close to your
data, running on hardware you control, or serving a private catalog.

The key thing to understand is the split: **only model inference is routed through
the hub**. Platform login, workspaces, datasets, and protocol workflows still talk
to ``biolm.ai``. You authenticate the same way, and everything except the model
prediction path is unchanged. The hub is a swap for the inference endpoint, not a
replacement for the platform.

Serving models with biolm-hub
=============================

Gateways come from the `biolm-hub <https://github.com/BioLM/biolm-hub>`_
repository, which packages the model server and its ``bh`` command-line tool.
Clone it, install, deploy the models you want, and serve the gateway:

.. code-block:: bash

    git clone https://github.com/BioLM/biolm-hub
    cd biolm-hub
    make install
    source .venv/bin/activate
    bh deploy esm2
    bh serve

``bh deploy <model>`` provisions a model so the gateway can serve it, and
``bh serve`` starts the gateway that the SDK connects to. See the biolm-hub
repository for the full set of deployable models and deployment options — this
guide covers the SDK side and does not duplicate the hub's own reference.

Deploying to Modal
------------------

For hosted inference, biolm-hub can deploy models to `Modal
<https://modal.com>`_ from the same ``bh deploy`` workflow, giving you a gateway
URL instead of a local address. The mechanics of the Modal deployment — accounts,
scaling, and resource configuration — live in the biolm-hub repository. From the
SDK's perspective a Modal gateway is just another URL: once it is serving, you
point ``biolm hub set`` at that URL exactly as you would a local one.

Pointing the CLI and SDK at a hub
=================================

Once a gateway is serving, tell the CLI and SDK to use it with ``biolm hub set``.
With no argument it defaults to ``http://127.0.0.1:8000`` — the address
``bh serve`` uses locally — and it saves ``hub_api_url`` to
``~/.biolm/config.yaml`` so the setting persists across sessions:

.. code-block:: bash

    bh serve                 # in the biolm-hub repo (or a deployed gateway URL)
    biolm hub set            # defaults to http://127.0.0.1:8000
    biolm login              # platform auth still goes to biolm.ai
    biolm model run esm2-8m encode -i seq.json

After ``biolm hub set``, ``biolm model list``, ``show``, ``example``, and
``run`` all resolve against the hub's catalog, so you discover and call exactly
the models your gateway serves. Note that ``biolm login`` still authenticates
against ``biolm.ai`` — the hub set step only redirects inference.

If you need to override the saved config for a single run — for example to point
at a different gateway in CI without editing your config file — set
``BIOLM_BASE_API_URL`` in the environment. It takes precedence over the value in
``~/.biolm/config.yaml``:

.. code-block:: bash

    BIOLM_BASE_API_URL=http://127.0.0.1:8000 biolm model list

Browsing a hub's catalog
========================

While ``bh serve`` is running, open ``http://127.0.0.1:8000/catalog`` in a
browser to see the models the gateway serves, along with their actions and
schemas. This is the hub equivalent of the web `model catalog
<https://biolm.ai/models>`_ and the fastest way to confirm what a gateway
exposes before you script against it.

Discovering hub models from Python
==================================

The SDK can list a gateway's models directly from its OpenAPI spec, which is
handy in notebooks and scripts where you want to enumerate what is available
before making calls:

.. code-block:: python

    from biolm.hub.discovery import list_models_from_openapi

    models = list_models_from_openapi("http://127.0.0.1:8000/api/v1")
    for m in models:
        print(m["model_slug"], m["actions"])

Each entry reports the model slug and the actions it supports, mirroring the four
inference actions from :doc:`what-are-biolms`. Once you know a slug and action,
the call itself is identical to any other BioLM call — construct a ``Model`` and
invoke ``encode``, ``predict``, or ``generate`` — because the hub speaks the same
interface. For persisting the gateway URL and other helpers, see
``biolm.hub.config`` and ``biolm.hub.catalog`` in the SDK reference.

Where to go next
================

- :doc:`choosing-models` — discover models in the catalog, CLI, and Python,
  including against a local hub.
- :doc:`running-inference` — run a model from Python or ``biolm model run``.
- :doc:`what-are-biolms` — the four inference actions and the single call pattern
  the hub reuses.
- :doc:`../cli/hub` — the complete ``biolm hub`` command reference.
- :doc:`../sdk/hub` — the ``biolm.hub`` modules for config, discovery, and
  catalog helpers.
- `biolm-hub <https://github.com/BioLM/biolm-hub>`_ — deploy and serve gateways
  with ``bh deploy`` and ``bh serve``.
