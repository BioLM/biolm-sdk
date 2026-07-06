=========
biolm-sdk
=========


.. image:: https://img.shields.io/pypi/v/biolm-sdk.svg
        :target: https://pypi.org/project/biolm-sdk

.. image:: https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml

.. image:: https://img.shields.io/badge/docs-docs.biolm.ai-blue
        :target: https://docs.biolm.ai




Python SDK and CLI for `BioLM <https://biolm.ai>`_ and `biolm-hub <https://github.com/BioLM/biolm-hub>`_.

* Repository: https://github.com/BioLM/biolm-sdk
* PyPI: https://pypi.org/project/biolm-sdk/

Install
=======

.. code-block:: bash

    pip install biolm-sdk
    pip install "biolm-sdk[pipeline]"   # optional pipeline extras

Import as ``biolm`` in Python. The legacy ``biolmai`` PyPI package is deprecated; see the migration guide in the repository docs.

Open-source models (biolm-hub)
================================

.. code-block:: bash

    # In biolm-hub: bh serve
    biolm hub set
    biolm model list
    biolm model run esm2-8m encode -i seq.json

See ``docs/cli/hub.rst``.

Basic usage
===========

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

Asynchronous usage
==================

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

Features
========

- High-level ``biolm()`` constructor for quick API calls
- Sync and async interfaces with automatic batching
- ``biolm hub set`` for local biolm-hub gateways
- Optional pipeline framework with DuckDB-backed caching
- CLI: ``biolm login``, ``biolm model``, ``biolm protocol``, and more

* Free software: Apache Software License 2.0
* Documentation: https://docs.biolm.ai
* Issues: https://github.com/BioLM/biolm-sdk/issues
