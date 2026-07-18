.. highlight:: shell

============
Installation
============


Stable release
--------------

To install BioLM SDK, run this command in your terminal:

.. code-block:: console

    $ pip install biolm-sdk

Optional extras:

.. code-block:: console

    $ pip install "biolm-sdk[pipeline]"  # pipeline features
    $ pip install "biolm-sdk[seqframe]"  # SeqFrame (Parquet + DuckDB sequence tables)
    $ pip install "biolm-sdk[mlflow]"    # MLflow datasets and protocol logging (CLI)

For open-source models, install and run `biolm-hub <https://github.com/BioLM/biolm-hub>`_,
then connect with ``biolm hub set``. See :doc:`../cli/hub`.

The ``[mlflow]`` extra is required for ``biolm dataset`` and ``biolm protocol log``.
See :sdklink:`SDK plugins <../../sdk/index.html#plugins-optional>`.

The ``biolmai`` package name is deprecated; use ``biolm-sdk``. See :doc:`../notes/migration-1.0`.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for biolm-sdk can be downloaded from the `GitHub repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:BioLM/biolm-sdk.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/BioLM/biolm-sdk/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e .


.. _GitHub repo: https://github.com/BioLM/biolm-sdk
.. _tarball: https://github.com/BioLM/biolm-sdk/tarball/main
