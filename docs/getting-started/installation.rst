.. highlight:: shell

============
Installation
============


Stable release
--------------

To install BioLM, run this command in your terminal:

.. code-block:: console

    $ pip install biolm

Optional extras:

.. code-block:: console

    $ pip install 'biolm[pipeline]'  # pipeline features

For open-source models, install and run `biolm-hub <https://github.com/BioLM/biolm-hub>`_,
then connect with ``biolm hub set``. See :doc:`../cli/hub`.

The ``biolmai`` package name is deprecated; use ``biolm``. See :doc:`migration-1.0`.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for BioLM can be downloaded from the `GitHub repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/BioLM/py-biolm

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/BioLM/py-biolm/tarball/production

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e ".[server]"


.. _GitHub repo: https://github.com/BioLM/py-biolm
.. _tarball: https://github.com/BioLM/py-biolm/tarball/production
