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

    $ pip install 'biolm[server]'    # local Modal proxy (biolm server)
    $ pip install 'biolm[pipeline]'  # pipeline features

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
