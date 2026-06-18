Models
======

The BioLM Python SDK provides a high-level interface for working with BioLM models.

**Quick start:**

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

For more control, use :doc:`usage/usage` and the :doc:`api-reference/index` (e.g. ``biolmai.models``).
