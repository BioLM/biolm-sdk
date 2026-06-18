.. _quickstart-sdk:

==========
Quickstart
==========

Install the package:

.. code-block:: bash

    pip install biolmai

Basic usage (one-off calls with ``biolm()``). These calls are synchronous (blocking) but use an async backend internally, so you get concurrent batch performance without writing async code.

.. code-block:: python

    from biolm import biolm

    # Encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Predict a batch of sequences
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

    # Write results to disk
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

    # Check for errors when not raising (raise_httpx=False)
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "BADSEQ"], raise_httpx=False)
    for r in result:
        if isinstance(r, dict) and "error" in r:
            print("Error:", r["error"])
        else:
            print("OK")

**Or use the class-based** :code:`Model` **when** you're working with one model and want to call :code:`.encode()`, :code:`.predict()`, or :code:`.generate()` on it:

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    result = model.encode(type="sequence", items="MSILVTRPSPAGEEL")

    model = Model("esmfold")
    result = model.predict(type="sequence", items=["SEQ1", "SEQ2"])

For core concepts (sync/async, batching, error handling, etc.), see :doc:`concepts`. For SDK usage and examples, see :doc:`../sdk/models`.
