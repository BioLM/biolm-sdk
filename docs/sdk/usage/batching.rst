========================
Batching and Input Flexibility
========================

The BioLM Python client supports a wide variety of input formats and batching strategies to maximize flexibility and efficiency. This document explains all supported input types, how auto-batching works, and how to use advanced batching for custom workflows.

------------------------
Supported Input Formats
------------------------

You can provide input in several ways:

**1. Single item (string or dict):**
  - For a single sequence or context.
  - Example:

    .. code-block:: python

        biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

**2. List of values (strings, numbers, etc):**
  - For a batch of simple items (e.g. sequences). Pass a type so the client knows how to interpret the values.
  - Example:

    .. code-block:: python

        biolm(entity="esm2-8m", action="encode", type="sequence", items=["SEQ1", "SEQ2"])

**3. List of dicts:**
  - For a batch of structured items. Type is inferred from the dict keys.
  - Example:

    .. code-block:: python

        biolm(entity="esmfold", action="predict", items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}])

**4. Generators and iterators (memory-efficient):**
  - Pass a generator or any iterable instead of a list. The client consumes it batch-by-batch, so you never hold all items in memory at once.
  - Ideal for large files, streams, or lazy data pipelines.
  - **Note:** The generator is fully consumed during the call; you cannot iterate it again afterwards.
  - Example:

    .. code-block:: python

        def sequences_from_file(path):
            with open(path) as f:
                for line in f:
                    seq = line.strip()
                    if seq:
                        yield {"sequence": seq}

        result = biolm(entity="esm2-8m", action="encode", items=sequences_from_file("sequences.txt"))

**5. List of lists of dicts (advanced/manual batching):**
  - Each inner list is treated as a batch and sent as a single API request.
  - Useful for custom batching, controlling batch size, or mixing valid/invalid items.
  - Example:

    .. code-block:: python

        batches = [
            [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}],  # batch 1
            [{"sequence": "SEQ3"}],                        # batch 2
        ]
        biolm(entity="esmfold", action="predict", items=batches)

------------------------
How auto-batching works
------------------------

The client asks the API for the model’s maximum batch size, splits your input into batches of that size, and sends each batch as a separate request. Results come back in the same order as your input. You don’t need to split manually.

**Example:**

.. code-block:: python

    # If the model's max batch size is 8, this will be split into 2 requests:
    items = ["SEQ" + str(i) for i in range(12)]
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=items)
    # result is a list of 12 results, in order

------------------------
Advanced: Manual Batching with List of Lists
------------------------

- If you provide a list of lists of dicts, **each inner list is treated as a batch**.
- This disables auto-batching: you control the batch size and composition.
- Useful for:
    - Forcing certain items to be batched together (e.g., for error isolation).
    - Working around API limits or bugs.
    - Testing error handling with mixed valid/invalid batches.

**Example:**

.. code-block:: python

    # Two batches: first has 2 items, second has 1
    items = [
        [{"sequence": "SEQ1"}, {"sequence": "BADSEQ"}],  # batch 1
        [{"sequence": "SEQ3"}],                          # batch 2
    ]
    result = biolm(entity="esmfold", action="predict", items=items, stop_on_error=False)
    # result is a flat list: [result1, result2, result3]

------------------------
Input validation
------------------------

- List of dicts: type is inferred from the keys.
- List of plain values (e.g. strings): pass a type (e.g. sequence) so the client knows how to interpret them.
- List of lists (manual batching): each inner list must be a list of dicts.

------------------------
Sequence validity
------------------------

Protein sequences must use only valid amino acid letters. The client accepts the standard set (e.g. ACDEFGHIKLMNPQRSTVWYBXZUO).

------------------------
Batch size and schema
------------------------

You can read the maximum batch size from the schema:

.. code-block:: python

    from biolm.core.http import BioLMApi
    model = BioLMApi("esm2-8m")
    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)
    print("Max batch size:", max_batch)

------------------------
Batching and errors
------------------------

If a batch has invalid items, the whole batch may fail. You can halt on the first error batch or process all batches and get error dicts in the results; with the API client you can also retry failed batches as single items. See :doc:`error-handling` for details and examples.

------------------------
Summary Table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - Input Format
     - Auto-batching?
     - Use Case
   * - Single value/dict
     - Yes
     - Single item
   * - List of values
     - Yes (pass type)
     - Batch of simple items
   * - List of dicts
     - Yes
     - Batch of structured items
   * - Generator/iterator
     - Yes (consumed in batches)
     - Large streams, low memory
   * - List of lists of dicts
     - No (manual batching)
     - Custom batch control

------------------------
Examples
------------------------

**Batching with list of dicts:**

.. code-block:: python

    from biolm import biolm

    items = [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}]
    result = biolm(entity="esm2-8m", action="encode", items=items)

**Batching with list of values:**

.. code-block:: python

    items = ["SEQ1", "SEQ2"]
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=items)

**Manual batching with list of lists:**

.. code-block:: python

    batches = [
        [{"sequence": "SEQ1"}, {"sequence": "BADSEQ"}],  # batch 1
        [{"sequence": "SEQ3"}],                          # batch 2
    ]
    result = biolm(entity="esmfold", action="predict", items=batches, stop_on_error=False)

------------------------
Best practices
------------------------

- Prefer a list of values or dicts and let the client auto-batch.
- For large datasets (files, streams), use a generator so items are consumed batch-by-batch.
- For very large result sets, write to disk (see :ref:`disk-output` in :doc:`usage`).
- Use manual batching (list of lists) only when you need custom batch sizes or composition.

------------------------
See Also
------------------------

- :doc:`error-handling`
- :doc:`usage` (includes Disk output)
- :doc:`../faq`
