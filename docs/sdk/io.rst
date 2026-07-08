``biolm.io``
============

Convert between biological file formats (FASTA, CSV, PDB, JSON) and BioLM API
request/response structures. For writing API results to disk from clients, see
:ref:`disk-output` in :doc:`../intro/client-interfaces`.

Primary functions
-----------------

.. autofunction:: biolm.io.load_fasta
.. autofunction:: biolm.io.to_fasta
.. autofunction:: biolm.io.load_csv
.. autofunction:: biolm.io.to_csv
.. autofunction:: biolm.io.load_pdb
.. autofunction:: biolm.io.to_pdb

FASTA Format
~~~~~~~~~~~~

Loading FASTA Files
~~~~~~~~~~~~~~~~~~~

The ``load_fasta()`` function parses FASTA files and returns a list of dictionaries suitable for API requests:

.. code-block:: python

    from biolm.io import load_fasta

    # Load sequences from file
    items = load_fasta("sequences.fasta")

    # Each item contains:
    # - "sequence": The sequence string
    # - "id": Sequence identifier from header
    # - "metadata": Additional metadata (if present)

    print(items[0])
    # {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'metadata': {}}

FASTA files support:
- Multi-line sequences (wrapped sequences are automatically concatenated)
- Headers with metadata (pipe-separated or space-separated)
- Multiple sequences in a single file

**Example FASTA file:**

.. code-block:: text

    >seq1|protein|test
    ACDEFGHIKLMNPQRSTVWY
    >seq2 description here
    MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL

Writing FASTA Files
~~~~~~~~~~~~~~~~~~~

The ``to_fasta()`` function writes sequences to FASTA format:

.. code-block:: python

    from biolm.io import to_fasta

    # Data from API response
    data = [
        {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"},
        {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "id": "seq2"},
    ]

    # Write to file
    to_fasta(data, "output.fasta")

    # With metadata
    data_with_metadata = [
        {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "id": "seq1",
            "metadata": {"description": "Test sequence", "type": "protein"},
        }
    ]
    to_fasta(data_with_metadata, "output.fasta")

You can also use a custom sequence key:

.. code-block:: python

    data = [{"seq": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
    to_fasta(data, "output.fasta", sequence_key="seq")

CSV Format
----------

Loading CSV Files
~~~~~~~~~~~~~~~~~

The ``load_csv()`` function parses CSV files with headers:

.. code-block:: python

    from biolm.io import load_csv

    # Load CSV file
    items = load_csv("data.csv")

    # Each row becomes a dictionary with column headers as keys
    print(items[0])
    # {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'score': '0.95'}

You can validate that a specific column exists:

.. code-block:: python

    # Raises ValueError if "sequence" column is missing
    items = load_csv("data.csv", sequence_key="sequence")

**Example CSV file:**

.. code-block:: text

    sequence,id,score,description
    ACDEFGHIKLMNPQRSTVWY,seq1,0.95,Test sequence 1
    MKTAYIAKQRQISFVKSHFSRQ,seq2,0.87,Test sequence 2

Writing CSV Files
~~~~~~~~~~~~~~~~~

The ``to_csv()`` function writes data to CSV format:

.. code-block:: python

    from biolm.io import to_csv

    # Data from API response
    data = [
        {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1", "score": 0.95},
        {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "id": "seq2", "score": 0.87},
    ]

    # Write to file
    to_csv(data, "output.csv")

    # With custom fieldnames
    to_csv(data, "output.csv", fieldnames=["sequence", "id"])

Missing keys are automatically filled with empty strings.

PDB Format
----------

Loading PDB Files
~~~~~~~~~~~~~~~~~

The ``load_pdb()`` function reads PDB structure files:

.. code-block:: python

    from biolm.io import load_pdb

    # Load single-model PDB
    items = load_pdb("structure.pdb")

    # Returns: [{"pdb": "HEADER    TEST\nATOM      1  N   MET A   1\n..."}]

    # For multi-model PDBs, returns one item per model
    items = load_pdb("multi_model.pdb")
    # Returns: [{"pdb": "MODEL 1..."}, {"pdb": "MODEL 2..."}]

Writing PDB Files
~~~~~~~~~~~~~~~~~

The ``to_pdb()`` function writes PDB structures:

.. code-block:: python

    from biolm.io import to_pdb

    # Data from API response
    data = [{"pdb": "HEADER    TEST\nATOM      1  N   MET A   1\nEND\n"}]

    # Write to file
    to_pdb(data, "output.pdb")

    # Multiple structures are concatenated
    data = [
        {"pdb": "MODEL 1\nATOM...\nENDMDL\n"},
        {"pdb": "MODEL 2\nATOM...\nENDMDL\n"},
    ]
    to_pdb(data, "output.pdb")

Integration with Model Class
----------------------------

The io module is designed to work seamlessly with the ``Model`` class:

.. code-block:: python

    from biolm.io import load_fasta, to_csv
    from biolm import Model

    # Load sequences from FASTA
    items = load_fasta("sequences.fasta")

    # Use with model
    model = Model("esm2-8m")
    results = model.encode(items=items)

    # Export results to CSV
    to_csv(results, "results.csv")

**Complete workflow example:**

.. code-block:: python

    from biolm.io import load_fasta, to_csv
    from biolm import Model

    # 1. Load input sequences
    sequences = load_fasta("input.fasta")

    # 2. Process with model
    model = Model("esmfold")
    structures = model.predict(items=sequences)

    # 3. Export results
    to_csv(structures, "output.csv")

File-like Objects
-----------------

All functions support both file paths and file-like objects:

.. code-block:: python

    import io
    from biolm.io import load_fasta, to_fasta

    # Load from file-like object
    file_obj = io.StringIO(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
    items = load_fasta(file_obj)

    # Write to file-like object
    output = io.StringIO()
    to_fasta(items, output)
    content = output.getvalue()

Error Handling
--------------

The io module raises clear exceptions for common errors:

.. code-block:: python

    from biolm.io import load_fasta

    try:
        items = load_fasta("nonexistent.fasta")
    except FileNotFoundError:
        print("File not found")

    try:
        items = load_fasta("empty.fasta")
    except ValueError as e:
        print(f"Invalid file: {e}")

Common error types:
- ``FileNotFoundError``: File path doesn't exist
- ``ValueError``: File is empty, malformed, or missing required fields
- ``KeyError``: Missing required keys in data dictionaries

Best Practices
--------------

- Validate input files before processing; use try/except for ``FileNotFoundError`` and ``ValueError``
- Preserve metadata in FASTA headers when exporting
- Test round-trips: load → process → save → load

See also
--------

- :doc:`../api-reference/biolm.io` — full ``biolm.io`` module reference
- :ref:`disk-output` in :doc:`../intro/client-interfaces` — writing API results to disk
