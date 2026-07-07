``biolm dataset``
===================

Work with MLflow-backed datasets.

Usage
-----

List datasets, show details, upload, or download.

.. code-block:: bash

   biolm dataset list
   biolm dataset show DATASET_ID
   biolm dataset upload DATASET_ID FILE_PATH
   biolm dataset download DATASET_ID OUTPUT_PATH

Command Reference
-----------------

.. click:: biolm.cli:dataset
   :prog: biolm dataset
   :show-nested:
