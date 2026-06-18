``biolmai dataset``
===================

Work with MLflow-backed datasets.

Usage
-----

List datasets, show details, upload, or download.

.. code-block:: bash

   biolmai dataset list
   biolmai dataset show DATASET_ID
   biolmai dataset upload DATASET_ID FILE_PATH
   biolmai dataset download DATASET_ID OUTPUT_PATH

Command Reference
-----------------

.. click:: biolm.cli:dataset
   :prog: biolmai dataset
   :show-nested:
