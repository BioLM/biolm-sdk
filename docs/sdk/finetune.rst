biolm.finetune
==============

The :class:`biolm.finetune.Finetune` class wraps BioLM finetuning APIs (XGBoost and
DSM workflows) so you can launch and track runs from Python without browser cookies.

Install ``biolm-sdk`` and authenticate with ``BIOLM_TOKEN`` or ``biolm login``.

XGBoost finetune
----------------

.. code-block:: python

   from biolm.finetune import Finetune

   result = Finetune.xgboost(
       train_data=[{"sequence": "MKTAYIAKQRQ", "label": 1}],
       embedding_models=["esm2-8m"],
       task_type="classification",
       run_name="my-xgb-run",
   )
   print(result["run_id"])

All methods have async variants (e.g. ``Finetune.xgboost_async``). Pass ``api_key=``
or set ``BIOLM_TOKEN`` for authentication.

See :doc:`../api-reference/modules` for the full ``Finetune`` API.
