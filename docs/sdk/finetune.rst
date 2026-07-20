``biolm.finetune``
==================

Launch and track BioLM finetuning runs (XGBoost and DSM) from Python without
browser cookies. Install ``biolm-sdk`` and authenticate with ``BIOLM_TOKEN`` or
``biolm account login``.

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

DSM finetune
------------

DSM is a two-stage workflow (masked-LM pretrain, then RL). Launch stage 1:

.. code-block:: python

   from biolm.finetune import Finetune

   result = Finetune.dsm_stage1(
       train_data=[{"sequence": "MKTAYIAKQRQ"}],
       run_name="my-dsm-stage1",
   )
   run_id = result["run_id"]
   Finetune.wait(run_id)

All methods have async variants (e.g. ``Finetune.xgboost_async``,
``Finetune.dsm_stage1_async``). Pass ``api_key=`` or set ``BIOLM_TOKEN`` for
authentication.

API
---

.. autoclass:: biolm.finetune.Finetune
   :members: xgboost, dsm_stage1, dsm_stage2, dsm_rl, wait, progress, get_run, cancel
   :undoc-members:

See also
--------

- :doc:`../api-reference/biolm` — ``biolm.finetune`` module reference
