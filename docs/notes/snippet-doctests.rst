========================================
Guide snippet doctests (maintainers)
========================================

Executable checks for **offline-safe** patterns used in ``docs/guide/``.
These mirror the construct / import / I/O shapes in the narrative guides
without calling the live BioLM API.

Network examples (``Model.encode``, ``pipeline.run()``, ``Finetune.wait``,
``biolm hub``, datasets) remain narrative-only and are not executed here.

Run with::

    make docs-doctest


Items normalization (``how-biolms-work``, ``what-are-biolms``)
=============================================================

.. testcode:: items-normalization

    from biolm.core.utils import prepare_items_for_api

    data, is_lol = prepare_items_for_api("MSILVTRPSPAGEEL", type="sequence")
    assert is_lol is False
    assert data == [{"sequence": "MSILVTRPSPAGEEL"}]

    data, is_lol = prepare_items_for_api({"sequence": "MSILVTRPSPAGEEL"})
    assert data == [{"sequence": "MSILVTRPSPAGEEL"}]

    # type alongside flat dicts is ignored (not rejected)
    data, _ = prepare_items_for_api([{"sequence": "AAA"}], type="sequence")
    assert data == [{"sequence": "AAA"}]

    # type + list-of-lists of dicts is rejected
    try:
        prepare_items_for_api([[{"sequence": "AAA"}]], type="sequence")
    except ValueError as exc:
        assert "list of lists" in str(exc).lower() or "Do not specify" in str(exc)
    else:
        raise AssertionError("expected ValueError for type + list-of-lists")


Model construct (``running-inference``, ``client-interfaces``)
=============================================================

.. testcode:: model-construct

    from biolm import Model
    from biolm import biolm as biolm_fn

    model = Model("esm2-8m")
    assert model.name == "esm2-8m"
    assert callable(model.encode)
    assert callable(model.predict)
    assert callable(model.generate)
    assert callable(model.lookup)
    assert callable(biolm_fn)


``biolm.io`` loaders and writers (``sequence-and-structure-data``)
=================================================================

.. testcode:: io-fasta-csv-json-pdb

    from biolm.io import (
        load_csv,
        load_fasta,
        load_json,
        load_pdb,
        to_csv,
        to_fasta,
        to_json,
        to_pdb,
    )

    fasta_path = _DOCTEST_TMP / "candidates.fasta"
    fasta_path.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nMKTAYIAKQRQ\n")
    items = load_fasta(fasta_path)
    assert len(items) == 2
    assert items[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
    assert items[0]["id"] == "seq1"
    assert "metadata" in items[0]

    out_fasta = _DOCTEST_TMP / "out.fasta"
    to_fasta(items, out_fasta)
    assert out_fasta.exists()

    csv_path = _DOCTEST_TMP / "library.csv"
    csv_path.write_text("sequence,id,score\nACDEFGHIKLMNPQRSTVWY,seq1,0.95\n")
    rows = load_csv(csv_path, sequence_key="sequence")
    assert rows[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
    assert rows[0]["score"] == "0.95"
    to_csv(rows, _DOCTEST_TMP / "out.csv")

    json_path = _DOCTEST_TMP / "payload.json"
    json_path.write_text('{"sequence": "AAA", "id": "x"}')
    assert load_json(json_path) == [{"sequence": "AAA", "id": "x"}]

    envelope = _DOCTEST_TMP / "envelope.json"
    envelope.write_text('{"items": [{"sequence": "BBB"}]}')
    assert load_json(envelope) == [{"sequence": "BBB"}]

    jsonl_path = _DOCTEST_TMP / "rows.jsonl"
    jsonl_path.write_text('{"sequence": "CCC"}\n{"sequence": "DDD"}\n')
    assert [r["sequence"] for r in load_json(jsonl_path)] == ["CCC", "DDD"]
    to_json([{"sequence": "EEE"}], _DOCTEST_TMP / "out.json")

    pdb_path = _DOCTEST_TMP / "backbone.pdb"
    pdb_path.write_text(
        "HEADER    TEST\n"
        "ATOM      1  N   MET A   1      11.104  13.207   8.134  1.00 20.00           N\n"
        "END\n"
    )
    structures = load_pdb(pdb_path)
    assert len(structures) == 1
    assert "pdb" in structures[0]
    assert "ATOM" in structures[0]["pdb"]
    to_pdb(structures, _DOCTEST_TMP / "out.pdb")


Saturation mutagenesis config (``saturation-mutagenesis``)
==========================================================

.. testcode:: sat-mut-config

    from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

    config = SaturationMutagenesisConfig(
        parent_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        scoring_model="thermompnn-d",
        positions=[3, 7, 10],
        score_field="ddg",
        top_n=25,
        ascending=True,
        pdb_str=None,
        batch_size=8,
    )
    assert config.score_field == "ddg"
    assert config.top_n == 25
    assert config.ascending is True
    assert config.positions == [3, 7, 10]

    pipeline = GenerativePipeline(configs=[config])
    assert pipeline is not None


Iterative masking DMS config (``iterative-masking-dms``)
=======================================================

.. testcode:: iterative-masking-config

    from biolm.pipeline import GenerativePipeline, IterativeMaskingDMSConfig

    config = IterativeMaskingDMSConfig(
        parent_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        model_name="esm2-650m",
        positions=[2, 5, 8],
        rounds=2,
        exclude_synonymous=True,
        batch_size=32,
    )
    assert config.rounds == 2
    assert config.action == "predict"
    assert config.exclude_synonymous is True

    try:
        IterativeMaskingDMSConfig(
            parent_sequence="MKTAYIAKQRQ",
            model_name="esm2-650m",
            rounds=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("rounds=3 must raise ValueError")

    pipeline = GenerativePipeline(configs=[config])
    assert pipeline is not None


Direct generation config (``structure-conditioned-generation``)
==============================================================

.. testcode:: direct-generation-config

    from biolm.pipeline import DirectGenerationConfig, GenerativePipeline

    pdb_path = _DOCTEST_TMP / "protein.pdb"
    pdb_path.write_text("HEADER    TEST\nEND\n")

    mpnn = DirectGenerationConfig(
        model_name="protein-mpnn",
        structure_path=str(pdb_path),
        item_field="pdb",
        params={"batch_size": 50, "temperature": 0.1},
        label="mpnn_T0.1",
    )
    assert mpnn.item_field == "pdb"
    assert mpnn.label == "mpnn_T0.1"

    dsm = DirectGenerationConfig(
        model_name="dsm-150m-base",
        sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        item_field="sequence",
        params={
            "num_sequences": 100,
            "temperature": 1.0,
            "remasking": "low_confidence",
            "step_divisor": 8,
        },
        label="dsm_baseline",
    )
    assert dsm.item_field == "sequence"

    pipeline = GenerativePipeline(configs=[mpnn, dsm])
    assert pipeline is not None


DataPipeline stages without API (``pipeline-workflows``)
=======================================================

.. testcode:: data-pipeline-construct

    from biolm.pipeline import DataPipeline
    from biolm.pipeline.filters import RankingFilter, ThresholdFilter

    sequences = ["MKTAYIAKQRQ", "MENDEL", "ACDEFGHIKLMNPQRSTVWY"]
    pipeline = DataPipeline(sequences=sequences, datastore=str(_DOCTEST_TMP / "run.duckdb"))
    threshold = ThresholdFilter("length", min_value=5)
    ranking = RankingFilter("length", n=2, ascending=False)
    assert threshold.column == "length"
    assert ranking.n == 2
    assert hasattr(pipeline, "run")
    assert hasattr(pipeline, "results")
    assert hasattr(pipeline, "get_final_data")
    assert hasattr(pipeline, "add_filter")


Pipeline metadata / context manager (``pipeline-caching``)
=========================================================

.. testcode:: pipeline-caching-construct

    from biolm.pipeline import DataPipeline

    db = str(_DOCTEST_TMP / "cache.duckdb")
    with DataPipeline(sequences=["MKTAYIAKQRQ"], datastore=db) as pipeline:
        meta = pipeline.metadata
        assert meta.db_path is not None
        assert meta.pipeline_id
        assert meta.cache_dir is not None


Finetune client surface (``finetuning-models``)
==============================================

.. testcode:: finetune-surface

    from biolm.finetune import TERMINAL_STATUSES, Finetune

    assert callable(Finetune.xgboost)
    assert callable(Finetune.dsm_stage1)
    assert callable(Finetune.dsm_stage2)
    assert callable(Finetune.dsm_rl)
    assert callable(Finetune.wait)
    assert callable(Finetune.get_run)
    assert callable(Finetune.progress)
    assert callable(Finetune.cancel)
    assert callable(Finetune.list_runs)
    assert callable(Finetune.xgboost_async)
    assert TERMINAL_STATUSES == {"succeeded", "failed", "cancelled", "error"}


Platform client import (``account-management``)
==============================================

.. testcode:: platform-client-import

    from biolm import PlatformClient

    assert PlatformClient is not None
