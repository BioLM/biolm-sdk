"""Tests for biolm.lab.runs."""

from __future__ import annotations

from biolm.lab.runs import LabRun, list_run_ids, list_runs, load_run, save_run


def test_save_load_list_run(tmp_path):
    run = LabRun(
        run_id="run_abc",
        connector="adaptyv",
        service_id="adaptyv-lltp.expression-v1",
        experiment="express",
        handle={"vendor_order_id": "exp-1", "experiment_id": "exp-1"},
    )
    path = save_run(run, root=tmp_path)
    assert path.is_file()

    loaded = load_run("run_abc", root=tmp_path)
    assert loaded.connector == "adaptyv"
    assert loaded.handle["experiment_id"] == "exp-1"

    assert list_run_ids(root=tmp_path) == ["run_abc"]
    runs = list_runs(root=tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == "run_abc"


def test_save_updates_status(tmp_path):
    run = LabRun(
        run_id="r1",
        connector="twist",
        service_id="twist-lltp.dna-synthesis-v1",
        handle={"quote_id": "q1", "vendor_order_id": "q1"},
    )
    save_run(run, root=tmp_path)
    run.status = "blocked"
    run.last_status = {"patches": [{"requirement_id": "req.quote-approval", "status": "AWAITING"}]}
    save_run(run, root=tmp_path)
    loaded = load_run("r1", root=tmp_path)
    assert loaded.status == "blocked"
    assert loaded.last_status["patches"][0]["status"] == "AWAITING"
