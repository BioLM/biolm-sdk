"""Tests for local-first biolm.datasets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from biolm.cli import cli
from biolm.datasets import (
    DatasetClient,
    DatasetExistsError,
    DatasetNotFoundError,
    DatasetSchemaError,
    DuplicateDatasetIdError,
)
from biolm.datasets.schema import DATASET_YAML, build_meta, load_dataset_yaml, write_dataset_yaml


@pytest.fixture
def home_root(tmp_path, monkeypatch):
    """Redirect HOME and cwd so discovery uses temp dirs."""
    monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "biolm.hub.config.read_config",
        lambda: {},
    )
    return tmp_path


class TestSchema:
    def test_build_and_roundtrip(self, tmp_path):
        meta = build_meta("ds-1", type="files", tags=["a"], description="hi")
        write_dataset_yaml(tmp_path, meta)
        loaded = load_dataset_yaml(tmp_path)
        assert loaded.id == "ds-1"
        assert loaded.type == "files"
        assert loaded.tags == ["a"]
        assert loaded.description == "hi"

    def test_missing_id_raises(self, tmp_path):
        yaml_path = tmp_path / DATASET_YAML
        yaml_path.write_text("schema_version: 1\n")
        with pytest.raises(DatasetSchemaError):
            load_dataset_yaml(tmp_path)


class TestDatasetClient:
    def test_create_list_get_add(self, home_root):
        client = DatasetClient(primary_root=home_root / ".biolm" / "datasets")
        ds = client.create("finetuning-v1", tags=["finetune"], type="files")
        assert ds.id == "finetuning-v1"
        assert (ds.path / "data").is_dir()
        assert (ds.path / DATASET_YAML).is_file()

        src = home_root / "train.csv"
        src.write_text("seq\nMKT\n")
        dest = ds.add(src)
        assert dest.exists()
        assert any(p.name == "train.csv" for p in ds.files())

        listed = client.list(tag="finetune")
        assert [d.id for d in listed] == ["finetuning-v1"]

        by_id = client.get("finetuning-v1")
        assert by_id.path == ds.path

        by_path = client.get(ds.path)
        assert by_path.id == "finetuning-v1"

    def test_create_refuses_overwrite(self, home_root):
        client = DatasetClient(primary_root=home_root / ".biolm" / "datasets")
        client.create("ds-1")
        with pytest.raises(DatasetExistsError):
            client.create("ds-1")

    def test_init_existing_dir(self, home_root):
        root = home_root / ".biolm" / "datasets"
        root.mkdir(parents=True)
        existing = root / "adopted"
        existing.mkdir()
        (existing / "data.csv").write_text("x")

        client = DatasetClient(primary_root=root, roots=[root])
        ds = client.init(existing, id="adopted-set", tags=["raw"])
        assert ds.id == "adopted-set"
        assert (existing / "data.csv").exists()
        assert client.get("adopted-set").path == existing.resolve()

    def test_init_outside_roots_warns(self, home_root):
        outside = home_root / "outside"
        outside.mkdir()
        client = DatasetClient(primary_root=home_root / ".biolm" / "datasets")
        with pytest.warns(UserWarning, match="outside discovery roots"):
            client.init(outside, id="outside-ds")

        with pytest.raises(DatasetNotFoundError):
            client.get("outside-ds")
        assert client.get(outside).id == "outside-ds"

    def test_duplicate_id_error(self, home_root):
        root_a = home_root / "root_a"
        root_b = home_root / "root_b"
        (root_a / "one").mkdir(parents=True)
        (root_b / "two").mkdir(parents=True)
        write_dataset_yaml(root_a / "one", build_meta("same-id"))
        write_dataset_yaml(root_b / "two", build_meta("same-id"))

        client = DatasetClient(roots=[root_a, root_b], primary_root=root_a)
        with pytest.raises(DuplicateDatasetIdError):
            client.list()

    def test_list_filters(self, home_root):
        root = home_root / ".biolm" / "datasets"
        client = DatasetClient(primary_root=root, roots=[root])
        client.create("a", type="files", tags=["x"])
        client.create("b", type="seqframe", tags=["y"])
        assert [d.id for d in client.list(type="seqframe")] == ["b"]
        assert [d.id for d in client.list(tag="x")] == ["a"]


class TestCLILocalDatasets:
    def test_cli_create_list_show_add(self, home_root):
        root = home_root / ".biolm" / "datasets"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dataset", "create", "cli-ds", "--tag", "t1", "--root", str(root)],
        )
        assert result.exit_code == 0, result.output
        assert "cli-ds" in result.output

        src = home_root / "f.txt"
        src.write_text("hi")
        result = runner.invoke(cli, ["dataset", "add", str(root / "cli-ds"), str(src)])
        assert result.exit_code == 0, result.output

        result = runner.invoke(cli, ["dataset", "list", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert any(row["id"] == "cli-ds" for row in data)

        result = runner.invoke(cli, ["dataset", "show", "cli-ds"])
        assert result.exit_code == 0, result.output
        assert "cli-ds" in result.output

    def test_cli_init(self, home_root):
        root = home_root / ".biolm" / "datasets"
        root.mkdir(parents=True)
        folder = root / "preexisting"
        folder.mkdir()
        (folder / "a.csv").write_text("1")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dataset", "init", str(folder), "--id", "preexisting"],
        )
        assert result.exit_code == 0, result.output
        assert (folder / DATASET_YAML).is_file()

    def test_cli_show_missing(self, home_root):
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "show", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
