"""Tests for BioLM definition recipes and build_model."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from biolm.models.definition import BIOLM_MANIFEST, BuiltPackage, build_model, load_recipe
from biolm.models.errors import BuildError, RecipeError


MINIMAL_RECIPE = """\
schema_version: 1
name: antibody-binder-clf
from: esm2-8m
layers:
  - type: embedding_head
    task: classification
    data: ./data/binders.csv
"""


@pytest.fixture
def recipe_tree(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "binders.csv").write_text("sequence,label\nMKTAYIAKQRQ,1\nMSILVTRPSPA,0\n")
    recipe = tmp_path / "antibody-binder-clf.yaml"
    recipe.write_text(MINIMAL_RECIPE)
    return recipe


class TestLoadRecipe:
    def test_valid_minimal(self, recipe_tree: Path):
        recipe = load_recipe(recipe_tree)
        assert recipe["name"] == "antibody-binder-clf"
        assert recipe["from"] == "esm2-8m"
        assert recipe["schema_version"] == 1
        assert len(recipe["layers"]) == 1
        assert recipe["layers"][0]["type"] == "embedding_head"
        assert recipe["layers"][0]["data_path"].is_file()
        assert recipe["layers"][0]["data_path"].is_absolute()

    def test_missing_name(self, tmp_path: Path):
        csv = tmp_path / "t.csv"
        csv.write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "from: esm2-8m\nlayers:\n  - type: embedding_head\n    task: classification\n"
            f"    data: {csv.name}\n"
        )
        with pytest.raises(RecipeError, match="name"):
            load_recipe(path)

    def test_missing_from(self, tmp_path: Path):
        csv = tmp_path / "t.csv"
        csv.write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nlayers:\n  - type: embedding_head\n    task: classification\n"
            f"    data: {csv.name}\n"
        )
        with pytest.raises(RecipeError, match="from"):
            load_recipe(path)

    def test_missing_data(self, tmp_path: Path):
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            "  - type: embedding_head\n    task: classification\n"
        )
        with pytest.raises(RecipeError, match="data"):
            load_recipe(path)

    def test_data_file_missing(self, tmp_path: Path):
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            "  - type: embedding_head\n    task: classification\n"
            "    data: ./missing.csv\n"
        )
        with pytest.raises(RecipeError, match="data"):
            load_recipe(path)

    def test_zero_layers(self, tmp_path: Path):
        path = tmp_path / "r.yaml"
        path.write_text("name: x\nfrom: esm2-8m\nlayers: []\n")
        with pytest.raises(RecipeError, match="exactly one"):
            load_recipe(path)

    def test_multiple_layers(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "binders.csv").write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            "  - type: embedding_head\n    task: classification\n    data: ./data/binders.csv\n"
            "  - type: embedding_head\n    task: regression\n    data: ./data/binders.csv\n"
        )
        with pytest.raises(RecipeError, match="exactly one"):
            load_recipe(path)

    def test_non_embedding_head(self, tmp_path: Path):
        csv = tmp_path / "t.csv"
        csv.write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            f"  - type: lora\n    task: classification\n    data: {csv.name}\n"
        )
        with pytest.raises(RecipeError, match="embedding_head"):
            load_recipe(path)

    def test_bad_task(self, tmp_path: Path):
        csv = tmp_path / "t.csv"
        csv.write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            f"  - type: embedding_head\n    task: clustering\n    data: {csv.name}\n"
        )
        with pytest.raises(RecipeError, match="task"):
            load_recipe(path)


class TestBuildModel:
    @pytest.fixture
    def home_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
        return tmp_path

    def test_build_writes_package(self, recipe_tree: Path, home_root: Path, monkeypatch):
        recipe_bytes = recipe_tree.read_bytes()

        def fake_xgboost(**kwargs):
            assert kwargs["embedding_models"] == ["esm2-8m"]
            assert kwargs["task_type"] == "classification"
            assert isinstance(kwargs["train_data"], str)
            assert "MKTAYIAKQRQ" in kwargs["train_data"]
            assert kwargs["run_name"] == "antibody-binder-clf"
            return {"run_id": "ALY_test123", "status": "scheduled"}

        def fake_wait(run_id, **kwargs):
            assert run_id == "ALY_test123"
            return {"run_id": run_id, "status": "succeeded", "metrics": {"auc": 0.9}}

        monkeypatch.setattr("biolm.models.definition.Finetune.xgboost", fake_xgboost)
        monkeypatch.setattr("biolm.models.definition.Finetune.wait", fake_wait)

        pkg = build_model(recipe_tree, tag="latest")
        assert isinstance(pkg, BuiltPackage)
        assert pkg.path == home_root / ".biolm" / "models" / "antibody-binder-clf" / "latest"
        manifest_path = pkg.path / BIOLM_MANIFEST
        assert manifest_path.is_file()
        assert recipe_tree.read_bytes() == recipe_bytes

        manifest = pkg.manifest
        assert manifest["name"] == "antibody-binder-clf"
        assert manifest["tag"] == "latest"
        assert manifest["from"]["slug"] == "esm2-8m"
        assert manifest["from"]["load"] == "lazy"
        assert manifest["layers"][0]["run_id"] == "ALY_test123"
        assert manifest["layers"][0]["artifact"]["load"] == "preload"
        assert Path(manifest["layers"][0]["data"]["path"]).is_absolute()
        assert manifest["actions"]["encode"]["input"] == "sequence"
        assert manifest["actions"]["encode"]["schema"] == "biolm.encode.v1"
        assert manifest["actions"]["predict"]["input"] == "sequence"
        assert manifest["actions"]["predict"]["task"] == "classification"
        assert manifest["actions"]["predict"]["schema"] == "biolm.predict.v1"
        assert manifest["built"]["status"] == "locked"
        assert "at" in manifest["built"]
        assert manifest["layers"][0].get("metrics") == {"auc": 0.9}

        on_disk = yaml.safe_load(manifest_path.read_text())
        assert on_disk["built"]["status"] == "locked"

    def test_artifact_uri_from_wait(self, recipe_tree: Path, home_root: Path, monkeypatch):
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_a"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {
                "run_id": run_id,
                "status": "succeeded",
                "artifact_uri": "https://example.com/model.joblib",
            },
        )
        pkg = build_model(recipe_tree)
        assert pkg.manifest["layers"][0]["artifact"]["uri"] == "https://example.com/model.joblib"

    def test_recipe_actions_validated(self, tmp_path: Path):
        csv = tmp_path / "t.csv"
        csv.write_text("sequence,label\nM,1\n")
        path = tmp_path / "r.yaml"
        path.write_text(
            "name: x\nfrom: esm2-8m\nlayers:\n"
            f"  - type: embedding_head\n    task: classification\n    data: {csv.name}\n"
            "actions:\n  encode:\n    input: sequence\n"
        )
        with pytest.raises(RecipeError, match="predict"):
            load_recipe(path)

    def test_bundle_copies_artifact(self, recipe_tree: Path, home_root: Path, monkeypatch):
        head = home_root / "head.joblib"
        head.write_bytes(b"fake-model")
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_b"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {"run_id": run_id, "status": "succeeded"},
        )
        pkg = build_model(recipe_tree, bundle=True, artifact=str(head))
        art = pkg.manifest["layers"][0]["artifact"]
        assert art["load"] == "preload"
        assert Path(art["path"]).is_file()
        assert Path(art["path"]).read_bytes() == b"fake-model"
        assert (pkg.path / "artifacts" / "head.joblib").is_file()

    def test_bundle_requires_artifact(self, recipe_tree: Path, home_root: Path, monkeypatch):
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_b"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {"run_id": run_id, "status": "succeeded"},
        )
        with pytest.raises(BuildError, match="bundle"):
            build_model(recipe_tree, bundle=True)

    def test_overwrite_latest(self, recipe_tree: Path, home_root: Path, monkeypatch):
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_1"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {"run_id": run_id, "status": "succeeded"},
        )
        build_model(recipe_tree)
        pkg_dir = home_root / ".biolm" / "models" / "antibody-binder-clf" / "latest"
        (pkg_dir / "stale.txt").write_text("old")
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_2"},
        )
        pkg = build_model(recipe_tree)
        assert pkg.manifest["layers"][0]["run_id"] == "ALY_2"
        assert not (pkg_dir / "stale.txt").exists()
        assert (pkg_dir / BIOLM_MANIFEST).is_file()

    def test_failed_wait_no_package(self, recipe_tree: Path, home_root: Path, monkeypatch):
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_fail"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {"run_id": run_id, "status": "failed"},
        )
        with pytest.raises(BuildError, match="failed"):
            build_model(recipe_tree)
        pkg_dir = home_root / ".biolm" / "models" / "antibody-binder-clf" / "latest"
        assert not (pkg_dir / BIOLM_MANIFEST).exists()

    def test_name_override(self, recipe_tree: Path, home_root: Path, monkeypatch):
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.xgboost",
            lambda **kw: {"run_id": "ALY_n"},
        )
        monkeypatch.setattr(
            "biolm.models.definition.Finetune.wait",
            lambda run_id, **kw: {"run_id": run_id, "status": "succeeded"},
        )
        pkg = build_model(recipe_tree, name="custom-name", tag="v1")
        assert pkg.path == home_root / ".biolm" / "models" / "custom-name" / "v1"
        assert pkg.manifest["name"] == "custom-name"
