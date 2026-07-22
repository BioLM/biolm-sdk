"""Tests for biolm model build CLI."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from biolm.cli import cli
from biolm.models.definition import BIOLM_MANIFEST, BuiltPackage
from biolm.models.errors import RecipeError


MINIMAL_RECIPE = """\
schema_version: 1
name: antibody-binder-clf
from: esm2-8m
layers:
  - type: embedding_head
    task: classification
    data: ./data/binders.csv
"""


def _write_recipe(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "binders.csv").write_text("sequence,label\nMKT,1\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(MINIMAL_RECIPE)
    return recipe


class TestModelBuildCli:
    def test_build_success(self, tmp_path: Path):
        recipe = _write_recipe(tmp_path)
        pkg_path = tmp_path / "pkg"
        pkg_path.mkdir()
        (pkg_path / BIOLM_MANIFEST).write_text("name: x\n")
        fake = BuiltPackage(
            path=pkg_path,
            manifest={"name": "antibody-binder-clf", "tag": "v1"},
        )

        with patch("biolm.models.definition.build_model", return_value=fake) as mock_build:
            runner = CliRunner()
            result = runner.invoke(cli, ["model", "build", str(recipe), "--tag", "v1"])

        assert result.exit_code == 0, result.output
        mock_build.assert_called_once()
        kwargs = mock_build.call_args
        assert kwargs[0][0] == str(recipe) or Path(kwargs[0][0]) == recipe
        assert kwargs[1]["tag"] == "v1"
        assert "Built package" in result.output
        assert "antibody-binder-clf:v1" in result.output
        assert "Model Build Complete" in result.output
    def test_build_recipe_error(self, tmp_path: Path):
        recipe = _write_recipe(tmp_path)
        with patch(
            "biolm.models.definition.build_model",
            side_effect=RecipeError("bad recipe"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["model", "build", str(recipe)])

        assert result.exit_code == 1
        assert "bad recipe" in result.output
        assert "Model Build Failed" in result.output
