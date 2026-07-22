"""BioLM definition recipes and local package builds."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from biolm.finetune import Finetune
from biolm.models.errors import BuildError, RecipeError
from biolm.models.paths import user_models_dir

BIOLM_MANIFEST = "BioLM"
SCHEMA_VERSION = 1
ALLOWED_TASKS = frozenset({"classification", "regression"})
PathLike = Union[str, Path]


@dataclass
class BuiltPackage:
    """Result of a successful ``build_model``."""

    path: Path
    manifest: Dict[str, Any]


def load_recipe(path: PathLike) -> Dict[str, Any]:
    """Load and validate a BioLM definition recipe YAML.

    Returns a normalized dict. The single layer includes ``data_path`` as an
    absolute :class:`~pathlib.Path` to the training CSV.
    """
    recipe_path = Path(path).expanduser().resolve()
    if not recipe_path.is_file():
        raise RecipeError(f"Recipe not found: {recipe_path}")

    try:
        raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RecipeError(f"Invalid YAML in {recipe_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RecipeError("Recipe must be a YAML mapping")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RecipeError("Recipe requires a non-empty 'name'")

    from_slug = raw.get("from")
    if not isinstance(from_slug, str) or not from_slug.strip():
        raise RecipeError("Recipe requires a non-empty 'from' (base model slug)")

    layers = raw.get("layers")
    if not isinstance(layers, list) or len(layers) != 1:
        raise RecipeError("v0 recipes require exactly one layer")

    layer = layers[0]
    if not isinstance(layer, dict):
        raise RecipeError("Layer must be a mapping")

    layer_type = layer.get("type")
    if layer_type != "embedding_head":
        raise RecipeError(
            "v0 only supports layer type 'embedding_head' "
            f"(got {layer_type!r})"
        )

    task = layer.get("task", "classification")
    if task not in ALLOWED_TASKS:
        raise RecipeError(
            f"Layer 'task' must be one of {sorted(ALLOWED_TASKS)} (got {task!r})"
        )

    data = layer.get("data")
    if data is None or (isinstance(data, str) and not data.strip()):
        raise RecipeError("Layer requires 'data' (local CSV path)")
    if not isinstance(data, str):
        raise RecipeError("v0 layer 'data' must be a local filesystem path string")

    data_path = Path(data).expanduser()
    if not data_path.is_absolute():
        data_path = (recipe_path.parent / data_path).resolve()
    else:
        data_path = data_path.resolve()
    if not data_path.is_file():
        raise RecipeError(f"Training data file not found: {data_path}")

    schema_version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema_version, int):
        raise RecipeError("'schema_version' must be an integer")

    embedding_models = layer.get("embedding_models")
    if embedding_models is None:
        embedding_models = [from_slug.strip()]
    elif not isinstance(embedding_models, list) or not all(
        isinstance(m, str) and m.strip() for m in embedding_models
    ):
        raise RecipeError("'embedding_models' must be a list of non-empty strings")

    target_column = layer.get("target_column", "label")
    text_column = layer.get("text_column", "sequence")
    if not isinstance(target_column, str) or not target_column.strip():
        raise RecipeError("'target_column' must be a non-empty string")
    if not isinstance(text_column, str) or not text_column.strip():
        raise RecipeError("'text_column' must be a non-empty string")

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise RecipeError("'description' must be a string when set")

    return {
        "schema_version": schema_version,
        "name": name.strip(),
        "description": description,
        "from": from_slug.strip(),
        "recipe_path": recipe_path,
        "layers": [
            {
                "type": "embedding_head",
                "task": task,
                "data": data,
                "data_path": data_path,
                "embedding_models": list(embedding_models),
                "target_column": target_column.strip(),
                "text_column": text_column.strip(),
            }
        ],
    }


def build_model(
    recipe_path: PathLike,
    *,
    tag: str = "latest",
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    poll_interval: float = 15.0,
    timeout: Optional[float] = None,
) -> BuiltPackage:
    """Compile a recipe into a locked ``BioLM`` package under ``~/.biolm/models``.

    Does not modify the recipe file. Overwrites an existing package dir for the
    same ``name``/``tag``.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise BuildError("tag must be a non-empty string")
    tag = tag.strip()

    recipe = load_recipe(recipe_path)
    model_name = (name or recipe["name"]).strip()
    if not model_name:
        raise BuildError("model name must be non-empty")

    layer = recipe["layers"][0]
    data_path: Path = layer["data_path"]
    train_data = data_path.read_text(encoding="utf-8")

    create_kwargs: Dict[str, Any] = {
        "train_data": train_data,
        "embedding_models": layer["embedding_models"],
        "task_type": layer["task"],
        "target_column": layer["target_column"],
        "text_column": layer["text_column"],
        "run_name": model_name,
    }
    if api_key is not None:
        create_kwargs["api_key"] = api_key

    try:
        created = Finetune.xgboost(**create_kwargs)
    except Exception as exc:
        raise BuildError(f"Failed to launch finetune: {exc}") from exc

    run_id = created.get("run_id") if isinstance(created, dict) else None
    if not run_id:
        raise BuildError(f"Finetune.xgboost did not return run_id: {created!r}")

    wait_kwargs: Dict[str, Any] = {"poll_interval": poll_interval, "timeout": timeout}
    if api_key is not None:
        wait_kwargs["api_key"] = api_key

    try:
        result = Finetune.wait(str(run_id), **wait_kwargs)
    except Exception as exc:
        raise BuildError(f"Finetune.wait failed: {exc}") from exc

    status = result.get("status") if isinstance(result, dict) else None
    if status != "succeeded":
        raise BuildError(
            f"Finetune run {run_id} ended with status {status!r} (expected 'succeeded')"
        )

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    package_layer: Dict[str, Any] = {
        "type": "embedding_head",
        "task": layer["task"],
        "data": {"path": str(data_path.resolve())},
        "embedding_models": list(layer["embedding_models"]),
        "target_column": layer["target_column"],
        "text_column": layer["text_column"],
        "run_id": str(run_id),
    }
    if isinstance(result, dict):
        if "artifact" in result:
            package_layer["artifact"] = result["artifact"]
        if "metrics" in result:
            package_layer["metrics"] = result["metrics"]

    manifest: Dict[str, Any] = {
        "schema_version": recipe["schema_version"],
        "name": model_name,
        "tag": tag,
        "from": {"slug": recipe["from"]},
        "layers": [package_layer],
        "actions": {
            "encode": {"input": "sequence"},
            "predict": {"input": "sequence", "task": layer["task"]},
        },
        "built": {
            "at": built_at,
            "status": "locked",
            "recipe_path": str(recipe["recipe_path"]),
        },
    }
    if recipe.get("description"):
        manifest["description"] = recipe["description"]

    pkg_dir = user_models_dir() / model_name / tag
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pkg_dir / BIOLM_MANIFEST
    manifest_path.write_text(
        yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return BuiltPackage(path=pkg_dir, manifest=manifest)
