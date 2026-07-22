"""BioLM definition recipes and local package builds."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import yaml

from biolm.finetune import Finetune
from biolm.models.errors import BuildError, RecipeError
from biolm.models.paths import user_models_dir

BIOLM_MANIFEST = "BioLM"
SCHEMA_VERSION = 1
ALLOWED_TASKS = frozenset({"classification", "regression"})
REQUIRED_ACTIONS = frozenset({"encode", "predict"})
PathLike = Union[str, Path]


@dataclass
class BuiltPackage:
    """Result of a successful ``build_model``."""

    path: Path
    manifest: Dict[str, Any]


def extract_artifact_uri(payload: Any) -> Optional[str]:
    """Best-effort artifact URI from a Finetune wait/get_run payload."""
    if not isinstance(payload, dict):
        return None

    def _from_value(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for key in ("uri", "url", "path"):
                cand = value.get(key)
                if isinstance(cand, str) and cand.strip():
                    return cand.strip()
        return None

    for key in ("artifact", "artifact_uri", "model_uri"):
        found = _from_value(payload.get(key))
        if found:
            return found

    results = payload.get("results")
    if isinstance(results, dict):
        found = _from_value(results.get("artifact"))
        if found:
            return found

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        found = _from_value(artifacts[0])
        if found:
            return found

    return None


def download_artifact(uri: str, dest: Path) -> Path:
    """Copy or download ``uri`` into ``dest`` (file path). Returns ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(uri)

    if parsed.scheme in ("", "file"):
        src = Path(parsed.path if parsed.scheme == "file" else uri).expanduser()
        if not src.is_file():
            src = Path(uri).expanduser()
        if not src.is_file():
            raise BuildError(f"Artifact file not found: {uri}")
        shutil.copy2(src, dest)
        return dest

    if parsed.scheme in ("http", "https"):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise BuildError("httpx is required to download HTTP artifacts") from exc
        try:
            with httpx.stream("GET", uri, follow_redirects=True, timeout=120.0) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        except Exception as exc:
            raise BuildError(f"Failed to download artifact from {uri!r}: {exc}") from exc
        return dest

    raise BuildError(f"Unsupported artifact URI scheme: {parsed.scheme!r} ({uri})")


def _validate_recipe_actions(raw_actions: Any) -> Optional[Dict[str, Any]]:
    if raw_actions is None:
        return None
    if not isinstance(raw_actions, dict):
        raise RecipeError("'actions' must be a mapping when set")
    missing = REQUIRED_ACTIONS - set(raw_actions.keys())
    if missing:
        raise RecipeError(
            f"Recipe 'actions' must include {sorted(REQUIRED_ACTIONS)} "
            f"(missing {sorted(missing)})"
        )
    return dict(raw_actions)


def _default_actions(task: str, recipe_actions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    actions: Dict[str, Any] = {
        "encode": {"input": "sequence", "schema": "biolm.encode.v1"},
        "predict": {
            "input": "sequence",
            "task": task,
            "schema": "biolm.predict.v1",
        },
    }
    if recipe_actions:
        for name, spec in recipe_actions.items():
            if isinstance(spec, dict):
                merged = dict(actions.get(name, {}))
                merged.update(spec)
                actions[name] = merged
            else:
                actions[name] = spec
        if isinstance(actions.get("encode"), dict):
            actions["encode"].setdefault("schema", "biolm.encode.v1")
            actions["encode"].setdefault("input", "sequence")
        if isinstance(actions.get("predict"), dict):
            actions["predict"].setdefault("schema", "biolm.predict.v1")
            actions["predict"].setdefault("input", "sequence")
            actions["predict"].setdefault("task", task)
    return actions


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

    recipe_actions = _validate_recipe_actions(raw.get("actions"))

    return {
        "schema_version": schema_version,
        "name": name.strip(),
        "description": description,
        "from": from_slug.strip(),
        "recipe_path": recipe_path,
        "actions": recipe_actions,
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


def _artifact_basename(uri: str) -> str:
    parsed = urlparse(uri)
    path = parsed.path if parsed.scheme else uri
    name = Path(path).name
    return name or "model.artifact"


def build_model(
    recipe_path: PathLike,
    *,
    tag: str = "latest",
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    poll_interval: float = 15.0,
    timeout: Optional[float] = None,
    bundle: bool = False,
    artifact: Optional[str] = None,
) -> BuiltPackage:
    """Compile a recipe into a locked ``BioLM`` package under ``~/.biolm/models``.

    Does not modify the recipe file. Overwrites an existing package dir for the
    same ``name``/``tag``.

    When ``bundle=True``, requires a resolvable head artifact URI (from the
    finetune result or ``artifact=`` override), downloads into
    ``artifacts/``, and sets ``artifact.path`` with ``load: preload``.
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

    artifact_uri = None
    if artifact:
        artifact_uri = str(artifact).strip() or None
    if not artifact_uri and isinstance(result, dict):
        artifact_uri = extract_artifact_uri(result)

    artifact_block: Dict[str, Any] = {"load": "preload"}
    if artifact_uri:
        artifact_block["uri"] = artifact_uri

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    package_layer: Dict[str, Any] = {
        "type": "embedding_head",
        "task": layer["task"],
        "data": {"path": str(data_path.resolve())},
        "embedding_models": list(layer["embedding_models"]),
        "target_column": layer["target_column"],
        "text_column": layer["text_column"],
        "run_id": str(run_id),
        "artifact": artifact_block,
    }
    if isinstance(result, dict) and "metrics" in result:
        package_layer["metrics"] = result["metrics"]

    manifest: Dict[str, Any] = {
        "schema_version": recipe["schema_version"],
        "name": model_name,
        "tag": tag,
        "from": {"slug": recipe["from"], "load": "lazy"},
        "layers": [package_layer],
        "actions": _default_actions(layer["task"], recipe.get("actions")),
        "built": {
            "at": built_at,
            "status": "locked",
            "recipe_path": str(recipe["recipe_path"]),
        },
    }
    if recipe.get("description"):
        manifest["description"] = recipe["description"]

    if bundle and not artifact_uri:
        raise BuildError(
            "bundle=True requires a head artifact URI from the finetune result "
            "or an explicit artifact= PATH|URL override"
        )

    pkg_dir = user_models_dir() / model_name / tag
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    if bundle and artifact_uri:
        arts = pkg_dir / "artifacts"
        arts.mkdir(parents=True, exist_ok=True)
        dest = arts / _artifact_basename(artifact_uri)
        download_artifact(artifact_uri, dest)
        package_layer["artifact"] = {
            "load": "preload",
            "uri": artifact_uri,
            "path": str(dest.resolve()),
        }

    manifest_path = pkg_dir / BIOLM_MANIFEST
    manifest_path.write_text(
        yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return BuiltPackage(path=pkg_dir, manifest=manifest)


def resolve_package(name_or_path: str, tag: Optional[str] = None) -> Path:
    """Resolve ``name``, ``name:tag``, or a package directory path to the package dir."""
    raw = str(name_or_path).strip()
    path = Path(raw).expanduser()
    if path.is_dir() and (path / BIOLM_MANIFEST).is_file():
        return path.resolve()
    if path.is_file() and path.name == BIOLM_MANIFEST:
        return path.parent.resolve()

    pkg_name = raw
    pkg_tag = tag or "latest"
    if tag is None and ":" in raw and not raw.startswith("."):
        pkg_name, _, pkg_tag = raw.partition(":")
    pkg_dir = user_models_dir() / pkg_name / pkg_tag
    if not (pkg_dir / BIOLM_MANIFEST).is_file():
        raise BuildError(f"BioLM package not found: {pkg_dir}")
    return pkg_dir.resolve()
