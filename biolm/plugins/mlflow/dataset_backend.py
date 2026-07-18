"""MLflow push/pull backend for local datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from biolm.datasets.errors import BackendNotAvailableError, DatasetError
from biolm.datasets.schema import (
    DATASET_YAML,
    build_meta,
    load_dataset_yaml,
    write_dataset_yaml,
)

if TYPE_CHECKING:
    from biolm.datasets.dataset import Dataset


class MLflowDatasetBackend:
    """Push/pull local datasets via MLflow runs tagged as datasets."""

    name = "mlflow"

    def push(self, dataset: "Dataset", **opts: Any) -> Dict[str, Any]:
        from biolm.plugins.mlflow.protocols import MLflowNotAvailableError
        from biolm.plugins.mlflow.datasets import (
            upload_dataset,
            _check_mlflow_available,
            _get_default_experiment_name,
            _get_mlflow_client,
            _get_or_create_experiment,
        )

        try:
            _check_mlflow_available()
        except MLflowNotAvailableError as exc:
            raise BackendNotAvailableError(
                "MLflow dataset backend is not available. "
                "Install with: pip install 'biolm-sdk[mlflow]'"
            ) from exc

        mlflow_uri = opts.get("mlflow_uri")
        experiment_name = opts.get("experiment_name")

        # Upload entire dataset directory (includes dataset.yaml + data/)
        result = upload_dataset(
            dataset_id=dataset.id,
            file_path=dataset.path,
            experiment_name=experiment_name,
            name=dataset.id,
            mlflow_uri=mlflow_uri,
            recursive=True,
        )

        # Enrich tags with local metadata for round-trip
        try:
            import mlflow

            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            client = _get_mlflow_client(mlflow_uri)
            exp_name = experiment_name or _get_default_experiment_name()
            experiment_id = _get_or_create_experiment(client, exp_name)
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=(
                    f"tags.dataset_id = '{dataset.id}' AND tags.type = 'dataset'"
                ),
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                tags = {
                    "type": "dataset",
                    "dataset_id": dataset.id,
                    "biolm.dataset_type": dataset.type,
                }
                if dataset.description:
                    tags["biolm.description"] = dataset.description
                if dataset.tags:
                    tags["biolm.tags"] = json.dumps(dataset.tags)
                if dataset.attrs:
                    tags["biolm.attrs"] = json.dumps(dataset.attrs)
                for key, value in tags.items():
                    client.set_tag(run_id, key, value)
                result["run_id"] = run_id
        except Exception:
            # Upload already succeeded; tag enrichment is best-effort
            pass

        result["backend"] = self.name
        return result

    def pull(
        self,
        dataset_id: str,
        dest: Path,
        **opts: Any,
    ) -> Dict[str, Any]:
        from biolm.plugins.mlflow.protocols import MLflowNotAvailableError
        from biolm.plugins.mlflow.datasets import (
            download_dataset,
            get_dataset,
            _check_mlflow_available,
        )

        try:
            _check_mlflow_available()
        except MLflowNotAvailableError as exc:
            raise BackendNotAvailableError(
                "MLflow dataset backend is not available. "
                "Install with: pip install 'biolm-sdk[mlflow]'"
            ) from exc

        mlflow_uri = opts.get("mlflow_uri")
        experiment_name = opts.get("experiment_name")
        force = bool(opts.get("force", False))

        remote = get_dataset(
            dataset_id,
            experiment_name=experiment_name,
            mlflow_uri=mlflow_uri,
        )
        if not remote:
            raise DatasetError(f"Remote dataset '{dataset_id}' not found (backend=mlflow)")

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        if (dest / DATASET_YAML).is_file() and not force:
            existing = load_dataset_yaml(dest)
            if existing.id != dataset_id:
                raise DatasetError(
                    f"Destination {dest} already has dataset id '{existing.id}'"
                )

        result = download_dataset(
            dataset_id=dataset_id,
            output_path=dest,
            experiment_name=experiment_name,
            mlflow_uri=mlflow_uri,
        )

        # MLflow may nest artifacts; prefer an existing dataset.yaml if downloaded
        yaml_candidates = list(dest.rglob(DATASET_YAML))
        if yaml_candidates:
            # If yaml landed in a subdirectory, promote dataset to dest root when needed
            chosen = yaml_candidates[0]
            if chosen.parent.resolve() != dest.resolve():
                # Move contents of chosen.parent up if dest only has that subtree
                pass
            try:
                meta = load_dataset_yaml(chosen)
                if chosen.parent.resolve() != dest.resolve():
                    write_dataset_yaml(dest, meta)
            except Exception:
                self._synthesize_yaml(dest, dataset_id, remote)
        else:
            self._synthesize_yaml(dest, dataset_id, remote)

        result["backend"] = self.name
        result["path"] = str(dest)
        return result

    @staticmethod
    def _synthesize_yaml(dest: Path, dataset_id: str, remote: Dict[str, Any]) -> None:
        tags = remote.get("tags") or {}
        dtype = tags.get("biolm.dataset_type") or "files"
        description = tags.get("biolm.description")
        tag_list = []
        attrs: Dict[str, Any] = {}
        if tags.get("biolm.tags"):
            try:
                tag_list = json.loads(tags["biolm.tags"])
            except (TypeError, json.JSONDecodeError):
                tag_list = []
        if tags.get("biolm.attrs"):
            try:
                attrs = json.loads(tags["biolm.attrs"])
            except (TypeError, json.JSONDecodeError):
                attrs = {}
        meta = build_meta(
            dataset_id,
            description=description,
            type=dtype,
            tags=tag_list if isinstance(tag_list, list) else [],
            attrs=attrs if isinstance(attrs, dict) else {},
        )
        write_dataset_yaml(dest, meta)
