"""Python API for LLTP lab submit / status / confirm / results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from biolm.lab.config import LabConfig, load_config
from biolm.lab.connectors import build_client, hydrate_vendor_session
from biolm.lab.runs import (
    LabRun,
    list_runs as _list_runs_disk,
    load_run,
    new_run_id,
    save_run,
)

# Payload keys that are not forwarded into to_lltp extras from experiment params
_META_KEYS = frozenset({"connector", "service_id"})


def _coarse_status_from_patches(patches: List[Mapping[str, Any]]) -> str:
    """Derive a coarse run status from LLTP requirement patches."""
    if not patches:
        return "in_progress"
    statuses = [str(p.get("status") or "").upper() for p in patches]
    if any(s == "REJECTED" for s in statuses):
        return "failed"
    if any(s == "CANCELLED" or s == "CANCELED" for s in statuses):
        return "cancelled"
    # BLOCKED / AWAITING on APPROVAL-style requirements
    awaiting = [
        p
        for p in patches
        if str(p.get("status") or "").upper() == "AWAITING"
    ]
    if awaiting:
        types = {str(p.get("type") or "").upper() for p in awaiting}
        if "APPROVAL" in types or any(
            "quote" in str(p.get("requirement_id") or "").lower() for p in awaiting
        ):
            return "blocked"
        return "in_progress"
    if statuses and all(s == "FULFILLED" for s in statuses):
        return "completed"
    return "in_progress"


def _resolve_submit_params(
    config: LabConfig,
    *,
    experiment: Optional[str],
    connector: Optional[str],
    service_id: Optional[str],
    payload_extras: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge experiment / connector defaults / explicit kwargs into submit params."""
    params: Dict[str, Any] = {}
    conn_name = connector
    exp_name = experiment

    if exp_name:
        exp = config.get_experiment(exp_name)
        conn_name = conn_name or exp.connector
        params.update(exp.params)

    if not conn_name:
        conn_name = config.default_connector
    if not conn_name:
        raise ValueError(
            "connector is required (pass connector=, experiment=, or set "
            "default_connector in lltp.yaml)"
        )

    conn_cfg = None
    if conn_name in config.connectors:
        conn_cfg = config.get_connector(conn_name)
        # connector defaults first; experiment params already applied above
        merged = dict(conn_cfg.defaults)
        merged.update(params)
        params = merged
    elif config.connectors:
        # Allow unknown-to-yaml connector names if built-in (adaptyv/twist)
        pass

    if service_id:
        params["service_id"] = service_id
    if "service_id" not in params:
        raise ValueError(
            "service_id is required (pass service_id=, set it on the experiment, "
            "or under connectors.<name>.defaults)"
        )

    # Explicit extras win
    params.update(dict(payload_extras))

    # Avoid hidden blocking on Twist scoring unless the user opted in
    if conn_name.strip().lower() == "twist" and "wait_for_scoring" not in params:
        params["wait_for_scoring"] = False

    return {
        "connector": conn_name.strip().lower(),
        "experiment": exp_name,
        "connector_cfg": conn_cfg,
        "params": params,
    }


def submit(
    sf: Any,
    *,
    experiment: Optional[str] = None,
    connector: Optional[str] = None,
    service_id: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[LabConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
    client: Any = None,
    **payload_extras: Any,
) -> LabRun:
    """Submit a SeqFrame to a lab connector; persist a run under ``.biolm/lltp/``.

    Provide ``experiment=`` (from ``lltp.yaml``) and/or ``connector=`` + ``service_id=``.
    """
    cfg = config or load_config(Path(config_path) if config_path else None)
    resolved = _resolve_submit_params(
        cfg,
        experiment=experiment,
        connector=connector,
        service_id=service_id,
        payload_extras=payload_extras,
    )
    conn_name = resolved["connector"]
    params = dict(resolved["params"])
    svc = str(params.pop("service_id"))

    to_kwargs = {k: v for k, v in params.items() if k not in _META_KEYS}
    if name is not None:
        to_kwargs["name"] = name

    payload = sf.lab.to_lltp(service_id=svc, **to_kwargs)

    vendor = client or build_client(conn_name, resolved["connector_cfg"])
    handle = vendor.submit(payload)
    if not isinstance(handle, dict):
        handle = {"raw": handle}
    else:
        handle = dict(handle)

    # Persist fields needed to rehydrate Twist sessions / confirm_quote across processes
    for key in (
        "payment_method_id",
        "shipping_address_id",
        "email",
        "wait_for_scoring",
    ):
        if key in payload and key not in handle:
            handle[key] = payload[key]

    run = LabRun(
        run_id=new_run_id(),
        connector=conn_name,
        service_id=svc,
        experiment=resolved["experiment"],
        status="submitted",
        handle=handle,
        extras={"payload_keys": sorted(payload.keys())},
    )
    save_run(run, root=root)
    return run


def status(
    run_id: str,
    *,
    config: Optional[LabConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
    client: Any = None,
) -> LabRun:
    """Poll the connector and update the run file (on-demand; no blocking wait)."""
    cfg = config or load_config(Path(config_path) if config_path else None)
    run = load_run(run_id, root=root)
    conn_cfg = cfg.connectors.get(run.connector)
    vendor = client or build_client(run.connector, conn_cfg)
    hydrate_vendor_session(vendor, run.handle)

    vendor_id = (
        run.handle.get("vendor_order_id")
        or run.handle.get("order_id")
        or run.handle.get("quote_id")
        or run.handle.get("experiment_id")
    )
    if not vendor_id:
        raise ValueError(
            f"Run {run_id!r} handle is missing vendor_order_id / experiment_id"
        )

    snapshot = vendor.poll(str(vendor_id))
    patches = vendor.map_status(snapshot) if hasattr(vendor, "map_status") else []
    run.last_status = {
        "snapshot": snapshot,
        "patches": patches,
    }
    run.status = _coarse_status_from_patches(list(patches or []))
    save_run(run, root=root)
    return run


def confirm(
    run_id: str,
    *,
    config: Optional[LabConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
    client: Any = None,
    **kwargs: Any,
) -> LabRun:
    """Confirm quote / approval on a blocked run (connector-specific)."""
    cfg = config or load_config(Path(config_path) if config_path else None)
    run = load_run(run_id, root=root)
    conn_cfg = cfg.connectors.get(run.connector)
    vendor = client or build_client(run.connector, conn_cfg)
    hydrate_vendor_session(vendor, run.handle)

    quote_or_exp = (
        run.handle.get("quote_id")
        or run.handle.get("experiment_id")
        or run.handle.get("vendor_order_id")
    )
    if not quote_or_exp:
        raise ValueError(f"Run {run_id!r} handle missing quote_id / experiment_id")

    if not hasattr(vendor, "confirm_quote"):
        raise TypeError(
            f"Connector {run.connector!r} does not support confirm_quote"
        )

    confirm_kwargs = dict(kwargs)
    for key in ("payment_method_id", "shipping_address_id", "email"):
        if key not in confirm_kwargs and run.handle.get(key) is not None:
            confirm_kwargs[key] = run.handle[key]

    new_handle = vendor.confirm_quote(str(quote_or_exp), **confirm_kwargs)
    if isinstance(new_handle, dict):
        merged = dict(run.handle)
        merged.update(new_handle)
        run.handle = merged
    run.status = "in_progress"
    save_run(run, root=root)
    return run


def results(
    run_id: str,
    *,
    config: Optional[LabConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
    client: Any = None,
) -> Any:
    """Fetch vendor results, convert via ``SeqFrame.lab.from_lltp``, return SeqFrame."""
    from biolm.seqframe.namespaces.lab import LabNamespace

    cfg = config or load_config(Path(config_path) if config_path else None)
    run = load_run(run_id, root=root)
    conn_cfg = cfg.connectors.get(run.connector)
    vendor = client or build_client(run.connector, conn_cfg)
    hydrate_vendor_session(vendor, run.handle)

    vendor_id = (
        run.handle.get("order_id")
        or run.handle.get("experiment_id")
        or run.handle.get("vendor_order_id")
    )
    if not vendor_id:
        raise ValueError(f"Run {run_id!r} handle missing order/experiment id")

    raw = vendor.fetch_results(str(vendor_id))
    order_id = str(
        run.handle.get("order_id")
        or run.handle.get("vendor_order_id")
        or vendor_id
    )

    if hasattr(vendor, "to_result"):
        dataset = vendor.to_result(
            raw, order_id=order_id, service_id=run.service_id
        )
    else:
        # Adaptyv (and similar): module-level to_lltp_result(experiment, results)
        to_lltp_result = None
        try:
            from adaptyv_lltp import to_lltp_result as _fn

            to_lltp_result = _fn
        except ImportError:
            to_lltp_result = getattr(vendor, "to_lltp_result", None)

        if callable(to_lltp_result):
            experiment = {}
            if run.last_status and isinstance(run.last_status.get("snapshot"), dict):
                experiment = run.last_status["snapshot"]
            elif isinstance(run.handle.get("detail"), dict):
                experiment = run.handle["detail"]
            dataset = to_lltp_result(
                experiment,
                raw,
                order_id=order_id,
                service_id=run.service_id,
            )
        elif isinstance(raw, dict) and "records" in raw:
            dataset = raw
        else:
            dataset = {
                "records": raw if isinstance(raw, list) else [],
                "order_id": order_id,
                "service_id": run.service_id,
            }

    sf_out = LabNamespace.from_lltp(dataset)
    run.result = {
        "dataset_id": dataset.get("dataset_id") if isinstance(dataset, dict) else None,
        "record_count": (
            dataset.get("record_count")
            if isinstance(dataset, dict)
            else None
        )
        or (len(dataset.get("records", [])) if isinstance(dataset, dict) else None),
        "order_id": dataset.get("order_id") if isinstance(dataset, dict) else None,
        "service_id": dataset.get("service_id") if isinstance(dataset, dict) else None,
    }
    run.status = "completed"
    save_run(run, root=root)
    return sf_out


def list_runs(*, root: Optional[Union[str, Path]] = None) -> List[LabRun]:
    """List persisted lab runs under ``.biolm/lltp/``."""
    return _list_runs_disk(root=root)
