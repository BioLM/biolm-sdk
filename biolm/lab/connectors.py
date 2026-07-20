"""Connector registry for LLTP VendorClient façades.

Connectors are soft-imported. Install from GitHub until published on PyPI::

    pip install "adaptyv-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=adaptyv-lltp/src/py"
    pip install "twist-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=twist-lltp/src/py"
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Type

from biolm.lab.config import ConnectorConfig, resolve_auth

# Short install hints shown when import fails.
_INSTALL_HINTS: Dict[str, str] = {
    "adaptyv": (
        'pip install "adaptyv-lltp @ '
        "git+https://github.com/BioLM/lltp-connectors.git"
        '#subdirectory=adaptyv-lltp/src/py"'
    ),
    "twist": (
        'pip install "twist-lltp @ '
        "git+https://github.com/BioLM/lltp-connectors.git"
        '#subdirectory=twist-lltp/src/py"'
    ),
}


def _load_adaptyv_vendor() -> Type[Any]:
    try:
        from adaptyv_lltp import VendorClient
    except ImportError as exc:
        raise ImportError(
            "Connector 'adaptyv' requires the adaptyv-lltp package.\n\n"
            f"Install with:\n\n    {_INSTALL_HINTS['adaptyv']}\n"
        ) from exc
    return VendorClient


def _load_twist_vendor() -> Type[Any]:
    try:
        from twist_lltp import VendorClient
    except ImportError as exc:
        raise ImportError(
            "Connector 'twist' requires the twist-lltp package.\n\n"
            f"Install with:\n\n    {_INSTALL_HINTS['twist']}\n"
        ) from exc
    return VendorClient


_LOADERS: Dict[str, Callable[[], Type[Any]]] = {
    "adaptyv": _load_adaptyv_vendor,
    "twist": _load_twist_vendor,
}


def known_connectors() -> tuple:
    return tuple(sorted(_LOADERS))


def get_vendor_client_class(name: str) -> Type[Any]:
    key = name.strip().lower()
    if key not in _LOADERS:
        known = ", ".join(known_connectors())
        raise KeyError(
            f"Unknown connector {name!r}. Built-in connectors: {known}"
        )
    return _LOADERS[key]()


def build_client(
    name: str,
    connector_cfg: Optional[ConnectorConfig] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    **overrides: Any,
) -> Any:
    """Instantiate a VendorClient for ``name`` using resolved auth.

    ``overrides`` are passed through to the client constructor after auth kwargs.
    """
    cls = get_vendor_client_class(name)
    auth = resolve_auth(
        (connector_cfg.auth if connector_cfg else {}) or {},
        environ=environ,
    )
    kwargs: Dict[str, Any] = dict(overrides)

    key = name.strip().lower()
    if key == "adaptyv":
        if auth.get("token") is not None:
            kwargs.setdefault("api_token", auth["token"])
            kwargs.setdefault("api_key", auth["token"])
    elif key == "twist":
        # TwistClient(api_token=..., jwt_token=..., user_email=...)
        if auth.get("end_user_token") is not None:
            kwargs.setdefault("api_token", auth["end_user_token"])
        elif auth.get("token") is not None:
            kwargs.setdefault("api_token", auth["token"])
        if auth.get("jwt") is not None:
            kwargs.setdefault("jwt_token", auth["jwt"])
        if auth.get("email") is not None:
            kwargs.setdefault("user_email", auth["email"])

    return cls(**kwargs)


def hydrate_vendor_session(client: Any, handle: Mapping[str, Any]) -> None:
    """Rehydrate in-memory connector session from a persisted run handle.

    Twist ``VendorClient`` keeps quote/order metadata in ``_sessions``; CLI/cron
    processes must restore it from ``.biolm/lltp/<run_id>.json``.
    """
    remember = getattr(client, "_remember", None)
    if not callable(remember):
        return

    quote_id = handle.get("quote_id")
    order_id = handle.get("order_id")
    meta = {
        "quote_id": quote_id,
        "order_id": order_id,
        "construct_ids": list(handle.get("construct_ids") or []),
        "external_ids": list(handle.get("external_ids") or []),
        "service_id": handle.get("service_id"),
        "email": handle.get("email"),
        "shipping_address_id": handle.get("shipping_address_id"),
        "payment_method_id": handle.get("payment_method_id"),
        "kind": "order" if order_id else "quote",
    }
    if quote_id:
        remember(str(quote_id), meta)
    if order_id:
        remember(str(order_id), meta)
    vendor_order_id = handle.get("vendor_order_id")
    if vendor_order_id and vendor_order_id not in {quote_id, order_id}:
        remember(str(vendor_order_id), meta)
