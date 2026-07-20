"""Load and resolve ``lltp.yaml`` project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

DEFAULT_CONFIG_NAME = "lltp.yaml"

EXAMPLE_LLTP_YAML = """\
# LLTP project config for biolm-sdk
version: 1

default_connector: adaptyv

connectors:
  adaptyv:
    auth:
      token_env: ADAPTYV_API_TOKEN
    defaults:
      service_id: adaptyv-lltp.expression-v1
  twist:
    auth:
      token_env: TWIST_END_USER_TOKEN
      jwt_env: TWIST_STAGING_JWT_TOKEN
      email_env: TWIST_STAGING_USER_EMAIL
    defaults:
      service_id: twist-lltp.dna-synthesis-v1
      wait_for_scoring: false

experiments:
  express:
    connector: adaptyv
    service_id: adaptyv-lltp.expression-v1
  synthesize:
    connector: twist
    service_id: twist-lltp.dna-synthesis-v1
    wait_for_scoring: false
"""


@dataclass
class ConnectorConfig:
    name: str
    auth: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    connector: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabConfig:
    path: Optional[Path]
    version: int = 1
    default_connector: Optional[str] = None
    connectors: Dict[str, ConnectorConfig] = field(default_factory=dict)
    experiments: Dict[str, ExperimentConfig] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def get_experiment(self, name: str) -> ExperimentConfig:
        if name not in self.experiments:
            known = ", ".join(sorted(self.experiments)) or "(none)"
            raise KeyError(
                f"Unknown experiment {name!r}. Known experiments: {known}"
            )
        return self.experiments[name]

    def get_connector(self, name: str) -> ConnectorConfig:
        if name not in self.connectors:
            known = ", ".join(sorted(self.connectors)) or "(none)"
            raise KeyError(
                f"Unknown connector {name!r}. Known connectors: {known}"
            )
        return self.connectors[name]


def find_config_path(
    start: Optional[Path] = None,
    *,
    filename: str = DEFAULT_CONFIG_NAME,
) -> Optional[Path]:
    """Search ``start`` and parents for ``lltp.yaml``."""
    cur = (start or Path.cwd()).resolve()
    for directory in [cur, *cur.parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        # Stop at filesystem root
        if directory.parent == directory:
            break
    return None


def load_config(
    path: Optional[Path] = None,
    *,
    start: Optional[Path] = None,
) -> LabConfig:
    """Load ``lltp.yaml`` from ``path`` or by searching upward from ``start``/cwd.

    Returns an empty :class:`LabConfig` when no file is found (callers may still
    submit with explicit ``connector`` / ``service_id``).
    """
    cfg_path = Path(path).resolve() if path else find_config_path(start)
    if cfg_path is None:
        return LabConfig(path=None)

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: expected a YAML mapping at the top level")

    connectors: Dict[str, ConnectorConfig] = {}
    for name, body in (raw.get("connectors") or {}).items():
        body = body or {}
        if not isinstance(body, dict):
            raise ValueError(f"{cfg_path}: connectors.{name} must be a mapping")
        connectors[str(name)] = ConnectorConfig(
            name=str(name),
            auth=dict(body.get("auth") or {}),
            defaults=dict(body.get("defaults") or {}),
        )

    experiments: Dict[str, ExperimentConfig] = {}
    for name, body in (raw.get("experiments") or {}).items():
        body = body or {}
        if not isinstance(body, dict):
            raise ValueError(f"{cfg_path}: experiments.{name} must be a mapping")
        connector = body.get("connector")
        if not connector:
            raise ValueError(
                f"{cfg_path}: experiments.{name} requires connector"
            )
        params = {k: v for k, v in body.items() if k != "connector"}
        experiments[str(name)] = ExperimentConfig(
            name=str(name),
            connector=str(connector),
            params=params,
        )

    return LabConfig(
        path=cfg_path,
        version=int(raw.get("version") or 1),
        default_connector=(
            str(raw["default_connector"])
            if raw.get("default_connector") is not None
            else None
        ),
        connectors=connectors,
        experiments=experiments,
        raw=raw,
    )


def resolve_auth(
    auth: Mapping[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve connector auth from yaml + environment.

    Supported keys:
      - ``token`` / ``token_env`` — API token (env wins over inline ``token``)
      - ``jwt`` / ``jwt_env`` — Twist JWT
      - ``email`` / ``email_env`` — Twist user email
      - ``end_user_token`` / ``end_user_token_env`` — Twist end-user token

    Environment values always win when the corresponding ``*_env`` var is set.
    """
    env = environ if environ is not None else os.environ
    out: Dict[str, Any] = {}

    def _pick(
        inline_key: str,
        env_key_name: str,
        out_key: str,
        default_env: Optional[str] = None,
    ) -> None:
        env_var = auth.get(env_key_name) or default_env
        if env_var and env.get(str(env_var)):
            out[out_key] = env[str(env_var)]
            return
        if auth.get(inline_key) is not None:
            out[out_key] = auth[inline_key]

    _pick("token", "token_env", "token")
    _pick("jwt", "jwt_env", "jwt")
    _pick("email", "email_env", "email")
    _pick("end_user_token", "end_user_token_env", "end_user_token")

    # Common aliases: token_env alone is enough for Adaptyv
    if "token" not in out and auth.get("token_env"):
        val = env.get(str(auth["token_env"]))
        if val:
            out["token"] = val

    return out


def write_example_config(path: Path, *, force: bool = False) -> Path:
    """Write a starter ``lltp.yaml``."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_LLTP_YAML, encoding="utf-8")
    return path
