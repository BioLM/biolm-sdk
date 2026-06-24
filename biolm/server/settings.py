"""Server configuration from environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8787
    auth_mode: str = "none"  # none | token
    server_token: str = ""
    refresh_seconds: int = 60
    models_env: str = ""
    config_path: str = ""
    modal_environment: str = "main"
    health_check: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "ServerSettings":
        defaults = cls(
            host=os.environ.get("BIOLM_SERVER_HOST", "127.0.0.1"),
            port=int(os.environ.get("BIOLM_SERVER_PORT", "8787")),
            auth_mode=os.environ.get("BIOLM_SERVER_AUTH", "none"),
            server_token=os.environ.get("BIOLM_SERVER_TOKEN", ""),
            refresh_seconds=int(os.environ.get("BIOLM_SERVER_REFRESH_SECONDS", "60")),
            models_env=os.environ.get("BIOLM_SERVER_MODELS", ""),
            config_path=os.environ.get(
                "BIOLM_SERVER_CONFIG_PATH",
                os.path.join(os.path.expanduser("~"), ".biolm", "server.yaml"),
            ),
            modal_environment=os.environ.get("BIOLM_SERVER_MODAL_ENV", "main"),
        )
        for key, val in overrides.items():
            if val is not None and hasattr(defaults, key):
                setattr(defaults, key, val)
        return defaults

    def validate(self) -> None:
        if self.auth_mode not in ("none", "token"):
            raise ValueError(f"Unsupported auth mode: {self.auth_mode}")
        if self.auth_mode == "token" and not self.server_token:
            raise ValueError("BIOLM_SERVER_TOKEN is required when auth mode is token")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/v3"

    def configured_slugs(self) -> List[str]:
        if not self.models_env.strip():
            return []
        return [s.strip() for s in self.models_env.split(",") if s.strip()]
