import multiprocessing
import os
import warnings

cpu_count = multiprocessing.cpu_count()
max_threads = cpu_count * 4


def _env(name: str, *legacy_names: str) -> str:
    """Read env var with canonical name first, then legacy fallbacks."""
    val = os.environ.get(name)
    if val:
        return val
    for legacy in legacy_names:
        val = os.environ.get(legacy)
        if val:
            warnings.warn(
                f"Environment variable {legacy} is deprecated; use {name} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return val
    return ""


def _env_bool(name: str, *legacy_names: str) -> bool:
    val = _env(name, *legacy_names)
    return str(val).lower() in ("true", "1", "yes")


def _ensure_scheme(domain: str) -> str:
    if domain and not domain.startswith(("http://", "https://")):
        return f"http://{domain}"
    return domain


def _strip_api_suffix(url: str) -> str:
    for suffix in ("/api/v3", "/api/v2", "/api/v1"):
        if url.rstrip("/").endswith(suffix):
            return url.rstrip("/")[: -len(suffix)]
    return url.rstrip("/")


# --- Base domain ---

_local = _env_bool("BIOLM_LOCAL", "BIOLMAI_LOCAL")
_domain_override = _env("BIOLM_BASE_DOMAIN", "BIOLMAI_BASE_DOMAIN")
_api_url_override = _env("BIOLM_BASE_API_URL", "BIOLMAI_BASE_API_URL")

if _domain_override:
    BIOLM_BASE_DOMAIN = _ensure_scheme(_domain_override)
elif _local:
    BIOLM_BASE_DOMAIN = "http://localhost:8000"
else:
    BIOLM_BASE_DOMAIN = "https://biolm.ai"

# --- API URL ---

if _api_url_override:
    BIOLM_BASE_API_URL = _ensure_scheme(_api_url_override)
elif _domain_override or _local:
    BIOLM_BASE_DOMAIN = _ensure_scheme(BIOLM_BASE_DOMAIN)
    BIOLM_BASE_API_URL = f"{BIOLM_BASE_DOMAIN.rstrip('/')}/api/v3"
else:
    BIOLM_BASE_API_URL = "https://biolm.ai/api/v3"

# BIOLM_BASE_API_URL overrides model inference only; platform domain stays on biolm.ai
# unless BIOLM_BASE_DOMAIN is explicitly set (hybrid: login on platform, models on proxy).

# Legacy aliases (deprecated)
BIOLMAI_BASE_DOMAIN = BIOLM_BASE_DOMAIN
BIOLMAI_BASE_API_URL = BIOLM_BASE_API_URL

USER_BIOLM_DIR = os.path.join(os.path.expanduser("~"), ".biolmai")
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, "credentials")
GEN_TOKEN_URL = f"{BIOLM_BASE_DOMAIN}/ui/accounts/user-api-tokens/"

# --- Thread pool ---

_threads_raw = _env("BIOLM_THREADS", "BIOLMAI_THREADS") or "16"
MULTIPROCESS_THREADS = _threads_raw
if isinstance(MULTIPROCESS_THREADS, str) and not MULTIPROCESS_THREADS:
    MULTIPROCESS_THREADS = 16
if int(MULTIPROCESS_THREADS) > max_threads or int(MULTIPROCESS_THREADS) > 128:
    err = (
        f"Maximum threads allowed is 4x number of CPU cores ("
        f"{max_threads}) or 128, whichever is lower."
    )
    err += " Please update environment variable BIOLM_THREADS."
    raise ValueError(err)
elif int(MULTIPROCESS_THREADS) <= 0:
    err = "Environment variable BIOLM_THREADS must be a positive integer."
    raise ValueError(err)

BASE_API_URL_V1 = f"{BIOLM_BASE_DOMAIN}/api/v1"
BASE_API_URL = f"{BIOLM_BASE_DOMAIN}/api/v3"

# --- OAuth ---

BIOLM_PUBLIC_CLIENT_ID = _env("BIOLM_OAUTH_CLIENT_ID", "BIOLMAI_OAUTH_CLIENT_ID") or (
    "2t_fFfnx9UjgmVp8EGbJRL24UbVynZ5Yo2JOv_R2eQc"
)
BIOLMAI_PUBLIC_CLIENT_ID = BIOLM_PUBLIC_CLIENT_ID

BIOLM_OAUTH_CLIENT_SECRET = (
    _env("BIOLM_OAUTH_CLIENT_SECRET", "BIOLMAI_OAUTH_CLIENT_SECRET")
    or os.environ.get("CLIENT_SECRET", "")
)
BIOLMAI_OAUTH_CLIENT_SECRET = BIOLM_OAUTH_CLIENT_SECRET

OAUTH_AUTHORIZE_URL = f"{BIOLM_BASE_DOMAIN}/o/authorize/"
OAUTH_TOKEN_URL = f"{BIOLM_BASE_DOMAIN}/o/token/"
if BIOLM_BASE_DOMAIN == "http://localhost:7777" or BIOLM_BASE_DOMAIN.endswith(":7777"):
    OAUTH_INTROSPECT_URL = "http://localhost:8000/o/introspect/"
else:
    OAUTH_INTROSPECT_URL = f"{BIOLM_BASE_DOMAIN}/o/introspect/"
OAUTH_REDIRECT_URI = "http://localhost:8765/callback"

# --- Server defaults ---

BIOLM_SERVER_HOST = os.environ.get("BIOLM_SERVER_HOST", "127.0.0.1")
BIOLM_SERVER_PORT = int(os.environ.get("BIOLM_SERVER_PORT", "8787"))
BIOLM_SERVER_TOKEN = os.environ.get("BIOLM_SERVER_TOKEN", "")
BIOLM_SERVER_AUTH = os.environ.get("BIOLM_SERVER_AUTH", "none")
BIOLM_SERVER_MODELS = os.environ.get("BIOLM_SERVER_MODELS", "")
BIOLM_SERVER_REFRESH_SECONDS = int(os.environ.get("BIOLM_SERVER_REFRESH_SECONDS", "60"))
BIOLM_SERVER_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".biolm", "server.yaml"
)


def get_base_domain() -> str:
    """Platform site root (OAuth, auth, hosted platform APIs)."""
    return BIOLM_BASE_DOMAIN.rstrip("/")


def get_model_catalog_base() -> str:
    """Site root for model catalog/list endpoints.

    When BIOLM_BASE_API_URL is set (e.g. biolm server proxy), catalog listing
    follows the model API host. Platform auth still uses get_base_domain().
    """
    if _api_url_override:
        return _ensure_scheme(_strip_api_suffix(BIOLM_BASE_API_URL)).rstrip("/")
    return BIOLM_BASE_DOMAIN.rstrip("/")


def get_base_api_url() -> str:
    """Return the v3 model API base URL."""
    return BIOLM_BASE_API_URL.rstrip("/")
