"""Runtime settings for the two ways this server can run.

local   (default)  Everything on your machine, no AWS, no accounts. Flask serves
                   the UI itself, so the page and the API share an origin: no
                   CORS, no pairing token. This is `python -m rl_learning.server.app`
                   and it behaves exactly as it always has.

worker             The UI is served from the cloud (CloudFront) and this process
                   is just the GPU/LLM engine on the user's own machine. The page
                   is a *different* origin, so we allow it through CORS and demand
                   a pairing token on every API call.

The GPU, the models and the API keys never leave the machine in either mode.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

LOCAL = "local"
WORKER = "worker"

# Where the pairing token is kept, so it survives a restart and the user does not
# have to re-pair the browser every time.
STATE_DIR = Path(os.environ.get("RL_STATE_DIR", Path.home() / ".rl_playground"))
TOKEN_FILE = STATE_DIR / "token"

DEFAULT_PORT = 5000
# Distinct from the local-mode port: a worker is a background service and should
# not fight the dev server for :5000.
DEFAULT_WORKER_PORT = 5057


def _read_token() -> str:
    """The pairing token, generated once and reused."""
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    except FileNotFoundError:
        pass

    token = secrets.token_urlsafe(24)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        TOKEN_FILE.chmod(0o600)  # best-effort; a no-op on Windows
    except OSError:
        pass
    return token


def _split(value: str) -> list[str]:
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    mode: str = LOCAL
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    # Cloud origins allowed to drive this worker. Anything not on this list is
    # refused, so a random page the user happens to visit cannot reach the GPU.
    allowed_origins: list[str] = field(default_factory=list)
    token: str = ""
    debug: bool = False

    @property
    def is_worker(self) -> bool:
        return self.mode == WORKER

    @property
    def serves_ui(self) -> bool:
        """In worker mode the cloud serves the UI; we are only the engine."""
        return self.mode == LOCAL


def load(**overrides) -> Settings:
    """Settings from defaults, then env vars, then explicit overrides (CLI)."""
    mode = os.environ.get("RL_MODE", LOCAL).strip().lower()
    if mode not in (LOCAL, WORKER):
        raise SystemExit(f"RL_MODE must be '{LOCAL}' or '{WORKER}', got {mode!r}")

    default_port = DEFAULT_WORKER_PORT if mode == WORKER else DEFAULT_PORT
    settings = Settings(
        mode=mode,
        # Loopback only. Binding 0.0.0.0 would expose an unauthenticated GPU and
        # the user's API keys to their whole network.
        host=os.environ.get("RL_HOST", "127.0.0.1"),
        port=int(os.environ.get("RL_PORT") or os.environ.get("PORT") or default_port),
        allowed_origins=_split(os.environ.get("RL_ALLOWED_ORIGINS", "")),
        debug=os.environ.get("RL_DEBUG", "").lower() in ("1", "true", "yes"),
    )

    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)

    # Re-derive the port if the mode was flipped on the command line but the port
    # was not given explicitly.
    if "mode" in overrides and overrides["mode"] and not (
        overrides.get("port") or os.environ.get("RL_PORT") or os.environ.get("PORT")
    ):
        settings.port = DEFAULT_WORKER_PORT if settings.is_worker else DEFAULT_PORT

    if settings.is_worker:
        settings.token = os.environ.get("RL_TOKEN") or _read_token()

    return settings
