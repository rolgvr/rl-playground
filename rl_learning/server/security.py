"""Who is allowed to drive this machine's GPU.

In worker mode the page lives on another origin (the cloud UI), so the browser
sends cross-origin requests to http://127.0.0.1. Two things have to be true
before we run anything:

1. The request comes from an origin we trust (`RL_ALLOWED_ORIGINS`).
2. It carries the pairing token the user copied out of this process.

Both matter. Without (1) any site the user browses could reach the loopback
server; without (2) a trusted-looking origin would be enough on its own. The
token is what ties *this* browser tab to *this* machine.

In local mode the UI is same-origin, so none of this applies and the whole
module is a no-op.
"""

from __future__ import annotations

import hmac

from flask import jsonify, request

# Reachable without a token, so a freshly-loaded page can ask "is a worker
# running here?" before it has been paired. It reveals nothing sensitive, and
# CORS still stops untrusted origins from reading the reply.
PUBLIC_PATHS = {"/api/health"}


def _origin_allowed(origin: str, settings) -> bool:
    return bool(origin) and origin.rstrip("/") in settings.allowed_origins


def _unauthorized(reason: str):
    return jsonify({"error": "unauthorized", "detail": reason}), 401


def init(app, settings) -> None:
    """Install the CORS + token hooks. Does nothing outside worker mode.

    Safe to call twice: settings are read from the environment at import time and
    again once the command line has been parsed. The hooks read the *current*
    settings out of app.config, so the later call wins.
    """
    app.config["_RL_SETTINGS"] = settings

    if not settings.is_worker or app.config.get("_RL_SECURITY_READY"):
        return
    app.config["_RL_SECURITY_READY"] = True

    @app.before_request
    def _authorize():
        settings = app.config["_RL_SETTINGS"]
        # The preflight itself carries no credentials by design; answer it and
        # let the browser send the real, token-bearing request.
        if request.method == "OPTIONS":
            return ("", 204)

        if not request.path.startswith("/api/"):
            return None

        origin = request.headers.get("Origin", "")
        # Same-origin/no-origin callers (curl, the user's own scripts) are fine;
        # a *browser* on an untrusted origin is not.
        if origin and not _origin_allowed(origin, settings):
            return _unauthorized(f"origin {origin} is not allowed")

        if request.path in PUBLIC_PATHS:
            return None

        header = request.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied:
            return _unauthorized("missing pairing token")
        if not hmac.compare_digest(supplied, settings.token):
            return _unauthorized("bad pairing token")

        return None

    @app.after_request
    def _cors(response):
        settings = app.config["_RL_SETTINGS"]
        origin = request.headers.get("Origin", "")
        if _origin_allowed(origin, settings):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Max-Age"] = "600"
            # Chrome's Private Network Access: a public HTTPS page reaching a
            # loopback server must be granted this on the preflight or the
            # request is dropped before it ever arrives.
            if request.headers.get("Access-Control-Request-Private-Network") == "true":
                response.headers["Access-Control-Allow-Private-Network"] = "true"
        # The reply differs per origin, so it must never be cached across them.
        response.headers.add("Vary", "Origin")
        return response
