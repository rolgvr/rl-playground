"""Optional behaviour-capture hooks (off by default).

When the private ``honeypot`` package is installed AND ``HONEYPOT_ENABLED=1``
is set (e.g. in the project-root .env), every request is checked against a set
of randomised traps and scored in DynamoDB, and calls to the LLM endpoints are
logged as model interactions. The point is observability: seeing how automated
clients — scrapers, scanners, and increasingly AI agents — actually behave when
they interact with the platform.

On a normal local install neither condition holds and this module is a no-op:
no import cost, no request overhead, no data leaves the machine. Every hook is
also fail-open — a capture error must never break a learner's session.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("rl.honeypot")

# LLM-backed endpoints whose usage is logged as a model interaction.
_MODEL_PATH_PREFIXES = ("/api/llm/", "/api/uni/ask", "/api/reason/run")


def _enabled() -> bool:
    return os.environ.get("HONEYPOT_ENABLED", "").strip().lower() in ("1", "true", "yes")


def init(app) -> None:
    """Install the capture hooks on *app*. Safe no-op unless explicitly enabled."""
    if not _enabled():
        return

    try:
        from honeypot import create_honeypot
        from honeypot.enforcer import apply_delay_sync
    except ImportError:
        logger.warning(
            "HONEYPOT_ENABLED is set but the honeypot package is not installed "
            "(see requirements-ops.txt); running without behaviour capture."
        )
        return

    try:
        hp = create_honeypot(os.environ.get("HONEYPOT_PROJECT", "rl_learning"))
    except Exception as exc:  # missing AWS creds, no network, etc.
        logger.warning("honeypot init failed (%s); running without behaviour capture", exc)
        return

    from flask import jsonify, request

    def _client_ip() -> str:
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.remote_addr or "unknown"

    @app.before_request
    def _behaviour_capture():
        try:
            ua = request.headers.get("User-Agent", "")
            result = hp.check_request(
                ip=_client_ip(),
                path=request.path,
                method=request.method,
                query_params=dict(request.args),
                body_params=dict(request.form),
                headers=dict(request.headers),
                user_agent=ua,
            )

            if request.method == "POST" and request.path.startswith(_MODEL_PATH_PREFIXES):
                hp.log_model_interaction(
                    ip=_client_ip(),
                    prompt_category=request.path.rstrip("/").rsplit("/", 1)[-1],
                    description=f"POST {request.path}",
                    endpoint=request.path,
                    user_agent=ua,
                )
        except Exception as exc:
            logger.debug("behaviour capture skipped: %s", exc)
            return None

        if result.action in ("blocked", "banned"):
            apply_delay_sync(result)
            return jsonify(error="denied"), 403
        apply_delay_sync(result)
        return None

    logger.info("behaviour capture active (project=%s)",
                os.environ.get("HONEYPOT_PROJECT", "rl_learning"))
