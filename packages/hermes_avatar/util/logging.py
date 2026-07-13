from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure logging for the ``hermes_avatar`` logger hierarchy.

    Resolution order for the level: an explicit ``level`` argument wins; otherwise
    the ``HERMES_LOG_LEVEL`` environment variable is read; otherwise we default to
    ``INFO``. The level is applied to the ``hermes_avatar`` logger so that every
    submodule using ``logging.getLogger(__name__)`` inherits it.

    A console handler is attached only when no handler is already configured on the
    root or ``hermes_avatar`` loggers. This keeps the helper idempotent across
    repeated calls (e.g. ``reload_config``) and avoids fighting an already-configured
    environment such as pytest, gunicorn, or the host process's own logging setup.
    """
    if level is None:
        level = os.environ.get("HERMES_LOG_LEVEL", "INFO").upper()
    numeric = getattr(logging, level, None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    logger = logging.getLogger("hermes_avatar")
    logger.setLevel(numeric)

    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)

    logger.info(
        "logging configured",
        extra={"audit": {"event": "logging.configured", "level": level}},
    )
