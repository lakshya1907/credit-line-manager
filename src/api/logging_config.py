"""
src/api/logging_config.py
────────────────────────────
Minimal structured (JSON-lines) logging -- one log record per line, each a
JSON object, so container log collectors (Docker, any hosting platform)
can parse fields without a separate log-shipping agent. This is not an
observability stack (no Prometheus/Grafana, no metrics, no tracing) --
that's a deliberate scope call for a single-instance project; see
CLAUDE.md's Infrastructure section.
"""

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Quiet down noisy third-party loggers at INFO; still shows warnings/errors.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_request(logger: logging.Logger, method: str, path: str, status_code: int, duration_ms: float) -> None:
    logger.info(
        f"{method} {path} {status_code}",
        extra={"extra_fields": {
            "method": method, "path": path, "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }},
    )


class Timer:
    """Small helper so the request-logging middleware doesn't hand-roll
    time.time() arithmetic inline."""
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        self.duration_ms = (time.time() - self._start) * 1000
