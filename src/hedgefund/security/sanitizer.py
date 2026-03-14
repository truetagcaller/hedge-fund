"""Log sanitization utilities for stripping sensitive values.

Provides recursive dict sanitization and a structlog processor that
automatically redacts secrets before they reach any log sink.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Set

from hedgefund.logger import get_logger

log = get_logger(__name__)

# Patterns that indicate a dict key holds sensitive data.
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"password",
        r"secret",
        r"token",
        r"key",
        r"credential",
        r"authorization",
    )
)

_REDACTED = "***REDACTED***"


class Sanitizer:
    """Recursively sanitize dicts by masking values whose keys match
    sensitive patterns.

    Example::

        sanitizer = Sanitizer()
        safe = sanitizer.sanitize_dict({"api_key": "abc123", "name": "test"})
        # safe == {"api_key": "***REDACTED***", "name": "test"}
    """

    def __init__(
        self,
        *,
        extra_patterns: list[str] | None = None,
        redacted_value: str = _REDACTED,
    ) -> None:
        patterns = list(_SENSITIVE_PATTERNS)
        if extra_patterns:
            patterns.extend(re.compile(p, re.IGNORECASE) for p in extra_patterns)
        self._patterns = tuple(patterns)
        self._redacted = redacted_value

    def _is_sensitive(self, key: str) -> bool:
        return any(p.search(str(key)) for p in self._patterns)

    def sanitize_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Return a deep copy of *d* with sensitive values replaced."""
        return self._walk(deepcopy(d))

    def _walk(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._redacted if self._is_sensitive(k) else self._walk(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._walk(item) for item in obj)
        return obj


# ---------------------------------------------------------------------------
# structlog processor
# ---------------------------------------------------------------------------

_default_sanitizer = Sanitizer()


def create_sanitizing_processor(
    *,
    extra_patterns: list[str] | None = None,
    redacted_value: str = _REDACTED,
) -> Any:
    """Return a structlog processor that sanitizes every event dict.

    Usage::

        import structlog
        structlog.configure(
            processors=[
                ...,
                create_sanitizing_processor(),
                ...,
            ]
        )
    """
    sanitizer = Sanitizer(extra_patterns=extra_patterns, redacted_value=redacted_value)

    def _processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return sanitizer.sanitize_dict(event_dict)

    return _processor
