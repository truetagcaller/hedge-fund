"""YAML configuration loader with environment overlay and env-var expansion.

Resolution order (last wins):
    1. ``config/base.yaml``
    2. ``config/{HEDGEFUND_ENV}.yaml``  (e.g. ``config/production.yaml``)
    3. Environment variables prefixed with ``HEDGEFUND_``

Environment variables are mapped to nested keys using ``__`` as separator.
For example ``HEDGEFUND_REDIS__HOST=10.0.0.5`` sets ``redis.host``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from hedgefund.logger import get_logger

log = get_logger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_ENV_PREFIX = "HEDGEFUND_"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_value(raw: str) -> Any:
    """Best-effort cast of a string env-var value to a native Python type."""
    if raw.lower() in ("true", "yes", "1"):
        return True
    if raw.lower() in ("false", "no", "0"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay environment variables onto *config*.

    ``HEDGEFUND_SECTION__KEY=val`` maps to ``config["section"]["key"] = val``.
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        parts = env_key[len(_ENV_PREFIX) :].lower().split("__")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_value(env_val)
    return config


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read and parse a YAML file, returning an empty dict if it doesn't exist."""
    if not path.is_file():
        return {}
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_config(
    config_dir: Path | str | None = None,
    env: str | None = None,
) -> Dict[str, Any]:
    """Load, merge, and return the final configuration dictionary.

    Parameters
    ----------
    config_dir:
        Directory containing YAML config files. Defaults to ``<repo>/config``.
    env:
        Environment name (e.g. ``production``, ``staging``). Falls back to the
        ``HEDGEFUND_ENV`` environment variable, then ``"development"``.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
    env = env or os.environ.get("HEDGEFUND_ENV", "development")

    base_path = config_dir / "base.yaml"
    env_path = config_dir / f"{env}.yaml"

    base_cfg = _load_yaml(base_path)
    if not base_cfg:
        log.warning("base_config_missing", path=str(base_path))

    env_cfg = _load_yaml(env_path)
    if env_cfg:
        log.info("env_config_loaded", env=env, path=str(env_path))

    merged = _deep_merge(copy.deepcopy(base_cfg), env_cfg)
    merged = _apply_env_overrides(merged)

    log.debug("config_loaded", env=env, keys=list(merged.keys()))
    return merged
