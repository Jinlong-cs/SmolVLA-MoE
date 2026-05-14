from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    merged: dict[str, Any] = {}
    defaults = data.pop("defaults", []) or []
    for default in defaults:
        default_path = (path.parent / default).resolve()
        merged = deep_update(merged, load_config(default_path))
    return deep_update(merged, data)


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor
