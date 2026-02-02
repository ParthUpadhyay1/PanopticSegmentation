from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate(obj: Any, scope: Dict[str, Any]) -> Any:
    """Recursively expand ${var} in strings using values from scope."""
    if isinstance(obj, dict):
        return {k: _interpolate(v, scope) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, scope) for v in obj]
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            return str(scope.get(key, os.environ.get(key, m.group(0))))
        return _VAR_PATTERN.sub(repl, obj)
    return obj


def load_yaml_with_includes(path: str) -> Dict[str, Any]:
    """Load YAML and optionally include another YAML file:
    paths:
      include: some.yaml

    Also supports ${var} interpolation (var can be defined in the same YAML).
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = cfg or {}
    include = None
    if isinstance(cfg, dict) and "paths" in cfg and isinstance(cfg["paths"], dict):
        include = cfg["paths"].get("include")

    merged = {}
    if include:
        include_path = include
        # Resolve relative include path against current file directory
        if not os.path.isabs(include_path):
            include_path = os.path.join(os.path.dirname(path), include_path)
        with open(include_path, "r", encoding="utf-8") as f:
            inc = yaml.safe_load(f) or {}
        merged.update(inc)

    merged.update(cfg)

    # Build interpolation scope from merged dict (top-level keys)
    scope = dict(merged)
    merged = _interpolate(merged, scope)
    return merged
