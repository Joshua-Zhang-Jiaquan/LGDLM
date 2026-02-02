import ast
from dataclasses import dataclass

import yaml


class DotDict(dict):
    """Recursive dict that supports attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _to_dotdict(obj):
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def to_plain_dict(obj):
    if isinstance(obj, DotDict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _to_dotdict(data)


def _parse_scalar(s: str):
    # Try strict YAML-like scalars first.
    lowered = s.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    # Fallback: literal_eval for quoted strings, lists, dicts.
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def apply_overrides(cfg: DotDict, overrides: list[str]) -> DotDict:
    for ov in overrides:
        if "=" not in ov:
            continue
        key, raw = ov.split("=", 1)
        value = _parse_scalar(raw)

        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = DotDict()
            cur = cur[p]
        cur[parts[-1]] = _to_dotdict(value)
    return cfg
