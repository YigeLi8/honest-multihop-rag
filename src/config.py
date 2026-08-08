"""Run configs: yaml files deep-merged over configs/base.yaml."""
from pathlib import Path
from types import SimpleNamespace

import yaml

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    return x


def load_config(path):
    with open(BASE) as f:
        cfg = yaml.safe_load(f)
    if Path(path).resolve() != BASE:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _ns(cfg)
