"""Layered YAML configuration for the fine-tuning harness.

Stdlib + pyyaml only. Every script gets the same override surface via
``add_config_args``: ``--config extra.yaml`` (repeatable) and
``--set dotted.key=value`` (repeatable, value parsed as YAML).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# $H — the harness root (jobs/fine-tune). Never derived from the CWD.
HARNESS_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = HARNESS_ROOT / "configs"


def harness_path(path: str | Path) -> Path:
    """Resolve a config-supplied path against the harness root, not the CWD."""
    p = Path(path)
    return p if p.is_absolute() else HARNESS_ROOT / p


def resolve_config_path(path: str | Path) -> Path:
    """Find a config file by absolute path, CWD-relative path, or config name."""
    p = Path(path)
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [Path.cwd() / p, CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"config file not found: {path!r} (looked in {[str(c) for c in candidates]})"
    )


def _read_yaml(path: str | Path) -> dict:
    resolved = resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"config file {resolved} must contain a mapping, got {type(data).__name__}"
        )
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``. Lists and scalars replace."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``dotted.key=yaml_scalar`` into (key path, value)."""
    if "=" not in item:
        raise ValueError(f"--set expects 'dotted.key=value', got {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--set expects a non-empty key, got {item!r}")
    return key.split("."), yaml.safe_load(raw)


def apply_override(cfg: dict, item: str) -> dict:
    """Return a copy of ``cfg`` with a single ``dotted.key=value`` applied.

    The key must already exist in the merged config. A typo like
    ``--set evla.batch_size=4`` would otherwise be accepted silently and cost a
    full evaluation or training run.
    """
    path, value = parse_override(item)
    merged = deepcopy(cfg)
    node: dict = merged
    for depth, part in enumerate(path[:-1]):
        child = node.get(part)
        if not isinstance(child, dict):
            prefix = ".".join(path[: depth + 1])
            raise KeyError(
                f"--set {item!r}: {prefix!r} is not a config section. "
                f"Known keys at this level: {sorted(node)}"
            )
        node = child
    if path[-1] not in node:
        raise KeyError(
            f"--set {item!r}: unknown key {'.'.join(path)!r}. "
            f"Known keys at this level: {sorted(node)}"
        )
    node[path[-1]] = value
    return merged


def load_config(
    stage: str | None = None,
    extra: list[str] | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Deep-merge base.yaml -> <stage>.yaml -> each ``extra`` file -> ``overrides``."""
    cfg = _read_yaml(CONFIG_DIR / "base.yaml")
    if stage:
        cfg = deep_merge(cfg, _read_yaml(CONFIG_DIR / f"{stage}.yaml"))
    for path in extra or []:
        cfg = deep_merge(cfg, _read_yaml(path))
    for item in overrides or []:
        cfg = apply_override(cfg, item)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable short digest of a resolved config (sha256 of canonical JSON)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def save_config(cfg: dict, path: str | Path) -> Path:
    """Write the resolved config as YAML, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            cfg, handle, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
    return target


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Look up ``a.b.c`` in a nested mapping, returning ``default`` if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the shared ``--config``/``--set`` override flags."""
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="extra YAML config layered on top of the defaults (repeatable)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set",
        metavar="KEY=VALUE",
        help="override a single config key, e.g. --set eval.batch_size=8 (repeatable)",
    )
    return parser
