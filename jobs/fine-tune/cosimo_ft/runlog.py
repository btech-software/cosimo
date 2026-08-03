"""Run directories and provenance records.

A run is one directory under ``runs/``::

    runs/<name>/
      resolved_config.yaml   the fully merged config the run executed with
      env.json               interpreter, package versions, GPU, git commit
      manifest.json          what the run produced
      eval/                  <suite>_generations.jsonl, metrics.json
      tb/                    tensorboard logs
      adapter/               trained LoRA adapter
      checkpoints/           trainer checkpoints
      merged/                merged bf16 weights (07_export_merge.py)

Import-light: torch/transformers are only touched inside ``env_info``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as config_mod

logger = logging.getLogger(__name__)

# Packages worth recording; missing ones are reported as null.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
)


def utc_now() -> str:
    """Current time as an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, payload: Any) -> Path:
    """Write pretty JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    return target


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, returning [] when it does not exist.

    Unparseable lines are skipped with a warning rather than raising: a crash
    between write and flush leaves a torn last line, and ``--resume`` is exactly
    the recovery path for that crash.
    """
    source = Path(path)
    if not source.is_file():
        return []
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "%s: skipping unparseable line %d (%s)", source, number, exc
                )
    return rows


def append_jsonl(path: str | Path, rows: list[dict]) -> Path:
    """Append rows to a JSONL file and fsync, so a crash keeps what was written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config_mod.HARNESS_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _gpu_info() -> dict:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
    except Exception:
        return info
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["capability"] = f"{props.major}.{props.minor}"
            info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def env_info() -> dict:
    """Snapshot of the execution environment for reproducibility."""
    return {
        "created_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "gpu": _gpu_info(),
        "git_commit": _git_commit(),
        "env": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE")
        },
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }


class RunDir:
    """The output directory of a single run."""

    def __init__(self, runs_dir: str | Path, name: str) -> None:
        self.name = name
        self.root = config_mod.harness_path(runs_dir) / name

    def __repr__(self) -> str:
        return f"RunDir(name={self.name!r}, root={str(self.root)!r})"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def tb_dir(self) -> Path:
        return self.root / "tb"

    @property
    def adapter_dir(self) -> Path:
        return self.root / "adapter"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def merged_dir(self) -> Path:
        return self.root / "merged"

    def exists(self) -> bool:
        return self.root.exists()

    def create(self, *subdirs: str) -> "RunDir":
        """Create the run root (and any named subdirectories)."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in subdirs:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def save_config(self, cfg: dict, dest: Path | None = None) -> Path:
        """Persist the resolved config as ``resolved_config.yaml``.

        ``dest`` selects the directory; evaluation passes ``eval_dir`` so that
        re-evaluating a checkpoint cannot overwrite the config it was trained with.
        """
        return config_mod.save_config(cfg, (dest or self.root) / "resolved_config.yaml")

    def save_env(self, extra: dict | None = None, dest: Path | None = None) -> Path:
        """Persist the environment snapshot as ``env.json``."""
        payload = env_info()
        if extra:
            payload.update(extra)
        return write_json((dest or self.root) / "env.json", payload)

    def read_manifest(self) -> dict:
        """Read ``manifest.json``, returning {} when absent or corrupt."""
        path = self.root / "manifest.json"
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("%s is unreadable (%s); starting a new manifest", path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def write_manifest(self, payload: dict) -> Path:
        """Persist what the run produced as ``manifest.json``."""
        data = {"run": self.name, "created_at": utc_now()}
        data.update(payload)
        return write_json(self.root / "manifest.json", data)

    def append_manifest_entry(self, section: str, entry: dict) -> Path:
        """Append an entry to a list section of ``manifest.json``, keeping the rest.

        Evaluation uses this instead of ``write_manifest`` because a run
        directory is shared with the training stage that produced it: replacing
        the manifest would destroy the record of how the checkpoint was trained.
        """
        data = self.read_manifest()
        data.setdefault("run", self.name)
        existing = data.get(section)
        entries = list(existing) if isinstance(existing, list) else []
        entries.append(entry)
        data[section] = entries
        return write_json(self.root / "manifest.json", data)
