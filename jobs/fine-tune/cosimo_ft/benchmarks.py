"""Evaluation suites.

Two Cosimo suites read prepared JSONL from ``data/processed/`` (produced by
``01_prepare_data.py``); GSM8K and MATH-500 are public math benchmarks used as
regression checks — they measure whether fine-tuning damaged general reasoning,
not finance skill.

Every suite yields items shaped
``{"id", "suite", "question", "gold", "question_type", "meta"}``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import config as config_mod

COSIMO_SUITES = {
    "cosimo_test": "eval_cosimo_test.jsonl",
    "cosimo_unseen_stems": "eval_cosimo_unseen_stems.jsonl",
}
MATH_SUITES = ("gsm8k", "math500")
ALL_SUITES = tuple(COSIMO_SUITES) + MATH_SUITES


def is_math_suite(name: str) -> bool:
    """Math suites are graded with ``grade_math`` rather than ``grade_cosimo``.

    They share the evaluation system message with the Cosimo suites: the identity
    applies everywhere, and GSM8K/MATH-500 are exam-format items, so the exam
    protocol is the right task block for them too.
    """
    return name in MATH_SUITES


def _subsample(items: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministic subsample preserving the original (id-sorted) order."""
    if n is None or n >= len(items):
        return items
    if n <= 0:
        return []
    picked = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in picked]


def _load_hf(dataset_id: str, *args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - requires the training image
        raise RuntimeError(
            "the `datasets` package is required for the gsm8k/math500 suites; "
            "run inside the fine-tune image or pass --suites cosimo_test"
        ) from exc
    try:
        return load_dataset(dataset_id, *args, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"could not load `{dataset_id}` from the Hugging Face Hub ({exc!r}). "
            "Check network access and HF_TOKEN, or run offline with "
            "--suites cosimo_test cosimo_unseen_stems"
        ) from exc


def load_cosimo_suite(path: str) -> list[dict]:
    """Read a prepared Cosimo eval JSONL into suite items."""
    source = config_mod.harness_path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"prepared eval file not found: {source}. "
            "Run scripts/01_prepare_data.py first."
        )
    suite = source.stem.replace("eval_", "")
    items = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                {
                    "id": str(row["id"]),
                    "suite": suite,
                    "question": row["question"],
                    "gold": row["answer"],
                    "question_type": row.get("question_type", ""),
                    "meta": row,
                }
            )
    items.sort(key=lambda item: item["id"])
    return items


def load_gsm8k(n: int, seed: int) -> list[dict]:
    """GSM8K test split; the gold value follows the '#### ' marker."""
    rows = _load_hf("openai/gsm8k", "main", split="test")
    items = []
    for index, row in enumerate(rows):
        items.append(
            {
                "id": f"gsm8k-{index:05d}",
                "suite": "gsm8k",
                "question": row["question"],
                "gold": row["answer"].split("####")[-1].strip(),
                "question_type": "Calculation",
                "meta": {"index": index},
            }
        )
    items.sort(key=lambda item: item["id"])
    return _subsample(items, n, seed)


def load_math500(n: int, seed: int) -> list[dict]:
    """MATH-500 test split; the gold value is the ``answer`` column."""
    rows = _load_hf("HuggingFaceH4/MATH-500", split="test")
    items = []
    for index, row in enumerate(rows):
        items.append(
            {
                "id": str(row.get("unique_id") or f"math500-{index:05d}"),
                "suite": "math500",
                "question": row["problem"],
                "gold": str(row["answer"]),
                "question_type": "Calculation",
                "meta": {
                    "index": index,
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                },
            }
        )
    items.sort(key=lambda item: item["id"])
    return _subsample(items, n, seed)


def load_suite(name: str, cfg: dict, *, limit: int | None = None) -> list[dict]:
    """Load one suite by name, honouring ``eval.samples`` and an optional ``limit``."""
    if name not in ALL_SUITES:
        raise ValueError(
            f"unknown suite {name!r}; known suites: {', '.join(ALL_SUITES)}"
        )
    seed = int(config_mod.get(cfg, "seed", 3407))
    n = config_mod.get(cfg, f"eval.samples.{name}")
    if name in COSIMO_SUITES:
        processed_dir = Path(
            config_mod.get(cfg, "paths.processed_dir", "data/processed")
        )
        items = load_cosimo_suite(str(processed_dir / COSIMO_SUITES[name]))
        items = _subsample(items, n, seed)
    elif name == "gsm8k":
        items = load_gsm8k(n, seed)
    else:
        items = load_math500(n, seed)
    if limit is not None:
        items = items[:limit]
    return items
