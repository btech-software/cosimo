"""Deterministic dataset splitting.

Pure python (stdlib only) so it can be unit-tested on a CPU-only laptop.

Four splits are produced:

``train``/``val``/``test``
    A stratified IID partition, stratified by ``(program, generator)``.
``unseen_stems``
    Every record whose *stem family* is held out. These never appear in
    ``train``/``val``/``test``, which makes them a measurement of
    generalisation to question structures the model never trained on.

Holding out is done by **family**, not by generator name: seven generators are
``v_``/``cr_``/``m_`` wrappers over a base stem, so excluding only the wrapper
would leak the same question structure into training.

The assignment is keyed by ``id`` and reused for the ``preference_pairs``
config, so a question can never be in DPO training and in the test set at the
same time.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterable

from .data_schema import stem_family

TRAIN = "train"
VAL = "val"
TEST = "test"
UNSEEN_STEMS = "unseen_stems"

SPLIT_NAMES = (TRAIN, VAL, TEST, UNSEEN_STEMS)
TRAINABLE_SPLITS = (TRAIN, VAL, TEST)


def _stratum_rng(seed: int, key: tuple[str, str]) -> random.Random:
    """A per-stratum RNG derived from ``seed`` and the stratum key.

    Deriving the stream from a hash of the key (rather than drawing from one
    shared generator) makes each stratum's *shuffle* independent of the other
    strata. The per-stratum *counts* are still coupled, because the cumulative
    allocator below carries rounding remainders forward in sorted key order.
    """
    material = f"{seed}\x00{key[0]}\x00{key[1]}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def assign_splits(
    records: list[dict],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
    holdout_families: Iterable[str],
) -> dict[str, str]:
    """Map every record ``id`` to one of :data:`SPLIT_NAMES`.

    ``records`` are dicts carrying at least ``id``, ``program`` and
    ``generator``; ``stem_family`` is used when present and derived from
    ``generator`` otherwise. Duplicate ids keep their first occurrence.

    Within each ``(program, generator)`` stratum the ids are sorted (stable
    input order), shuffled with the stratum RNG, then the leading ``test_frac``
    become ``test``, the next ``val_frac`` become ``val`` and the rest
    ``train``. Per-stratum counts are allocated against a running cumulative
    target so rounding cannot systematically starve the small splits.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac!r}")
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac!r}")
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac must leave room for training, got "
            f"{val_frac!r} + {test_frac!r}"
        )

    families = {str(f) for f in (holdout_families or ())}
    assignment: dict[str, str] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[str] = set()

    for record in records:
        record_id = str(record["id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        generator = str(record.get("generator") or "unknown")
        family = str(record.get("stem_family") or "") or stem_family(generator)
        if family in families:
            assignment[record_id] = UNSEEN_STEMS
            continue
        strata[(str(record.get("program") or ""), generator)].append(record_id)

    cumulative = 0
    taken_test = 0
    taken_val = 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        _stratum_rng(seed, key).shuffle(ids)
        size = len(ids)
        cumulative += size
        n_test = max(0, min(round(test_frac * cumulative) - taken_test, size))
        n_val = max(0, min(round(val_frac * cumulative) - taken_val, size - n_test))
        taken_test += n_test
        taken_val += n_val
        for record_id in ids[:n_test]:
            assignment[record_id] = TEST
        for record_id in ids[n_test : n_test + n_val]:
            assignment[record_id] = VAL
        for record_id in ids[n_test + n_val :]:
            assignment[record_id] = TRAIN
    return assignment
