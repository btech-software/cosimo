"""Deterministic seeds for v3 generation (spec §5.2).

Same contract as v1/v2 -- no wall clock, no PID, no iteration order -- but a
different key tuple: ``(work_type, family, record_type, variant)`` with the
schema version, because the v3 unit of generation is a scenario family, not a
(template, program) pair. The hash function stays ``pipelines.core.stable_hash``
(shunned builtin ``hash`` is salted per process by PYTHONHASHSEED, which is what
made the first corpus impossible to resume).

``pack_seed`` deliberately puts the literal ``"pack"`` in the record-type slot:
one fact pack is shared by every record type rendered from the same
(work_type, family, variant), so all five renderers must see byte-identical
facts. The per-record-type seed lives in :func:`render_seed` and drives only
phrasing choices (register, trace style, fault schedule), never numbers.
"""

from __future__ import annotations

import os
import random
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from pipelines import core  # noqa: E402

from .config import SCHEMA_VERSION  # noqa: E402


def pack_seed(work_type: str, family: str, variant: int) -> int:
    """Seed of the scenario's fact pack, shared across every record type."""
    return core.stable_hash(f"{work_type}|{family}|pack|{variant}|{SCHEMA_VERSION}")


def render_seed(work_type: str, family: str, record_type: str, variant: int) -> int:
    """Seed of one render job's phrasing randomness, per spec §5.2."""
    return core.stable_hash(
        f"{work_type}|{family}|{record_type}|{variant}|{SCHEMA_VERSION}"
    )


def rng_for(seed: int) -> random.Random:
    """A fresh Random for one deterministic draw sequence."""
    return random.Random(seed)
