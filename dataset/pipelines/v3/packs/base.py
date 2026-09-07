"""The fact pack: v3's unit of truth (spec §4).

Every v3 row -- exam, prose, agentic, implementation -- is rendered over one
of these. It is frozen, JSON-serialisable, and carries its own verification
contract: the numbers a completion may use (:data:`FactPack.allowed_numbers`),
the points it must make, and the claims it must not. ``verify_v3`` recomputes
the pack from ``(work_type, family, variant)`` through the registered computer
and requires equality; that is why every field here is data, not prose.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass
from typing import Iterable


# A computer that cannot produce a *consistent* pack raises this; the inventory
# marks the variant skipped, not dead-lettered (spec §5.3). A dead letter is a
# teacher that misbehaved; a PackError is a scenario that does not exist.
class PackError(Exception):
    """This (work_type, family, variant) yields no coherent scenario."""


@dataclass(frozen=True)
class FactPack:
    schema_version: str
    scenario_id: str
    work_type: str
    seed: int
    variant: int
    entities: list[dict]
    inputs: dict
    computed: dict
    formulas: list[str]
    allowed_numbers: list[float]
    forbidden_claims: list[str]
    must_mention: list[str]
    register: str
    as_of: str
    question: str
    stimulus: str | None = None
    program: str | None = None
    topic: str | None = None
    subtopic: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        """The Hub-compatible single-string form (v2's JSON-column compromise)."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    def number_set(self) -> set[float]:
        return {float(x) for x in self.allowed_numbers}


def assemble_numbers(
    inputs: dict, computed: dict, extra: Iterable[float] = ()
) -> list[float]:
    """Flatten every scalar a pack may render into ``allowed_numbers``.

    Deterministic and deduplicated: sorted ascending, floats cleaned to 12 dp so
    a binary-noise tail (``0.30000000000000004``) cannot make the same economic
    number appear twice under two spellings. ``extra`` is where the caller adds
    the values it renders as percentages or counts directly in the question text
    (a ``3.00%`` prints ``3.0``, which is a different token from ``0.03``).
    """
    values: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return  # a flag is not a number a completion may quote
        if isinstance(node, (int, float)):
            values.append(round(float(node), 12))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(inputs)
    walk(computed)
    for value in extra:
        values.append(round(float(value), 12))
    return sorted(set(values))


AS_OF_DATES = ("2025-12-31", "2026-03-31", "2026-06-30")


def pick_as_of(rng: random.Random) -> str:
    """An as-of date from a fixed pool -- drawn from the seed, never the clock.

    A wall-clock date in a deterministic row would make two runs of the same
    seed produce different shards, which breaks resume and the recompute gate.
    """
    return rng.choice(list(AS_OF_DATES))


def make_ticker(rng: random.Random) -> str:
    """A synthetic 4-letter ticker; real tickers would invite real data drift."""
    return "".join(rng.sample(string.ascii_uppercase, 4))
