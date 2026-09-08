"""The prose contract, executable (analysis spec §6.3; arch spec §5.5, §6).

One function, :func:`gate_violations`, is the whole difference between "the
teacher wrote something" and "the corpus may carry it": invented numbers
(against the pack's ``allowed_numbers``, with the tiny structural whitelist),
``must_mention`` coverage, ``forbidden_claims``, the exam-only ``FINAL ANSWER:``
tag, and the word budget. The renderer's repair loop calls it per attempt;
``verify_v3`` calls the same helpers over the written shards; the publish-time
audit slice calls it again. One policy, three call sites -- the v1/v2 failure
of having the generator and the auditor disagree about "clean" is exactly the
bug class v3 was written to kill.

Coverage is *all* points, not "at least k": the pack's contract says cover
every ``must_mention``, and a k-threshold is a dial nobody would defend at a
review. A row missing a point is repaired, not bartered.

Matching is case- and whitespace-insensitive on purpose: the teacher is
allowed to re-flow a sentence for the register, but the *clause* has to be
recognisable. No synonym fuzzy-matching -- a pack author who wants a point
phrased loosely writes it into ``must_mention`` loosely; the gate does not
guess at meaning it was never given.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verification import nums  # noqa: E402
from verification.gates import FINAL_ANSWER_TAG  # noqa: E402

from ..config import NUMBER_WHITELIST  # noqa: E402
from ..teacher.prompts import WORD_BUDGETS  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402


def whitelist_for(pack: dict) -> frozenset[str]:
    """The token strings the pack itself authorises outside ``allowed_numbers``.

    ``config.NUMBER_WHITELIST`` (percent denominators, the two of a two-way
    bridge, the trading-day convention) plus every number spelled in the
    pack's ``as_of`` date -- a row is allowed to *date* its analysis, and the
    date is the pack's own, drawn from the seeded pool, so leaking it through
    the whitelist re-authorises nothing the pack did not already state.
    """
    as_of_tokens = nums.TOKEN.findall(str(pack.get("as_of") or ""))
    return frozenset(NUMBER_WHITELIST) | frozenset(as_of_tokens)


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def missing_mentions(pack: dict, text: str) -> list[str]:
    """``must_mention`` points the text failed to carry, in pack order."""
    haystack = _norm(text)
    return [
        point
        for point in pack.get("must_mention") or []
        if _norm(point).rstrip(".") not in haystack
    ]


def forbidden_hits(pack: dict, text: str) -> list[str]:
    """``forbidden_claims`` the text asserted anyway, in pack order."""
    haystack = _norm(text)
    return [
        claim
        for claim in pack.get("forbidden_claims") or []
        if _norm(claim).rstrip(".") in haystack
    ]


def gate_violations(pack: dict, text: str, kind: str) -> list[str]:
    """Every way *text* breaks the prose contract for ``kind``, as sentences.

    Empty list means shippable. The returned strings are written to be read
    twice -- once by the repair prompt (the model must be able to find its
    own error in them) and once by whoever post-mortems a dead letter.
    """
    violations: list[str] = []
    offenders = invented_numbers(text, pack["allowed_numbers"], whitelist_for(pack))
    if offenders:
        violations.append(
            "invented numbers not in the fact pack: "
            + ", ".join(repr(t) for t in offenders)
        )
    missing = missing_mentions(pack, text)
    if missing:
        violations.append(
            "must_mention points not covered: " + "; ".join(repr(p) for p in missing)
        )
    hits = forbidden_hits(pack, text)
    if hits:
        violations.append(
            "forbidden claims asserted: " + "; ".join(repr(c) for c in hits)
        )
    if FINAL_ANSWER_TAG.casefold() in _norm(text):
        violations.append(
            f"{FINAL_ANSWER_TAG!r} is an exam contract, not prose style "
            "(spec §6 axis 5); drop it"
        )
    low, high = WORD_BUDGETS[kind]
    words = len(str(text).split())
    if not low <= words <= high:
        violations.append(
            f"length {words} words is outside the {kind} budget {low}-{high}"
        )
    return violations
