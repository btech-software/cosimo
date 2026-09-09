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

What changed after the first live run is *what a point is*. Matching was a
verbatim substring test over an English clause, and it failed correct answers
on articles and word order -- see :func:`missing_mentions`. Points are now
short **anchors** of content words, matched on stems, in any order, ignoring
function words. Still no synonym table: the gate does not guess at meaning it
was never given, it only stops insisting on one conjugation and one sentence
shape. The analytical *depth* an anchor stands for is driven by the brief and
checked by human review, not by a string test that cannot see it.
"""

from __future__ import annotations

import os
import re
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


#: Suffixes stripped before an anchor is matched, longest first. English
#: inflection only, and only enough of it to stop a *correct* answer failing on
#: grammar: a pack anchored on "scaling" must accept "scales", one anchored on
#: "normality" must accept "normal". No stemmer library and no synonym table --
#: the gate still refuses to guess at meaning it was never given, it just stops
#: insisting on a particular conjugation.
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ion", "ity", "ed", "es", "s", "y")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


#: Function words carry no analytical content, so requiring them would make an
#: anchor fail on grammar rather than on substance -- "arrival versus decision
#: benchmark" must match an answer that writes "an arrival benchmark; a
#: decision-price benchmark would ...". Only these are dropped; every content
#: word of an anchor is still required.
_STOPWORDS = frozenset(
    """a an the and or nor but of in on to for with by at from as is are was
    were be been being it its this that these those than then so such not no
    versus vs against near over under into via per about""".split()
)


def _stems(text: str) -> set[str]:
    """Content-word stems of *text*, hyphens treated as spaces.

    Symmetric on both sides of the comparison, which is the point: an anchor
    written "square-root" has to match an answer that writes "square root",
    and an anchor written "decision" has to match "decision-price". Splitting
    only one side, or keeping the compound whole, fails one of those.
    """
    words = re.findall(r"[a-z0-9]+", _norm(text).replace("-", " "))
    return {_stem(w) for w in words if w not in _STOPWORDS}


def missing_mentions(pack: dict, text: str) -> list[str]:
    """``must_mention`` anchors the text failed to carry, in pack order.

    Every word of the anchor must appear in the answer, compared on stems and
    in any order -- not the anchor as one verbatim run of characters.

    That used to be a verbatim substring test, and it did not survive contact
    with a real teacher. Asked for "the square-root impact scaling", a correct
    answer wrote "Square-root impact scaling drives the cost" and failed on the
    missing definite article; asked for "participation near the ADV cap as the
    schedule constraint" it wrote "the schedule itself becomes the risk only
    when participation approaches the ADV cap" and failed on word order. Both
    answers made the point. The gate scored 0 of 3 on prose whose every
    keyword was present, and the repair loop then spent two more calls asking
    the model to fix writing that was already right -- so the instrument was
    selecting for verbatim quotation of the pack's own notes, which is the
    opposite of the desk voice the register asks for.

    Anchors are therefore short and distinctive (see the pack modules), and
    matching is on content: all terms present, any order, any inflection.
    """
    haystack = _stems(text)
    return [
        point
        for point in pack.get("must_mention") or []
        if not _stems(point) <= haystack
    ]


#: Negations that flip a forbidden claim into a warning against it. A desk
#: answer that says "do not assume the printable mid is achievable" is doing
#: exactly what the pack wants; scoring it as having asserted the claim -- which
#: a plain substring test does -- rejects the best answers for being explicit.
_NEGATIONS = (
    "not",
    "never",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "don't",
    "doesn't",
    "avoid",
    "reject",
    "refuse",
    "beware",
    "wrong",
    "false",
    "myth",
    "mistake",
    "fallacy",
    "trap",
    "assume",
)

#: How many words before the claim are inspected for a negation. Wide enough
#: for "do not make the mistake of assuming that X", narrow enough that an
#: unrelated "not" two sentences earlier cannot launder an assertion.
_NEGATION_WINDOW = 8


def forbidden_hits(pack: dict, text: str) -> list[str]:
    """``forbidden_claims`` the text asserted anyway, in pack order.

    A claim counts as asserted only where it appears *without* a negation in
    the words immediately before it. The contract in the teacher's system turn
    says not to assert a forbidden claim "even as a hedge or a disclaimer", and
    that is still enforced -- an unnegated mention is a hit wherever it occurs.
    What is no longer a hit is the explicit warning, which is the form a senior
    desk answer actually takes.
    """
    haystack = _norm(text)
    hits: list[str] = []
    for claim in pack.get("forbidden_claims") or []:
        needle = _norm(claim).rstrip(".")
        if not needle:
            continue
        for match in re.finditer(re.escape(needle), haystack):
            window = haystack[: match.start()].split()[-_NEGATION_WINDOW:]
            if not any(word.strip(",.;:") in _NEGATIONS for word in window):
                hits.append(claim)  # asserted plainly at least once
                break
    return hits


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
