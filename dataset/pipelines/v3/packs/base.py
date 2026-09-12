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
from dataclasses import asdict, dataclass, field
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
    #: The one official value of each named quantity (amendment §D). This, not
    #: ``allowed_numbers``, is what an *answer* is graded against: the allow-list
    #: is a union that necessarily carries the question's own printed roundings,
    #: and a union cannot tell 372.6 from the 373 the question prints beside it.
    canonical: dict = field(default_factory=dict)
    #: Every spelling of a canonical quantity the *question* prints, by the
    #: same key. Two jobs, and both matter:
    #:
    #: 1. they are authorised in ``allowed_numbers``, which is what makes a
    #:    question legal -- a pack whose own question quotes a token its
    #:    allow-list does not carry is a corrupt pack, and the registry sweep
    #:    fails on exactly that;
    #: 2. the *whole-number* ones are what ``rounding_drift`` reads: a question
    #:    that opens "behind by 373 bp" of a 372.6 has rounded away a digit the
    #:    answer is supposed to report.
    #:
    #: Most entries only ever do the first job (a percent spelling of a
    #: fraction is not drift, and the gate says so). Declaring them anyway is
    #: how the pack states what its question is allowed to print, in one place,
    #: instead of leaving it to an untyped tuple of extras nobody could read.
    aliases: dict = field(default_factory=dict)
    #: How the desk spells a figure -- ``"430,567"``, not ``430567.0``. Carried
    #: so the repair turn can quote the spelling it wants rather than asking the
    #: teacher to guess at house style.
    display: dict = field(default_factory=dict)
    #: Anchors and claims whose truth is an *inequality in this pack*, kept
    #: after they are folded (:func:`fold_conditionals`) so a reader can see
    #: why a row was told what it was told. Entries are
    #: ``{"when": "<quantity> <op> <number|quantity>", "mention"|"claim": str}``.
    #:
    #: The amendment's §A.2: ``must_mention`` was a slogan list, so a TCA pack
    #: whose participation is half the cap still had an answer enacting "the
    #: schedule itself becomes the risk", and a VaR pack with a negative daily
    #: mean still had one calling it a drift in its favour. A point whose truth
    #: depends on a number is a *test*, and the pack is where the number is.
    conditional_mentions: list[dict] = field(default_factory=list)
    conditional_forbids: list[dict] = field(default_factory=list)
    #: The question this pack *cannot* answer, and the quantity it is missing.
    #:
    #: The abstention lane's reason for existing, and until this existed it had
    #: none: abstention rows were rendered over the pack's own question, which
    #: the pack answers in full, so every one of them came back as an analysis
    #: with a caveat welded on ("...and there is no decision price here"). That
    #: is the opposite of the behaviour the record type is for -- 360 rows of
    #: it would teach a student to hedge rather than to decline.
    #:
    #: So the refusal is a *property of the pack*: a question about a quantity
    #: the fact computer genuinely does not produce, and the name of that
    #: quantity so the gate can ask whether the answer named it too.
    abstention_question: str = ""
    abstention_missing: str = ""
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

    def canonical_values(self) -> list[float]:
        """The numeric authority an answer is graded against, sorted.

        Falls back to ``allowed_numbers`` for a pack that declares no canonical
        map -- a test fixture, or a computer written before the amendment. The
        fallback is the old behaviour exactly, so an undeclared pack is graded
        no more loosely than it was, only no more tightly either.
        """
        if not self.canonical:
            return list(self.allowed_numbers)
        return sorted({round(float(v), 12) for v in _scalars(self.canonical)})

    def alias_tokens(self) -> dict[str, list[str]]:
        """``{quantity: [token, ...]}`` -- the spellings an answer may not use.

        Only quantities that *have* a canonical value contribute: an alias
        without an official number behind it is just a number, and refusing it
        would be refusing the pack's own arithmetic.
        """
        return {
            key: [str(token) for token in _listify(value)]
            for key, value in self.aliases.items()
            if key in self.canonical
        }


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


def _listify(value: object) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalars(node: object) -> list[float]:
    """Every non-boolean number reachable in *node*, depth first."""
    out: list[float] = []

    def walk(item: object) -> None:
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                walk(value)

    walk(node)
    return out


def flatten_quantities(node: object, prefix: str = "") -> dict:
    """``{dotted.name: value}`` for every scalar in a pack's inputs/computed.

    The name is the audit's half of :data:`FactPack.canonical`: a bare list of
    numbers cannot say *which* quantity 372.6 is, and the whole point of the
    canonical map is that a rounding drift can be reported as "active_bp", not
    as "some number near 373".
    """
    flat: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(flatten_quantities(value, f"{prefix}{key}."))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            flat.update(flatten_quantities(value, f"{prefix}{index}."))
    elif isinstance(node, bool):
        pass  # a flag is not a quantity
    elif isinstance(node, (int, float)):
        flat[prefix.rstrip(".")] = round(float(node), 12)
    return flat


def assemble_contract(
    inputs: dict,
    computed: dict,
    *,
    aliases: dict | None = None,
    display: dict | None = None,
    canonical_extra: dict | None = None,
) -> dict:
    """The four number fields of a pack, built once from one declaration.

    Returns ``{"canonical", "aliases", "display", "allowed_numbers"}`` ready to
    splat into :class:`FactPack`. ``canonical`` is every scalar of ``inputs``
    and ``computed`` under its dotted name -- one value per quantity, which is
    the amendment's whole rule -- while ``allowed_numbers`` stays the *union*
    the rest of the pipeline already reads: the oracle returns pack figures as
    tool results, the exam composer derives distractors from them and the
    implementation suite pins them into generated tests, and narrowing that set
    here would change four contracts to fix one.

    The split is the point. The union decides what may appear *anywhere*; the
    canonical map decides what an answer may claim is the number.

    ``canonical_extra`` is for the quantities a scenario has that are neither
    an input nor a computed result: the confidence levels a VaR question asks
    for, the sensitivity band a DCF question flexes, the participation cap a
    TCA question measures against. They used to ride an untyped ``extra``
    tuple into ``allowed_numbers`` and nowhere else, which was fine while the
    union was the answer gate's authority and became a bug the moment it was
    not -- a risk answer naming its own 95% and 99% VaRs read as inventing two
    numbers. They are quantities; they get names.
    """
    aliases = dict(aliases or {})
    display = dict(display or {})
    canonical = flatten_quantities(inputs)
    canonical.update(flatten_quantities(computed))
    canonical.update(flatten_quantities(canonical_extra or {}))
    alias_values = [
        float(token if not isinstance(token, str) else str(token).replace(",", ""))
        for tokens in aliases.values()
        for token in _listify(tokens)
    ]
    return {
        "canonical": canonical,
        "aliases": aliases,
        "display": display,
        "allowed_numbers": assemble_numbers(
            inputs,
            computed,
            extra=(*alias_values, *_scalars(canonical_extra or {})),
        ),
    }


#: The comparisons a ``when`` clause may make. Deliberately three tokens and
#: nothing else: a pack condition has to be readable in the pack, checkable by
#: a reader with the canonical map in front of them, and impossible to turn
#: into a second language nobody debugs. Anything an expression evaluator would
#: buy here is a quantity the computer can compute and name instead.
_COMPARATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _resolve(token: str, quantities: dict) -> float:
    """A ``when`` operand: a quantity of this pack, or a plain number."""
    if token in quantities:
        return float(quantities[token])
    try:
        return float(token)
    except ValueError as exc:
        raise PackError(
            f"condition names {token!r}, which is neither a quantity of this "
            f"pack nor a number (known: {', '.join(sorted(quantities)[:8])}...)"
        ) from exc


def holds(when: str, quantities: dict) -> bool:
    """Is *when* true of this pack? ``PackError`` if it cannot be read.

    Loud on a typo on purpose. A condition that silently evaluated false would
    quietly drop the forbid it carries, and a forbid nobody notices missing is
    the exact failure mode §A.2 exists to end.
    """
    parts = str(when).split()
    if len(parts) != 3 or parts[1] not in _COMPARATORS:
        raise PackError(
            f"condition {when!r} is not '<quantity> <op> <number|quantity>' "
            f"(ops: {', '.join(sorted(_COMPARATORS))})"
        )
    left, op, right = parts
    return _COMPARATORS[op](_resolve(left, quantities), _resolve(right, quantities))


def fold_conditionals(
    inputs: dict,
    computed: dict,
    *,
    must_mention: Iterable[str] = (),
    forbidden_claims: Iterable[str] = (),
    conditional_mentions: Iterable[dict] = (),
    conditional_forbids: Iterable[dict] = (),
) -> dict:
    """The four contract-of-points fields of a pack, conditions already run.

    Returns ``{"must_mention", "forbidden_claims", "conditional_mentions",
    "conditional_forbids"}`` ready to splat into :class:`FactPack`. The
    unconditional lists come first and the conditional entries are appended in
    declaration order, deduplicated, so a pack reads as one list of points
    whatever produced them -- the teacher never sees a condition, only the
    points its own numbers make true, which is the whole of §A.2.

    The declarations survive on the pack because a row is audited long after
    it is written, and "why was this answer forbidden that sentence" is a
    question the row itself should be able to answer.
    """
    quantities = flatten_quantities(inputs)
    quantities.update(flatten_quantities(computed))
    mentions = list(must_mention)
    claims = list(forbidden_claims)
    declared_mentions = [dict(entry) for entry in conditional_mentions]
    declared_forbids = [dict(entry) for entry in conditional_forbids]
    for entry in declared_mentions:
        if holds(entry["when"], quantities) and entry["mention"] not in mentions:
            mentions.append(entry["mention"])
    for entry in declared_forbids:
        if holds(entry["when"], quantities) and entry["claim"] not in claims:
            claims.append(entry["claim"])
    return {
        "must_mention": mentions,
        "forbidden_claims": claims,
        "conditional_mentions": declared_mentions,
        "conditional_forbids": declared_forbids,
    }


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
