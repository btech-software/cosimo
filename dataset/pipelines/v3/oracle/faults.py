"""The fault layer: which agentic jobs get a dirty oracle, and which fault.

Spec §5.8 splits the responsibilities exactly here: *"a planner template only
chooses which faults to inject ... It does not write the assistant
utterances."* This module is that planner, and it is a pure function of the
job's coordinates -- no RNG, no wall clock, no dispatch order. A replay of
``schedule_of(work_type, family, variant)`` on any machine reproduces the
exact same dirty-program schedule, which is what lets ``verify_v3`` re-run
the oracle byte for byte and call the trajectory a lie when the bytes
disagree.

The five faults are the spec's list, verbatim, and each one is a *shape* the
serving oracle could honestly return -- this layer does not simulate outages,
it uses the same response builders the clean path uses, degraded:

* ``empty_result``  -- the query is well formed and the data simply is not there;
* ``stale_asof``    -- the block is real but its as-of stamp pre-dates the pack's;
* ``wrong_ticker``  -- the block carries another entity's identifier;
* ``rate_limit``    -- "too many requests"; the second identical call succeeds;
* ``schema_drift``  -- the payload's field names have drifted from the registry.

What the corpus is teaching is the *response*: a tool result is evidence, not
gospel, and an answer that quotes a faulted block as fact is a dead letter,
not a near miss. That judgement is enforced mechanically in
``verification/agentic.py`` (per-fault acknowledgement vocabularies plus the
rate-limit re-issue rule), never by a vibes-based judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .. import config

#: Rotation order for the faulted slice: ``(variant // stride) % 5`` walks it,
#: so consecutive faulted siblings cycle the taxonomy and the 20-row gate's
#: four faulted conversations cover four *distinct* faults, not four copies
#: of whichever injected the deepest.
FAULTS = ("empty_result", "stale_asof", "wrong_ticker", "rate_limit", "schema_drift")

#: The stale stamp. A date, not a number: quoting it stays legal (it rides in
#: a tool result, and the spec's grounding rule is "pack ∪ tool results"),
#: while ignoring it is what the acknowledgement check below exists to catch.
STALE_AS_OF = "2019-01-01"

#: Every fault must be *seen* in the final answer: at least one token from
#: the fault's vocabulary has to survive into the last assistant turn. They
#: are word-lists, deliberately loose (case- and punctuation-insensitive on
#: purpose -- the desk says "no data", the gate must read it either way).
FAULT_ACK = {
    "empty_result": ("no data", "empty", "not returned", "no rows", "missing"),
    "stale_asof": ("stale", "outdated", "as of", "old", "pre-dates"),
    "wrong_ticker": (
        "mismatch",
        "another entity",
        "different entity",
        "not the",
        "wrong",
        "unexpected identifier",
    ),
    "rate_limit": (
        "rate limit",
        "rate-limit",
        "unavailable",
        "retry",
        "re-issued",
        "reissued",
        "without the call",
    ),
    "schema_drift": (
        "schema",
        "renamed",
        "unexpected field",
        "unknown field",
        "drift",
        "unrecognised",
    ),
}


class FaultError(ValueError):
    """An fault the planner was asked to inject is not in the taxonomy."""


#: The one rule by which renderer, harness and board all agree on *which*
#: argument a fault speaks for -- the identifier a call asks about. Keeping it
#: a function of the arguments (not of any call site's local idiom) is what
#: lets the byte-exact replay compare apples with apples.
def requested_for(arguments: dict) -> str:
    for field in ("symbol", "portfolio_id", "book_id"):
        value = arguments.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


@dataclass(frozen=True)
class Schedule:
    """What this conversation is going to happen to the teacher."""

    mode: str  # "no_call" | "clean" | "faulted"
    fault: str | None = None

    def as_render_field(self) -> dict:
        """The ``verification.render`` slice that makes the run replayable.

        ``verify_v3`` re-derives the schedule from the coordinates *and* reads
        this field; agreement is required. Tampering with either side breaks
        the byte-for-byte tool replay, so the pair is tamper-evident.
        """
        return {"mode": self.mode, "fault": self.fault}


def schedule_of(rank: int) -> Schedule:
    """The planner's only output for one job: clean, no-call, or faulted.

    ``rank`` is the job's ordinal in ``render.agentic.select_agentic_jobs`` --
    the shared selector's verdict, so renderer, harness and any later audit
    count the *same* ordinals, and "every fifth job" becomes arithmetic
    instead of folklore: within any run of five consecutive ranks exactly one
    is a no-call conversation and exactly one carries a fault, which holds
    the plan-wide mix at 20% for both slices -- inside the spec's 15-20%
    band for no-call -- and makes the PR3 gate's "20 conversations, 4
    faulted, 4 no-call" assertable rather than hoped for. The residues are
    disjoint because a no-call conversation has no tool result to fault:
    scheduling a fault there would plan an event that cannot happen.
    """
    if rank < 0:
        raise FaultError(f"rank {rank!r} is not a selector ordinal")
    residue = rank % config.AGENTIC_STRIDE
    if residue == config.AGENTIC_NO_CALL_RESIDUE:
        return Schedule("no_call")
    if residue == config.AGENTIC_FAULT_RESIDUE:
        return Schedule(
            "faulted", FAULTS[(rank // config.AGENTIC_STRIDE) % len(FAULTS)]
        )
    return Schedule("clean")


def canonical(result: dict) -> str:
    """The one byte form of a tool result, shared by loop, gate and board.

    ``sort_keys`` + fixed separators + ``ensure_ascii=False``: the renderer
    appends this string to the conversation, the fixture hashed its reply on
    it, and ``verify_v3`` replays the oracle against it. Three call sites,
    one dump policy, or the replay is a byte-diff away from a false alarm.
    """
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _renamed(key: str) -> str:
    """A field name as it appears after the schema drifted: last letter cut."""
    return key[:-1] if len(key) > 5 else key


def _pollute_keys(node: dict) -> dict:  # the faulted block is built, not parsed
    return {
        (_renamed(key) if isinstance(key, str) else key): (
            _pollute_keys(value) if isinstance(value, dict) else value
        )
        for key, value in node.items()
    }


def _rotate_entity(payload: dict, requested: str) -> dict:
    """Return the block with another entity's identifier grafted on.

    The identifier fields of the family are ``*symbol``, ``*ticker``,
    ``*book_id``; the first one found gets the letter appended -- an
    off-by-one rotation through the alphabet, so a faulted block is always
    *recognisably* someone else's data rather than a random string.
    """
    for key in list(payload):
        low = key.casefold()
        if low.endswith("symbol") or low.endswith("ticker") or low.endswith("book_id"):
            value = payload[key]
            if isinstance(value, str) and value:
                shifted = chr(
                    (ord(value[-1].casefold()) - ord("a") + 1) % 26 + ord("a")
                )
                payload[key] = value[:-1] + shifted
                payload["requested"] = requested
                payload["warning"] = "identifier in the block is not the one requested"
                return payload
    payload["requested"] = requested
    payload["warning"] = "block belongs to another entity"
    return payload


def inject(name: str, result: dict, fault: str | None, *, requested: str = "") -> dict:
    """The clean block, as the dirty oracle would have returned it.

    Pure and total: the same ``(name, result, fault, requested)`` always
    builds the same bytes. ``rate_limit`` is *state* dependent at the call
    site (the loop counts identical calls; only the first one is refused),
    which the caller expresses by passing ``result`` already chosen -- this
    function only renders what it is handed.
    """
    if fault is None:
        return result
    if fault not in FAULTS:
        raise FaultError(f"unknown fault {fault!r} (taxonomy: {', '.join(FAULTS)})")
    polluted = {"tool": name, "__fault__": fault}
    if fault == "empty_result":
        polluted.update({"rows": [], "series": [], "note": "no rows returned"})
    elif fault == "stale_asof":
        body = dict(result)
        body["as_of"] = STALE_AS_OF
        body["stale"] = True
        polluted.update(body)
    elif fault == "wrong_ticker":
        polluted.update(_rotate_entity(dict(result), requested))
    elif fault == "rate_limit":
        polluted.update(
            {
                "error": "too many requests",
                "status": "unavailable",
                "note": "the same call may be re-issued",
            }
        )
    elif fault == "schema_drift":
        polluted.update(_pollute_keys(dict(result)))
        polluted["note"] = "field names in this block are not the registered ones"
    return polluted
