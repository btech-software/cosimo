"""The agentic contract, executable (analysis spec §5.8; arch spec §5.4, §6).

Two ideas carry this whole module, and both are the spec's words, not ours:

* *"validate tool_calls against cosimo.tools.registry"* -- a trajectory is
  only as real as the server's ability to have produced it, so every call is
  checked against the same registry the serving app registers, and every
  tool result is **re-executed** against the recomputed pack. A stored
  trajectory is a claim about a computation; here it is audited as one.
  Byte-for-byte, with the fault designator applied exactly as the render loop
  applies it (``expected_tool_contents`` is shared, so gate and loop cannot
  disagree about which call was the dirty one -- the v1/v2 disease again).
* *"every number in the final assistant turn is in the union of fact pack
  and tool results"* -- the §5.8 verify rule, verbatim, as a numeric-subset
  check: nothing about it is a heuristic, and a number quoted from a
  *faulted* block is legal precisely because the block is a tool result;
  what makes a fault survivable is not the whitelist but the acknowledgement
  clause below.

The disciplines that make the mix educational rather than theatrical are
mechanical, in the order the gate reports them:

1. **shape** -- roles in order, calls paired with results, a final plain turn;
2. **budget** -- at most ``AGENTIC_MAX_TOOL_CALLS`` calls, the conversation
   inside the message band (the no-call shape is its own, two-turn shape);
3. **mode** -- a no-call conversation that called is waste, and is said so;
   a calling conversation that never called is not the loop it claims;
4. **replay** -- stored tool bytes equal the re-executed oracle's bytes;
5. **grounding** -- the final turn's numbers ride in the pack ∪ the results;
6. **acknowledgement** -- an injected fault must be *named* in the final
   turn (a trajectory that quotes a stale block as fact is a dead letter, not
   a near miss) and a rate-limited call must have been re-issued or waived
   in words;
7. **the shared prose clauses** -- must_mention coverage, forbidden_claims,
   no exam-only tag: one contract, every kind of row.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cosimo.tools.wire import TOOL_CALL_OPEN, parse_tool_calls  # noqa: E402
from verification.gates import FINAL_ANSWER_TAG  # noqa: E402

from .. import config  # noqa: E402
from ..oracle import faults, runtime  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402
from .invented_numbers import read_tokens  # noqa: E402
from .prose import forbidden_hits, missing_mentions  # noqa: E402
from .prose import whitelist_for as _pack_whitelist  # noqa: E402


class TrajectoryError(ValueError):
    """A trajectory so malformed that grading it would be guessing."""


#: Tag on every violation the *replay* produced, so the board can file it
#: under axis 10 without string archaeology and the repair prompt can read
#: the same sentence the operator would. One constant, both call sites.
REPLAY_TAG = "replay: "
#: Ditto for the §5.8 grounding subset check: the spec's own axis 3 is "the
#: invented-number subset check", and its agentic reading ("the union of
#: fact pack and tool results") is that same axis with a wider authority --
#: so the board files these under 3, where an audit already knows to look.
GROUNDING_TAG = "grounding: "


def call_digest(name: str, arguments: dict) -> str:
    """The identity of a call: what a re-issue must match to be the same ask."""
    return faults.canonical({"arguments": arguments, "name": name})


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def calls_in(messages: list[dict]) -> list[dict]:
    """The ordered calls of a conversation, ``{"name", "arguments"}`` each.

    Accepts both wire spellings of ``arguments`` (an OpenAI JSON string or a
    local dict) and returns dicts with a parsed ``arguments`` mapping; a call
    whose arguments will not parse is returned with a ``"__bad__"`` marker so
    the gate reports it instead of crashing on it -- an unrarsable call is a
    violation with a name, not an exception with a stack.
    """
    out: list[dict] = []
    for message in messages:
        for call in message.get("tool_calls") or ():
            function = (call or {}).get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
            if arguments is None:
                arguments = {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    out.append(
                        {
                            "name": name,
                            "arguments": {},
                            "__bad__": "arguments are not valid json",
                        }
                    )
                    continue
            if not isinstance(arguments, dict):
                out.append(
                    {
                        "name": name,
                        "arguments": {},
                        "__bad__": "arguments are not an object",
                    }
                )
                continue
            out.append({"name": name, "arguments": arguments})
    return out


def tool_contents(messages: list[dict]) -> list[str]:
    """The stored bytes of every tool result, in conversation order."""
    return [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]


def expected_tool_contents(
    pack: dict, calls: list[dict], *, mode: str, fault: str | None
) -> list[str]:
    """The bytes the oracle *would have* returned, call by call.

    The one authority on which call was faulted: the first, and for
    ``rate_limit`` only the first -- a re-issue of the identical call is the
    refusal being withdrawn, and must come back clean. The render loop calls
    this very function, so loop and gate agree by construction rather than by
    two transcriptions of a rule.
    """
    if fault is not None and fault not in faults.FAULTS:
        raise TrajectoryError(f"unknown fault {fault!r}")
    seen: set[str] = set()
    contents: list[str] = []
    for index, call in enumerate(calls):
        if "__bad__" in call:
            contents.append(f"__unreplayable__:{call['__bad__']}")
            continue
        clean = runtime.execute(
            runtime.Call(name=call["name"], arguments=call["arguments"]), pack
        )
        applied: str | None = None
        digest = call_digest(call["name"], call["arguments"])
        if mode == "faulted" and fault is not None:
            if index == 0:
                applied = fault
            elif fault == "rate_limit" and digest in seen:
                applied = None
        if applied is not None:
            dirty = faults.inject(
                call["name"],
                clean,
                applied,
                requested=faults.requested_for(call["arguments"]),
            )
            contents.append(faults.canonical(dirty))
        else:
            contents.append(faults.canonical(clean))
        seen.add(digest)
    return contents


def _tool_tokens(messages: list[dict]) -> frozenset[str]:
    """Every numeric token printed inside any tool result -- fair to quote.

    Extraction runs over the *raw* result bytes, not the parsed values: what
    the teacher was shown is the authority, glyph for glyph, which is exactly
    the reach of the spec's "union of fact pack and tool results".

    Read with :func:`read_tokens`, the same reader the answer is read with. A
    tool result dated ``2026-01-01`` is three tokens and which three depends
    on whether a hyphen counts as a sign; two readers here means a set that
    authorises spellings the answer cannot produce.
    """
    tokens: set[str] = set()
    for content in tool_contents(messages):
        tokens.update(read_tokens(content))
    return frozenset(tokens)


def trajectory_violations(
    pack: dict, messages: list[dict], *, mode: str, fault: str | None
) -> list[str]:
    """Every way this conversation breaks the agentic contract, as sentences.

    Empty means shippable. Written to be read twice: once by the repair
    prompt (the teacher must find its own error in them) and once by whoever
    post-mortems the dead letter.
    """
    violations: list[str] = []
    # The shipped transcript opens on the user's goal, not on a system turn.
    # The factory's ``AGENTIC_SYSTEM`` is a brief -- it announces the mode and
    # quotes the number policy -- and the amendment's §A keeps it off every
    # trainable surface. A *system* turn here is therefore itself a finding:
    # this gate runs over the row that shipped, and the row that shipped must
    # not carry the factory's instructions.
    if not messages or messages[0].get("role") != "user":
        return ["the conversation does not open with the user's goal"]
    if any(m.get("role") == "system" for m in messages):
        violations.append(
            "the conversation carries a system turn -- the factory's brief is "
            "not part of the trajectory the student is trained on (§A)"
        )
    body = messages[1:]
    if not body:
        return violations + ["the conversation has no exchanges at all"]
    for position, message in enumerate(body):
        if message.get("role") not in ("assistant", "tool"):
            violations.append(
                f"exchange {position + 1} has role {message.get('role')!r}; "
                "after the goal only assistant and tool turns belong here"
            )
    last = body[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        violations.append("the conversation does not close with a plain assistant turn")
    final = str(last.get("content") or "")
    if not final.strip():
        violations.append("the final assistant turn is empty")

    # -- 2/3: budget and mode discipline ------------------------------------
    calls = calls_in(body)
    for call in calls:
        if "__bad__" in call:
            violations.append(f"malformed tool call: {call['__bad__']}")
    bad_names = [c["name"] for c in calls if c["name"] not in runtime.SCHEMA_NAMES]
    if bad_names:
        violations.append(
            "calls name tools outside the registry: "
            + ", ".join(repr(n) for n in bad_names)
        )
    for call in calls:
        if "__bad__" in call or call["name"] not in runtime.SCHEMA_NAMES:
            continue
        errors = runtime.validate_call(call["name"], call["arguments"])
        if errors:
            violations.append(
                f"call to {call['name']!r} violates its schema: " + "; ".join(errors)
            )
    tool_turns = tool_contents(body)
    if len(tool_turns) != len(calls):
        violations.append(
            f"{len(calls)} calls were made but {len(tool_turns)} results were "
            "returned -- a call without a result is a lie about the server"
        )
    # The band counts the *exchanges*: the user's goal and everything after
    # it -- "6-16 turns" with a two-call loop (user, call, result, call,
    # result, answer) landing exactly on the floor. It used to subtract the
    # system turn; there is no longer one to subtract, so the arithmetic is
    # the same number reached without the correction.
    n_exchange = len(messages)
    if mode == "no_call":
        if calls or tool_turns:
            violations.append(
                "this job was planned answer-from-the-pack (no_call) and a tool "
                "was called anyway: the pack already carried the data, so the "
                "call was waste -- spec §5.8 records it as a defect"
            )
    else:
        if not calls:
            violations.append(
                "no tool was called at all, yet this job was planned as a "
                "looping conversation -- if the pack truly sufficed, the plan "
                "said so; answer again or be dead-lettered"
            )
        if len(calls) > config.AGENTIC_MAX_TOOL_CALLS:
            violations.append(
                f"{len(calls)} calls exceed the budget of "
                f"{config.AGENTIC_MAX_TOOL_CALLS}"
            )
        if not config.AGENTIC_MIN_MESSAGES <= n_exchange <= config.AGENTIC_MAX_MESSAGES:
            violations.append(
                f"the conversation runs {n_exchange} exchanges, outside the "
                f"{config.AGENTIC_MIN_MESSAGES}-{config.AGENTIC_MAX_MESSAGES} band "
                "(a 'real multi-step loop, 6-16 turns' is what §5.8 promised)"
            )

    # -- 4: replay: the oracle is the only writer ----------------------------
    if not any("__bad__" in c for c in calls) and not bad_names:
        try:
            expected = expected_tool_contents(pack, calls, mode=mode, fault=fault)
        except runtime.OracleError as exc:
            violations.append(f"{REPLAY_TAG}oracle could not re-execute a call: {exc}")
        else:
            for position, (got, want) in enumerate(zip(tool_turns, expected)):
                if got != want:
                    violations.append(
                        f"{REPLAY_TAG}tool result {position + 1} does not match "
                        "the oracle re-executed against the recomputed pack -- "
                        f"stored bytes {got[:80]!r} vs {want[:80]!r}"
                    )

    # -- 5: grounding on pack ∪ results --------------------------------------
    allowed = [float(x) for x in pack.get("allowed_numbers") or []]
    if not allowed:
        violations.append(
            "the recomputed pack carries no allowed_numbers; grounding cannot "
            "be graded against an empty authority -- fail closed"
        )
    else:
        whitelist = frozenset(_pack_whitelist(pack)) | _tool_tokens(body)
        offenders = invented_numbers(final, allowed, whitelist)
        if offenders:
            violations.append(
                f"{GROUNDING_TAG}final answer quotes numbers found neither in "
                "the pack nor in any tool result: "
                + ", ".join(repr(t) for t in offenders)
            )

    # -- 6: the fault must be seen, not steered around -----------------------
    if fault is not None:
        haystack = _norm(final)
        if not any(token in haystack for token in faults.FAULT_ACK[fault]):
            violations.append(
                f"the run injected {fault!r} and the final answer never names "
                f"it (say one of: {', '.join(repr(t) for t in faults.FAULT_ACK[fault])}); "
                "a fault quoted as fact is a dead letter, not a near miss"
            )
        if fault == "rate_limit" and calls:
            first = call_digest(calls[0]["name"], calls[0]["arguments"])
            reissued = any(
                call_digest(c["name"], c["arguments"]) == first for c in calls[1:]
            )
            if not reissued and "without the call" not in haystack:
                violations.append(
                    "the first call was rate-limited; a looping desk re-issues "
                    "the identical call once, or says in words that it proceeds "
                    'without it ("without the call")'
                )

    # -- 7: the clauses every row shares --------------------------------------
    for point in missing_mentions(pack, final):
        violations.append(f"must_mention not covered: {point!r}")
    for claim in forbidden_hits(pack, final):
        violations.append(f"forbidden claim asserted: {claim!r}")
    if FINAL_ANSWER_TAG.casefold() in _norm(final):
        violations.append(
            f"{FINAL_ANSWER_TAG!r} is an exam contract, not desk prose (axis 5); drop it"
        )
    if TOOL_CALL_OPEN in final or parse_tool_calls(final):
        violations.append(
            "the final turn still carries tool-call markup -- a completion "
            "that ends in a call was never a completion; the server parses "
            "markers, the desk reads prose"
        )
    return violations


def gate_violations(
    pack: dict, messages: list[dict], *, mode: str, fault: str | None
) -> list[str]:
    """Alias of :func:`trajectory_violations` mirroring the prose gate's name.

    One policy, three call sites -- the render loop, ``verify_v3``, and the
    fixture harness all read the contract through this one door.
    """
    return trajectory_violations(pack, messages, mode=mode, fault=fault)
