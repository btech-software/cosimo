"""The student row: the only object verify, publish and prepare ever see (§A).

Every renderer used to write *one* object and it was the teacher's transcript
with an ``answer`` key bolted on: ``messages`` opened with the factory's own
system turn, the user turn was the brief (fact pack as JSON, contract, word
budget), and the assistant turn was the answer. That row trained fine, which
was the problem -- what it trained was the labelling protocol. A model fed
those bytes learns that the shape of a question is a JSON contract naming
``allowed_numbers``, and the served product never asks a question that way.

So a render now produces two objects and only one of them is trainable:

* the **teacher log** -- full messages, think text, usage, the repair ladder --
  written to ``teacher_logs/<kind>/<id>.json`` and only when an operator asks
  (``COSIMO_V3_KEEP_TEACHER_MESSAGES=1``). Debug, gitignored, never a shard;
* the **student row** -- this module -- carrying the question the desk would
  actually ask, the answer, and enough provenance to re-derive both.

Four fields are new and each closes a hole the old row left open:

``family`` / ``holdout``
    were recoverable but never *stated*: ``family`` by slicing ``scenario_id``
    at the length of ``work_type`` (which contains dots, so a naive split was
    always wrong), and ``holdout`` not at all -- the row did not know whether
    it was allowed to train. Both are now written down, which is what lets
    ``01_prepare_data`` refuse a leak instead of re-deriving one.

``fact_pack``
    the pack as a JSON *object*, not a string and not absent. The eval sidecar
    needs it to score an invented-number rate against the row's own contract,
    and a corpus whose numbers can only be checked by re-running the generator
    is a corpus whose numbers are checked once.

``verified``
    the explicit claim. ``is_verified`` in the harness used to read "carries a
    verification stamp" for a v3 row, which is a proxy: a row that failed its
    gate and was written anyway would carry a stamp too. The row now says so.
"""

from __future__ import annotations

import re

from . import config

#: Everything a stored fact-pack line carries *around* the pack proper.
#: Stripped before the pack is handed to a brief builder or written onto a
#: student row: the teacher must see exactly ``FactPack.to_dict()``,
#: byte-for-byte the dict the replay fixture hashed it as.
PACK_ENVELOPE = ("id", "verification")

#: Vendor think channels that leak into ``content`` instead of staying in
#: ``reasoning_content``. Non-greedy, dot-matches-newline, and an *unclosed*
#: opener is stripped to end of text: a truncated chain of thought is the one
#: case where the tag is there and the closer never arrives, and leaving that
#: in the answer would ship the whole reasoning trace as the target.
_THINK_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning)\s*>.*?(?:<\s*/\s*\1\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)


def visible_answer(text: str | None) -> str:
    """The assistant text a student may be trained on, and nothing else.

    Strips any inline think block and then the leading blank lines that a
    reasoning teacher leaves behind when its chain of thought ends -- the
    committed example rows all begin ``"\\n\\nSell 430,567 ..."``, and a target
    that opens on two newlines teaches the model to open on two newlines.
    """
    return _THINK_BLOCK.sub("", str(text or "")).strip()


def strip_envelope(pack_line: dict) -> dict:
    """The pack proper, without the shard line's ``id``/``verification``."""
    return {k: v for k, v in pack_line.items() if k not in PACK_ENVELOPE}


def family_of(pack_or_row: dict) -> str:
    """The scenario family of a pack line or a row.

    A length slice, never ``split(".")``: ``work_type`` contains dots
    (``valuation.equity.dcf``) and ``scenario_id`` is ``f"{work_type}.{family}"``,
    so splitting on the separator returns the first token of the work type.
    """
    work_type = str(pack_or_row.get("work_type") or "")
    scenario_id = str(pack_or_row.get("scenario_id") or "")
    prefix = f"{work_type}."
    if work_type and scenario_id.startswith(prefix):
        return scenario_id[len(prefix) :]
    return scenario_id


def carries_teacher_brief(row: dict) -> bool:
    """True when any byte of this row is the factory talking to itself.

    The fingerprint is :data:`config.TEACHER_FINGERPRINT`, which appears in the
    system turn of every brief the teacher module builds. Checked over the
    whole serialised row rather than over ``messages`` alone: the leak the
    amendment is closing is "the protocol reached a training target", and a
    brief pasted into ``question`` leaks exactly as hard as one left in
    ``messages``.
    """
    return config.TEACHER_FINGERPRINT in _flatten(row)


def _flatten(node: object) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_flatten(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return " ".join(_flatten(v) for v in node)
    return ""


def student_row(
    *,
    row_id: str,
    kind: str,
    pack: dict,
    holdout: bool,
    answer: str,
    stamp: dict,
    teacher: dict | None,
    render: dict,
    attempts: int,
    invented: list[str] | None = None,
    missing: list[str] | None = None,
    register_ok: bool = True,
    extra: dict | None = None,
) -> dict:
    """One trainable row, in the §A schema. ``extra`` carries per-kind fields.

    ``question`` is ``pack["question"]`` and only that -- never the brief, never
    the brief's ``allowed_numbers`` block pasted in front of it. The kinds whose
    prompt genuinely differs from the pack question (an exam item shows four
    labelled options) say so in a named field of their own via ``extra``, so
    the difference is a column an auditor can see rather than an index into a
    message list.

    ``answer`` is stored **verbatim**. Running :func:`visible_answer` here
    looked tidier and was wrong: the exam and implementation lanes are
    *composed*, their answers are bytes the verify board re-derives from the
    pack and compares exactly, and trimming a trailing newline off a reference
    implementation reads to that board as a tamper. So the transform belongs at
    the two call sites whose answer is a teacher's text, and those call it.
    """
    return {
        "id": row_id,
        "record_type": kind,
        "work_type": pack["work_type"],
        "scenario_id": pack["scenario_id"],
        "family": family_of(pack),
        "holdout": bool(holdout),
        "variant": pack["variant"],
        "register": pack["register"],
        "question": pack["question"],
        "answer": answer,
        "fact_pack": pack,
        "verified": True,
        "verification": {
            "computed_by": stamp.get("computed_by", "unknown"),
            "pack_seed": stamp.get("pack_seed", f"{pack['seed']:016x}"),
            "teacher": teacher,
            "attempts": attempts,
            # The gate's own verdict, recorded rather than implied. All three
            # are empty/true on a row that shipped -- that is what shipping
            # means -- and writing them down is what lets the eval sidecar and
            # the publish audit read a row's compliance without re-running the
            # gate over a pack they would have to recompute first.
            "invented_numbers": list(invented or ()),
            "missing_mentions": list(missing or ()),
            "register_ok": bool(register_ok),
            "render": render,
        },
        **(extra or {}),
    }


def teacher_log(
    *,
    row_id: str,
    kind: str,
    pack: dict,
    messages: list[dict],
    result,
    history: list[dict] | None = None,
) -> dict:
    """The debug surface's payload (§A): everything the student row refuses.

    Deliberately not deduplicated against the row. A post-mortem asks "what did
    the teacher see and what did it say", and answering that from two files
    that have to be joined on an id is how post-mortems stop happening.
    """
    return {
        "id": row_id,
        "record_type": kind,
        "work_type": pack.get("work_type"),
        "scenario_id": pack.get("scenario_id"),
        "variant": pack.get("variant"),
        "messages": list(messages),
        "think": getattr(result, "think", None),
        "text": getattr(result, "text", None),
        "finish_reason": getattr(result, "finish_reason", None),
        "usage": dict(getattr(result, "usage", {}) or {}),
        "model": getattr(result, "model", None),
        "history": list(history or ()),
    }
