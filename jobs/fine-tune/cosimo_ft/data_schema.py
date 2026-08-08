"""Normalisation of the published Cosimo corpora.

Two corpora are read through this module and they do not have the same shape:

* ``btech-software/cosimo-cfa-frm-71k`` (v1) — exam records only. ``metadata``
  and ``verification`` are Arrow structs; preference pairs are embedded on the
  record as ``preference_pair`` and live in the ``preference_pairs`` config
  under ids **shared** with the supervised rows.
* ``btech-software/cosimo-quant-reasoning-v2`` (v2) — five record types.
  ``metadata``, ``verification``, ``conversation`` and ``tool_schemas`` are
  JSON-**encoded strings** (their key sets differ per record type, so the
  publisher chose one stable column type over an inferred union struct), and
  preference pairs are standalone rows in the ``preference`` config under
  ``cosimopref_`` ids **disjoint** from every supervised row.

Both shapes normalise onto :class:`CosimoRecord`; a v1 row simply carries
``record_type="exam"`` and empty non-exam fields.

Pure python + the chat helpers. The JSONL field names produced here are a
contract shared with the training and evaluation scripts:

* eval rows: ``id, record_type, program, topic, subtopic, difficulty,
  question_type, generator, stem_family, question, answer, distractors,
  reasoning_trace``
* SFT rows: eval fields + ``prompt, completion, text``
* preference rows: eval fields + ``pitfall, prompt, chosen, rejected``
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from . import chat

# Wrapper prefixes: a vignette / constructed-response / MCQ variant of a base stem.
STEM_PREFIXES = ("v_", "cr_", "m_")

# The record types the corpus discriminates on. Only EXAM carries the
# `FINAL ANSWER:` grading contract; the other four are the reason v2 exists, and
# rendering them in exam shape would rebuild the style collapse v1 produced.
EXAM = "exam"
ANALYSIS = "analysis"
ABSTENTION = "abstention"
AGENTIC = "agentic"
IMPLEMENTATION = "implementation"

# Not a supervised record type: the discriminator v2's standalone preference
# rows carry. They have no gold `answer` -- the two sides are the whole content.
PREFERENCE = "preference"

EVAL_FIELDS = (
    "id",
    "record_type",
    "program",
    "topic",
    "subtopic",
    "difficulty",
    "question_type",
    "generator",
    "stem_family",
    "question",
    "answer",
    "distractors",
    "reasoning_trace",
)


@dataclass(frozen=True)
class CosimoRecord:
    """One normalised dataset row."""

    id: str
    program: str
    topic: str
    subtopic: str
    difficulty: str
    question_type: str
    question: str
    answer: str
    distractors: tuple[str, ...]
    reasoning_trace: str
    generator: str
    pitfalls: tuple[str, ...]
    chosen: dict | None
    rejected: dict | None
    pitfall: str | None
    # v2 additions. Defaulted so a v1 row constructs unchanged.
    record_type: str = EXAM
    code: str = ""
    test_code: str = ""
    conversation: tuple[dict, ...] = ()
    tool_schemas: tuple[dict, ...] = ()
    # The preference failure mode (`false_confidence`, `invented_term`, ...) on
    # a v2 standalone pair. Reported per mode so a stage that moves one mode and
    # not the others is visible rather than averaged away.
    pref_mode: str = ""


def stem_family(generator: str) -> str:
    """Strip a leading ``v_``/``cr_``/``m_`` wrapper prefix.

    Held-out stems must be held out by family, otherwise the wrapper variant
    leaks the same question structure into training.
    """
    name = str(generator or "unknown")
    for prefix in STEM_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _decode(value: Any) -> Any:
    """Parse a JSON-encoded column, passing structured values straight through.

    v1 publishes ``metadata``/``verification`` as Arrow structs, v2 publishes
    them (and ``conversation``/``tool_schemas``) as JSON strings. Silently
    returning ``{}`` for a string — which is what an ``isinstance(value, dict)``
    guard does — would resolve every v2 ``generator`` to ``"unknown"``, collapse
    the split stratification into one stratum and make every configured holdout
    family match nothing.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return value


def _as_dict(value: Any) -> dict:
    decoded = _decode(value)
    return decoded if isinstance(decoded, dict) else {}


def decode_mapping(value: Any) -> dict:
    """A JSON-encoded or struct mapping column, as a dict.

    Public because ``01_prepare_data.py`` reads ``metadata.generator`` and
    ``verification.template`` straight off the raw Hub row to cross-check them.
    """
    return _as_dict(value)


def _as_dicts(value: Any) -> tuple[dict, ...]:
    """A JSON-encoded or native list of objects, as a tuple of dicts."""
    decoded = _decode(value)
    if not isinstance(decoded, list):
        return ()
    return tuple(item for item in decoded if isinstance(item, dict))


def _side(value: Any, fallback_answer: str) -> dict | None:
    """Normalise a preference side to ``{"answer", "reasoning_trace"}``.

    Tolerates the stale plain-string shape documented in ``dataset/FORMAT.md``
    (a bare string is the reasoning trace; the answer falls back to the row's).
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return {"answer": fallback_answer, "reasoning_trace": value}
    if isinstance(value, dict):
        return {
            "answer": _text(value.get("answer")) or fallback_answer,
            "reasoning_trace": _text(value.get("reasoning_trace")),
        }
    return None


def normalize_record(row: dict) -> CosimoRecord:
    """Normalise a row of the ``default`` config, in either corpus's shape."""
    metadata = _as_dict(row.get("metadata"))
    verification = _as_dict(row.get("verification"))
    generator = (
        _text(metadata.get("generator"))
        or _text(verification.get("template"))
        or "unknown"
    )
    answer = _text(row.get("answer"))
    preference = _as_dict(row.get("preference_pair"))
    chosen = _side(preference.get("chosen"), answer)
    rejected = _side(preference.get("rejected"), answer)
    # v1 has no record_type column; every one of its rows is an exam item.
    record_type = (
        _text(row.get("record_type")) or _text(metadata.get("record_type")) or EXAM
    )
    return CosimoRecord(
        id=_text(row.get("id")),
        program=_text(row.get("program")),
        topic=_text(row.get("topic")) or _text(metadata.get("topic")),
        subtopic=_text(row.get("subtopic")) or _text(metadata.get("subtopic")),
        difficulty=_text(row.get("difficulty")) or _text(metadata.get("difficulty")),
        question_type=_text(row.get("question_type"))
        or _text(metadata.get("question_type")),
        question=_text(row.get("question")),
        answer=answer,
        distractors=tuple(_text(d) for d in (row.get("distractors") or [])),
        reasoning_trace=_text(row.get("reasoning_trace")),
        generator=generator,
        pitfalls=tuple(_text(p) for p in (metadata.get("pitfalls_addressed") or [])),
        chosen=chosen,
        rejected=rejected,
        pitfall=_text(preference.get("pitfall")) or None,
        record_type=record_type,
        code=_text(row.get("code")),
        test_code=_text(row.get("test_code")),
        conversation=_as_dicts(row.get("conversation")),
        tool_schemas=_as_dicts(row.get("tool_schemas")),
    )


def normalize_standalone_pref_row(row: dict) -> CosimoRecord:
    """Normalise a row of v2's ``preference`` config.

    These rows are *not* a projection of a supervised record: their ids live in
    the ``cosimopref_`` namespace, disjoint from every supervised id by
    construction, and ``chosen``/``rejected`` are complete assistant responses
    rather than ``{answer, reasoning_trace}`` structs. Nothing here carries an
    ``answer`` column, so the ``FINAL ANSWER:`` contract does not apply and
    :func:`to_pref_row` must render the two sides verbatim.

    That disjointness is the point: the id overlap between SFT targets and
    preference ``chosen`` sides is what made the first DPO run a zero-gradient
    no-op, and it cannot recur when no supervised row shares an id.
    """
    metadata = _as_dict(row.get("metadata"))
    prompt = _text(row.get("prompt")) or _text(row.get("question"))
    chosen = _text(row.get("chosen"))
    rejected = _text(row.get("rejected"))
    return CosimoRecord(
        id=_text(row.get("id")),
        program=_text(row.get("program")),
        topic=_text(row.get("topic")) or _text(metadata.get("topic")),
        subtopic=_text(row.get("subtopic")) or _text(metadata.get("subtopic")),
        difficulty=_text(row.get("difficulty")) or _text(metadata.get("difficulty")),
        question_type=_text(row.get("question_type"))
        or _text(metadata.get("question_type")),
        question=prompt,
        # No gold value exists for a standalone pair; the chosen side is the
        # target, and it is carried as the reasoning trace so `is_blank_record`
        # and the SFT projection see a non-degenerate record.
        answer="",
        distractors=(),
        reasoning_trace=chosen,
        generator=_text(metadata.get("generator"))
        or _text(_as_dict(row.get("verification")).get("template"))
        or "unknown",
        pitfalls=(),
        chosen={"answer": "", "reasoning_trace": chosen} if chosen else None,
        rejected={"answer": "", "reasoning_trace": rejected} if rejected else None,
        pitfall=_text(row.get("pitfall")) or None,
        record_type=PREFERENCE,
        pref_mode=_text(row.get("mode")) or _text(metadata.get("mode")),
    )


def normalize_pref_row(row: dict) -> CosimoRecord:
    """Normalise a row of the ``preference_pairs`` config.

    That config carries no ``generator``; ids are shared with the ``default``
    config, so callers that need the real generator (and therefore the stem
    family) must join on ``id`` and pass it in via a ``generator`` key.
    """
    answer = _text(row.get("answer"))
    chosen = _side(row.get("chosen"), answer)
    rejected = _side(row.get("rejected"), answer)
    return CosimoRecord(
        id=_text(row.get("id")),
        program=_text(row.get("program")),
        topic=_text(row.get("topic")),
        subtopic=_text(row.get("subtopic")),
        difficulty=_text(row.get("difficulty")),
        question_type=_text(row.get("question_type")),
        question=_text(row.get("prompt")) or _text(row.get("question")),
        answer=answer,
        distractors=(),
        reasoning_trace=(chosen or {}).get("reasoning_trace", ""),
        generator=_text(row.get("generator")) or "unknown",
        pitfalls=(),
        chosen=chosen,
        rejected=rejected,
        pitfall=_text(row.get("pitfall")) or None,
    )


def has_preference(rec: CosimoRecord) -> bool:
    """True when the record carries a usable chosen/rejected pair."""
    if rec.chosen is None or rec.rejected is None:
        return False
    chosen = (
        _text(rec.chosen.get("answer")).strip(),
        _text(rec.chosen.get("reasoning_trace")).strip(),
    )
    rejected = (
        _text(rec.rejected.get("answer")).strip(),
        _text(rec.rejected.get("reasoning_trace")).strip(),
    )
    return any(chosen) and any(rejected) and chosen != rejected


def is_valid_python(source: str) -> bool:
    """True when ``source`` parses."""
    if not source.strip():
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def normalize_python_block(source: str) -> str:
    """A parseable code block, or ``""`` when it cannot be made into one.

    v2 revisions before 2026-08-07 ship 7,500 of their 13,000 ``implementation``
    records with a ``test_code`` field that does not parse::

        forwards = bootstrapped_yield([0.02, 0.03, 0.04])
            assert len(forwards) == 4

    The generator applied ``.strip()`` *before* ``textwrap.dedent()``, so the
    first line lost its indent, dedent then measured a common prefix of ``""``
    and did nothing, and every continuation line kept its indentation. Fixed at
    the source in ``dataset/pipelines/templates/v2_implementation.py`` and
    republished, so ``main`` is clean — but ``dataset.revision`` is *meant* to be
    pinned to an older sha for a reproducible result, and those revisions still
    carry it. So the damage is undone here: re-dedent the continuation lines.

    The repair is only attempted on a block that does not already parse, and is
    only accepted when the result parses. A legitimately indented block — a
    ``for`` body, a function with a suite — parses on the first check and is
    never touched. Training a model meant to write idiomatic Python on Python
    that does not parse is worse than training it on less Python, so anything
    still unparseable after the repair is dropped rather than rendered.
    """
    text = (source or "").strip("\n")
    if not text.strip():
        return ""
    if is_valid_python(text):
        return text.strip()
    lines = text.splitlines()
    continuation = [line for line in lines[1:] if line.strip()]
    if not continuation:
        return ""
    indent = min(len(line) - len(line.lstrip()) for line in continuation)
    if not indent:
        return ""
    repaired = "\n".join(
        [lines[0]] + [line[indent:] if line.strip() else line for line in lines[1:]]
    )
    return repaired.strip() if is_valid_python(repaired) else ""


def is_exam(rec: CosimoRecord) -> bool:
    """True when the record carries the ``FINAL ANSWER:`` grading contract.

    The single place that decision is made. It selects the system block (the
    exam protocol is appended only here), the supervised target shape, and
    whether the record is eligible for the graded evaluation suites.
    """
    return rec.record_type == EXAM


def to_eval_row(rec: CosimoRecord) -> dict:
    """The shared evaluation projection of a record."""
    return {
        "id": rec.id,
        "record_type": rec.record_type,
        "program": rec.program,
        "topic": rec.topic,
        "subtopic": rec.subtopic,
        "difficulty": rec.difficulty,
        "question_type": rec.question_type,
        "generator": rec.generator,
        "stem_family": stem_family(rec.generator),
        "question": rec.question,
        "answer": rec.answer,
        "distractors": list(rec.distractors),
        "reasoning_trace": rec.reasoning_trace,
    }


def build_supervised_completion(rec: CosimoRecord, tag: str) -> str:
    """The supervised target text for a non-agentic record.

    Only ``exam`` gets the ``FINAL ANSWER:`` line — it is a grading contract,
    not a house style, and appending it to a 900-token analysis is exactly the
    uniformity that flattened the first run into a calculator.

    ``implementation`` is composed rather than taken from ``answer``: the corpus
    puts the substance in ``code``/``test_code`` and leaves ``answer`` as the
    bare result (``"Value=$1,625,956,825"``, ~20 characters), which on its own
    teaches nothing. Both blocks go through :func:`normalize_python_block`, so
    nothing unparseable is ever a training target.
    """
    if is_exam(rec):
        return chat.build_completion(rec.reasoning_trace, rec.answer, tag)
    if rec.record_type == IMPLEMENTATION:
        parts = [
            f"```python\n{block}\n```"
            for block in (
                normalize_python_block(rec.code),
                normalize_python_block(rec.test_code),
            )
            if block
        ]
        if rec.answer.strip():
            parts.append(rec.answer.strip())
        return "\n\n".join(parts)
    return rec.answer.strip()


def to_sft_row(rec: CosimoRecord, tokenizer: Any, system: str, tag: str) -> dict:
    """Eval fields plus the rendered ``prompt`` / ``completion`` / ``text``.

    ``system`` is composed by the caller, which is also what decides whether the
    exam protocol is attached; pass the ``exam=is_exam(rec)`` variant.

    An ``agentic`` record is rendered as the whole multi-turn conversation with
    its tool schemas bound, split at the first assistant turn. Its interior tool
    results stay masked at training time because the chat template renders them
    as ``<|user|>`` turns, which is what ``train_on_responses_only`` splits on.
    """
    row = to_eval_row(rec)
    if rec.record_type == AGENTIC:
        messages = [{"role": "system", "content": system}, *rec.conversation]
        row.update(
            chat.render_tool_example(tokenizer, messages, list(rec.tool_schemas))
        )
        return row
    completion = build_supervised_completion(rec, tag)
    row.update(chat.render_example(tokenizer, rec.question, completion, system))
    return row


def to_pref_row(rec: CosimoRecord, tokenizer: Any, system: str, tag: str) -> dict:
    """Eval fields plus ``pitfall`` and the rendered preference pair.

    ``chosen``/``rejected`` are rendered through the same path as SFT
    completions, so DPO/ORPO see exactly the SFT output distribution.
    """
    if rec.chosen is None or rec.rejected is None:
        raise ValueError(f"record {rec.id!r} has no preference pair")
    row = to_eval_row(rec)
    rendered = {}
    for side, payload in (("chosen", rec.chosen), ("rejected", rec.rejected)):
        if is_exam(rec):
            completion = chat.build_completion(
                payload.get("reasoning_trace", ""), payload.get("answer", ""), tag
            )
        else:
            # A v2 standalone pair is two complete assistant responses about
            # judgement -- hedging versus false confidence, a real term versus an
            # invented one. Neither side has a gold value and neither carries the
            # grading contract, so appending a bare `FINAL ANSWER:` would train
            # the exam shape onto the very rows meant to teach its absence.
            completion = payload.get("reasoning_trace", "").strip()
        rendered[side] = chat.render_example(
            tokenizer, rec.question, completion, system
        )
    row["pitfall"] = rec.pitfall
    row["prompt"] = rendered["chosen"]["prompt"]
    # Strip the template's trailing EOS so each trainer appends its own. TRL's DPO
    # tokenizer appends EOS unconditionally (dpo_trainer.py:734) while ORPO appends
    # it only when absent (orpo_trainer.py:497), so leaving it in would train DPO
    # on a doubled EOS and make the two stages differ on identical pairs. The SFT
    # `text` column keeps its EOS: TRL's add_eos map there is a no-op when the text
    # already ends with one.
    eos = getattr(tokenizer, "eos_token", None) or ""
    for side in ("chosen", "rejected"):
        text = rendered[side]["completion"]
        row[side] = text[: -len(eos)] if eos and text.endswith(eos) else text
    return row
