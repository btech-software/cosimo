"""Normalisation of the published Cosimo corpora.

Three corpora are read through this module and none of them have the same
shape:

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

* ``btech-software/cosimo-quant-assistant-v3`` (v3) — eight record types
  (v2's five plus ``memo``, ``critique``, ``grounded``). Written as raw JSONL,
  so ``messages``/``tool_schemas``/``verification`` are **native** lists and
  dicts rather than JSON strings. It shares *no* taxonomy column with the
  others: no ``program``, ``topic``, ``subtopic``, ``difficulty``,
  ``question_type``, ``metadata`` or ``verified``. Its axes are ``work_type``,
  ``scenario_id`` (``<work_type>.<family>``) and ``register``. Preference pairs
  are standalone rows under ``cosimov3pref_`` ids, disjoint from the
  ``cosimov3_`` supervised namespace.

All three shapes normalise onto :class:`CosimoRecord`; a v1 row simply carries
``record_type="exam"`` and empty non-exam fields, and a v3 row maps its own
axes onto ``program``/``generator``/``stem_family`` so the splitter needs no
branch (see :func:`normalize_v3_record`).

Pure python + the chat helpers. The JSONL field names produced here are a
contract shared with the training and evaluation scripts:

* eval rows: ``id, record_type, program, topic, subtopic, difficulty,
  question_type, generator, stem_family, question, answer, distractors,
  reasoning_trace, work_type, scenario_id, register``
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
# `FINAL ANSWER:` grading contract; the others are the reason v2 and v3 exist,
# and rendering them in exam shape would rebuild the style collapse v1 produced.
EXAM = "exam"
ANALYSIS = "analysis"
ABSTENTION = "abstention"
AGENTIC = "agentic"
IMPLEMENTATION = "implementation"

# v3 additions. All three are prose kinds -- they render as the answer verbatim,
# exactly like ANALYSIS -- and differ from each other only in register and
# brief, which is a corpus-side distinction the harness reports but does not act
# on. `grounded` answers a question against a supplied stimulus; `memo` is the
# long IC-memo register; `critique` argues against a supplied position.
MEMO = "memo"
CRITIQUE = "critique"
GROUNDED = "grounded"

# Not a supervised record type: the discriminator v2's and v3's standalone
# preference rows carry. They have no gold `answer` -- the two sides are the
# whole content.
PREFERENCE = "preference"

# Id namespaces. v3 supervised rows are `cosimov3_<record_type>_<seed:016x>`
# and its pairs `cosimov3pref_<seed:016x>`, disjoint by prefix and enforced on
# the corpus side by verify_v3. The prefix is also the corpus discriminator the
# harness dispatches on: a v3 row has no `metadata` column to sniff, and
# `record_type` alone cannot tell a v2 `analysis` from a v3 one.
V3_PREFERENCE_ID_PREFIX = "cosimov3pref_"
V3_ID_PREFIX = "cosimov3_"

#: The factory's own signature, as it appears in every brief the v3 teacher
#: module builds. A student row must never carry it, and this side of the join
#: enforces that independently of the corpus side: the two subsystems meet on
#: the Hub, and either can be handed bytes the other never saw. Kept as a
#: literal rather than imported -- `dataset` and `jobs` do not import each
#: other (spec §I), and a shared constant would be exactly such an import.
V3_TEACHER_FINGERPRINT = "Cosimo v3 teacher"

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
    # v3 axes. Empty strings on a v1/v2 row, so the column set stays one shape
    # across corpora and a reader of the written JSONL can tell which corpus a
    # row came from without joining back to the id namespace.
    "work_type",
    "scenario_id",
    "register",
    # §G.5. Null on a v1/v2 row, so the column set stays one shape.
    "fact_pack",
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
    # a v2 standalone pair, or the named pitfall on a v3 pair. Reported per mode
    # so a stage that moves one mode and not the others is visible rather than
    # averaged away.
    pref_mode: str = ""
    # v3 additions. Defaulted so a v1/v2 row constructs unchanged.
    #
    # `scenario_id` is `<work_type>.<family>` and is the v3 holdout axis, the
    # way `stem_family(generator)` is the v1/v2 one. `register` is the voice the
    # answer was written in (`desk_chat`, `ic_memo`, ...); carried through to
    # the eval rows so a register the model cannot hold is visible per-slice.
    work_type: str = ""
    scenario_id: str = ""
    register: str = ""
    # The v3 fact pack, as an object (amendment §G.5). It goes to the *eval
    # sidecar* and to nowhere else: an eval slice has to be able to score an
    # invented-number rate against the row's own contract, and a corpus whose
    # numbers can only be rechecked by re-running the generator is a corpus
    # whose numbers are checked once. Putting it in a chat turn instead would
    # be the labelling protocol back in the prompt by another name -- which is
    # the thing §A exists to stop.
    fact_pack: dict | None = None


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


# --------------------------------------------------------------------------
# v3
# --------------------------------------------------------------------------


def is_v3_row(row: dict) -> bool:
    """True when the row came from the v3 corpus, decided by its id namespace.

    The id is the only discriminator that works on every config: v3 drops
    `metadata` entirely, and `record_type` is shared vocabulary (a v2 row and a
    v3 row both say `analysis`).
    """
    return str(row.get("id", "")).startswith((V3_ID_PREFIX, V3_PREFERENCE_ID_PREFIX))


def _flatten_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_flatten_text(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return " ".join(_flatten_text(v) for v in node)
    return ""


def carries_teacher_brief(row: dict) -> bool:
    """True when any byte of this row is the v3 factory talking to itself.

    Checked over the whole row rather than over ``messages`` alone: the leak
    being closed is "the labelling protocol reached a training target", and a
    brief pasted into ``question`` leaks exactly as hard as one left in a
    message list. A row that trips this is dropped, never repaired -- the
    repair would be guessing which half of the text was the lesson.
    """
    return V3_TEACHER_FINGERPRINT in _flatten_text(row)


def is_holdout(row: dict) -> bool:
    """Whether the corpus says this row's scenario family is eval material.

    Read off the row's own column. It is defence in depth: the render side
    already writes holdout families to ``eval/`` rather than ``sft/``, so a
    holdout row reaching here means either an old shard tree or a bug, and
    both are reasons to drop rather than to train.
    """
    return bool(row.get("holdout"))


def scenario_family(work_type: str, scenario_id: str) -> str:
    """The family half of a v3 ``scenario_id``.

    ``scenario_id`` is ``f"{work_type}.{family}"`` and *work_type itself
    contains dots* (``valuation.equity.dcf``), so this is a length slice, never
    a ``split(".")``. Falls back to the whole id when the two disagree, which
    keeps a malformed row in one identifiable stratum instead of silently
    joining another family's.
    """
    work_type = str(work_type or "")
    scenario_id = str(scenario_id or "")
    prefix = f"{work_type}."
    if work_type and scenario_id.startswith(prefix):
        return scenario_id[len(prefix) :]
    return scenario_id


def _v3_messages(row: dict) -> tuple[dict, ...]:
    return _as_dicts(row.get("messages"))


def _v3_question(row: dict) -> str:
    """The prompt as the student should see it.

    ``question`` first, which is the fact pack's own question and, since the
    corpus's two-surface split, the only prompt a prose row carries. It used to
    read ``messages[1]`` in preference, because that is where the *brief* lived
    -- the teacher's system contract, the pack as JSON, the word budget -- and
    training on that taught the model that a question looks like a labelling
    protocol. That was the leak; this is the fix on the reading side.

    ``question_text`` is the one honest exception and it is a named column
    rather than a message index: an exam item's four labelled options are part
    of its prompt, and asking for an option letter without showing the options
    is not a question. Implementation and agentic rows keep their own
    transcripts, handled in :func:`normalize_v3_record`.
    """
    explicit = _text(row.get("question_text"))
    if explicit:
        return explicit
    question = _text(row.get("question"))
    if question:
        return question
    for message in _v3_messages(row):
        if message.get("role") == "user":
            return _text(message.get("content"))
    return ""


def split_final_answer(text: str, tag: str = "FINAL ANSWER:") -> tuple[str, str]:
    """Split a completed exam target into ``(reasoning_trace, answer)``.

    v3 ships the exam answer as one finished assistant turn, but the harness
    stores the trace and the value apart -- ``chat.build_completion`` reassembles
    them, ``grading.grade_cosimo`` reads the value, and ``is_blank_record``
    requires both. The split is on the LAST occurrence of the tag, so a trace
    that quotes the contract on its way to using it does not truncate the row.
    """
    text = (text or "").rstrip()
    index = text.rfind(tag)
    if index == -1:
        return text, ""
    return text[:index].rstrip(), text[index + len(tag) :].strip()


def _v3_options(row: dict) -> tuple[str, list[str]]:
    """``(gold, distractors)`` rendered as ``"<label>. <text>"`` strings.

    ``grading``'s MCQ path reads a leading option letter off each string, and
    its numeric fallback reads the value out of the same string, so one
    rendering serves both. The gold option is the one whose ``value`` is
    ``answer_value``: the option list is the authority on which letter is
    correct, not the prose that reached it.

    The gold rendering is only a *fallback* for the row's answer -- see
    :func:`normalize_v3_record`. It is returned so the distractor list can be
    built by exclusion in the same pass.
    """
    options = [o for o in _as_dicts(row.get("options"))]
    gold_value = row.get("answer_value")
    gold = ""
    distractors: list[str] = []
    for option in options:
        rendered = f"{_text(option.get('label'))}. {_text(option.get('text'))}".strip()
        if not gold and gold_value is not None and option.get("value") == gold_value:
            gold = rendered
        else:
            distractors.append(rendered)
    return gold, distractors


def normalize_v3_record(row: dict) -> CosimoRecord:
    """Normalise one row of the v3 ``default`` config onto :class:`CosimoRecord`.

    v3 shares no taxonomy columns with v1/v2 -- there is no ``program``,
    ``topic``, ``subtopic``, ``difficulty``, ``question_type`` or ``metadata``.
    What it has instead is ``work_type`` / ``scenario_id`` / ``variant``, so the
    two axes the rest of the harness is built on are mapped onto it:

    ``program``    <- ``work_type``     (the manifest's coarse composition axis)
    ``generator``  <- ``scenario_id``   (the split stratum)
    ``stem_family`` <- ``scenario_id``  (the holdout key)

    That mapping is why ``splits.py`` needs no v3 branch at all: it strata on
    ``(program, generator)`` and holds out on ``stem_family``, and all three are
    populated. ``stem_family`` is the full ``scenario_id`` rather than the bare
    family so two work types cannot collide on a shared family name.
    """
    record_type = _text(row.get("record_type")) or ANALYSIS
    work_type = _text(row.get("work_type"))
    scenario_id = _text(row.get("scenario_id"))
    question = _v3_question(row)
    answer = _text(row.get("answer"))
    reasoning_trace = ""
    distractors: tuple[str, ...] = ()
    question_type = ""
    code = ""
    test_code = ""
    conversation: tuple[dict, ...] = ()

    if record_type == EXAM:
        # The grader needs the option letter and the value apart from the trace
        # that reached them; the corpus ships one finished turn.
        reasoning_trace, final = split_final_answer(answer)
        gold, distractors_list = _v3_options(row)
        # The corpus's own final line, verbatim -- `B -- 28.16 bp`, not a
        # reconstruction from the option list. It has to be verbatim because
        # `chat.build_completion` reassembles trace + tag + this into the
        # supervised target, and that target must be byte-identical to the
        # assistant turn the corpus verified and published. Rebuilding it as
        # `B. 28.16` would drop the unit and the separator, and train the model
        # to close in a form `prompt.exam_protocol` does not ask for.
        #
        # Both readings name the same option -- verify_v3 requires the final
        # line to agree with `answer_value` -- so this is a choice of surface,
        # not of answer. The option rendering stands in only for a row whose
        # final line is missing, which the corpus gate should already have
        # refused to publish.
        answer = final or gold
        distractors = tuple(distractors_list)
        question_type = "MCQ"
    elif record_type == AGENTIC:
        # The one record type whose target *is* a transcript, so it is the one
        # `messages` is read for (§G.1). Two turns are refused: the corpus's own
        # system turn -- the harness composes its own from prompt.identity, and
        # binding both would put the renderer's instructions in front of the
        # persona being trained -- and any turn still bearing the factory
        # fingerprint, which a current corpus does not produce and an older
        # shard tree does.
        conversation = tuple(
            m
            for m in _v3_messages(row)
            if m.get("role") != "system"
            and V3_TEACHER_FINGERPRINT not in _text(m.get("content"))
        )
    elif record_type == IMPLEMENTATION:
        # v3 renames v2's two code fields and adds a third. The reference
        # implementation and the public tests are the supervised substance;
        # `limitations` is the one teacher-authored field on the row and reads
        # as the prose that follows the code, which is exactly the slot
        # `build_supervised_completion` gives `answer`.
        code = _text(row.get("reference_code"))
        test_code = "\n\n".join(
            _text(t) for t in (row.get("public_tests") or []) if _text(t).strip()
        )
        answer = _text(row.get("limitations"))

    return CosimoRecord(
        id=_text(row.get("id")),
        program=work_type,
        topic="",
        subtopic="",
        difficulty="",
        question_type=question_type,
        question=question,
        answer=answer,
        distractors=distractors,
        reasoning_trace=reasoning_trace,
        generator=scenario_id,
        pitfalls=(),
        chosen=None,
        rejected=None,
        pitfall=None,
        record_type=record_type,
        code=code,
        test_code=test_code,
        conversation=conversation,
        tool_schemas=_as_dicts(row.get("tool_schemas")),
        work_type=work_type,
        scenario_id=scenario_id,
        register=_text(row.get("register")),
        fact_pack=_as_dict(row.get("fact_pack")) or None,
    )


def normalize_v3_pref_row(row: dict) -> CosimoRecord:
    """Normalise one row of the v3 ``preference`` config.

    Same disjointness guarantee as v2's standalone pairs, in a different
    namespace (``cosimov3pref_`` against ``cosimov3_``), and the same shape
    consequence: ``chosen``/``rejected`` are complete assistant responses rather
    than ``{answer, reasoning_trace}`` structs, so there is no gold value and
    the ``FINAL ANSWER:`` contract does not apply. ``parent_kind`` names the
    supervised type the pair paraphrases; it is never ``exam`` or
    ``implementation``, so no v3 pair ever needs the exam rendering.
    """
    prompt = ""
    for message in _as_dicts(row.get("prompt")):
        if message.get("role") == "user":
            prompt = _text(message.get("content"))
    prompt = prompt or _text(row.get("question"))
    chosen = _text(row.get("chosen"))
    rejected = _text(row.get("rejected"))
    work_type = _text(row.get("work_type"))
    scenario_id = _text(row.get("scenario_id"))
    return CosimoRecord(
        id=_text(row.get("id")),
        program=work_type,
        topic="",
        subtopic="",
        difficulty="",
        question_type="",
        question=prompt,
        answer="",
        distractors=(),
        reasoning_trace=chosen,
        generator=scenario_id,
        pitfalls=(),
        chosen={"answer": "", "reasoning_trace": chosen} if chosen else None,
        rejected={"answer": "", "reasoning_trace": rejected} if rejected else None,
        pitfall=_text(row.get("pitfall")) or None,
        record_type=PREFERENCE,
        pref_mode=_text(row.get("pitfall")),
        work_type=work_type,
        scenario_id=scenario_id,
        register=_text(row.get("register")),
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
        "work_type": rec.work_type,
        "scenario_id": rec.scenario_id,
        "register": rec.register,
        "fact_pack": rec.fact_pack,
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
