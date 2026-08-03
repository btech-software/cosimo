"""Normalisation of `btech-software/cosimo-cfa-frm-71k` rows.

Pure python + the chat helpers. The JSONL field names produced here are a
contract shared with the training and evaluation scripts:

* eval rows: ``id, program, topic, subtopic, difficulty, question_type,
  generator, stem_family, question, answer, distractors, reasoning_trace``
* SFT rows: eval fields + ``prompt, completion, text``
* preference rows: eval fields + ``pitfall, prompt, chosen, rejected``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import chat

# Wrapper prefixes: a vignette / constructed-response / MCQ variant of a base stem.
STEM_PREFIXES = ("v_", "cr_", "m_")

EVAL_FIELDS = (
    "id",
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


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


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
    """Normalise a row of the ``default`` config."""
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


def to_eval_row(rec: CosimoRecord) -> dict:
    """The shared evaluation projection of a record."""
    return {
        "id": rec.id,
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


def to_sft_row(rec: CosimoRecord, tokenizer: Any, system: str, tag: str) -> dict:
    """Eval fields plus the rendered ``prompt`` / ``completion`` / ``text``."""
    row = to_eval_row(rec)
    completion = chat.build_completion(rec.reasoning_trace, rec.answer, tag)
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
        completion = chat.build_completion(
            payload.get("reasoning_trace", ""), payload.get("answer", ""), tag
        )
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
