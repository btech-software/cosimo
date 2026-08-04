"""Behavioural metrics for the assistant the harness is actually trying to build.

`cosimo_ft/grading.py` answers "was the number right". Nothing answered "is this
still an assistant", and the first full run showed why that gap matters: exam
accuracy rose while mean response length fell from ~750 tokens to 120, and the
served checkpoint answered an open-ended hedging question in `Step 1./Step 2.`
form while inventing a term ("Durbin-Watson duration") that does not exist.

Four things are measured here, none of which need a gold answer:

``exam_shape``
    Did an exam-format trace leak into a non-exam answer. The single most direct
    read on style collapse, and it is a regex.
``unknown_terms``
    Technical-looking terms absent from the curriculum vocabulary. A triage aid,
    not a verdict -- see :func:`unknown_terms`.
``abstention``
    Did the model ask for what it is missing, or answer anyway. The persona
    claims honesty about what it does not know and every supervised target is a
    confident computation, so this measures a claim training actively undercuts.
``tool trajectory``
    Whether a multi-step tool conversation actually completes. Training contains
    exactly one round-trip per example, so anything longer is extrapolation.

Pure python and stdlib only, so the whole module is unit-testable on a CPU-only
machine alongside the rest of `tests/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from . import config as config_mod

# --------------------------------------------------------------------------
# exam-shape leakage
# --------------------------------------------------------------------------

# The three fingerprints of the training corpus's trace format. `ASSUMPTIONS:`
# and the `Step N.` enumeration come from the generator templates
# (dataset/pipelines/templates/*.py); `FINAL ANSWER:` is the grading contract,
# which prompt.exam_protocol attaches only to exam items. Any of them appearing
# in an answer to an open question is the exam format bleeding out of its lane.
EXAM_SHAPE_PATTERNS = {
    "assumptions_header": re.compile(r"^\s*ASSUMPTIONS:", re.MULTILINE),
    "numbered_steps": re.compile(r"^\s*Step\s+\d+\s*[.:]", re.MULTILINE),
    "final_answer_tag": re.compile(r"^\s*FINAL ANSWER:", re.MULTILINE),
}

# One `Step 1.` is a legitimate way to walk through a method; three or more is
# the corpus template. Counting rather than flagging the first occurrence keeps
# the metric from firing on ordinary well-structured prose.
NUMBERED_STEP_RUN = 3


def exam_shape_markers(text: str) -> list[str]:
    """Which exam-trace fingerprints appear in ``text``.

    ``numbered_steps`` requires a run of at least :data:`NUMBERED_STEP_RUN`, so
    a single enumerated step does not count as collapse.
    """
    found = []
    for name, pattern in EXAM_SHAPE_PATTERNS.items():
        matches = pattern.findall(text or "")
        if name == "numbered_steps":
            if len(matches) >= NUMBERED_STEP_RUN:
                found.append(name)
        elif matches:
            found.append(name)
    return sorted(found)


def has_exam_shape(text: str) -> bool:
    """True when an open-ended answer is wearing the exam trace format."""
    return bool(exam_shape_markers(text))


# --------------------------------------------------------------------------
# abstention / calibration
# --------------------------------------------------------------------------

# Phrasings that indicate the model noticed something was missing rather than
# inventing it. Deliberately conservative: this metric exists to show that
# abstention is RARE, so a loose pattern that over-credits would hide the very
# finding it was built to surface.
ABSTENTION_PATTERNS = (
    re.compile(r"\b(?:could|can|would) you (?:clarify|specify|confirm|tell me)", re.I),
    re.compile(r"\bI (?:don't|do not|cannot|can't) (?:know|say|determine)", re.I),
    re.compile(r"\b(?:I'd|I would) need (?:to know|more)", re.I),
    re.compile(r"\bnot enough information\b", re.I),
    re.compile(r"\b(?:is|are) (?:not specified|unspecified|missing)\b", re.I),
    re.compile(r"\bwhich (?:.{0,30})?\b(?:did you mean|are you asking)", re.I),
    re.compile(r"\bthis question (?:is|cannot)\b.{0,40}\b(?:ill-posed|be answered)", re.I),
    re.compile(r"\bno (?:single |one )?(?:correct|right) answer\b", re.I),
)


def is_abstention(text: str) -> bool:
    """True when the response asks for missing information or declines.

    Only the OPENING of the response is inspected. A model that produces a full
    confident answer and then adds "of course, I'd need to know your horizon"
    has not abstained -- it has already committed, and crediting that would make
    the metric measure politeness rather than calibration.
    """
    head = (text or "").strip()[:600]
    return any(pattern.search(head) for pattern in ABSTENTION_PATTERNS)


# --------------------------------------------------------------------------
# terminology validity
# --------------------------------------------------------------------------

# Multi-word Title Case phrases and hyphenated eponyms are where fabricated
# terminology shows up ("Durbin-Watson duration", "Carino Smoothing Ratio").
# Ordinary prose capitalisation at the start of a sentence is excluded by
# requiring at least two capitalised words, or an internal hyphen between them.
# The trailing lowercase noun is what turns a known eponym into an unknown
# collocation: "Durbin-Watson" is a real statistic, "Durbin-Watson duration" is
# not, and only the second should be reported. Plurals are matched (`s?`) because
# the failure that motivated this appeared as "Durbin-Watson durations".
_TERM_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+(?:\s+(?:ratio|model|duration|measure|test|"
    r"factor|premium|theorem|equation|process|statistic)s?)?)\b"
)

# Words that begin sentences and would otherwise pair with a following proper
# noun to look like a compound term.
_SENTENCE_STARTERS = frozenset(
    {"The", "This", "That", "These", "Those", "It", "If", "When", "While", "However"}
)


def normalize_term(term: str) -> str:
    """Casefolded, whitespace- and hyphen-normalised form for vocabulary lookup."""
    return re.sub(r"[\s\-]+", " ", (term or "").strip()).casefold()


def load_vocabulary(paths: Iterable[str | Path]) -> set[str]:
    """Build the known-term vocabulary from the shipped reference files.

    Accepts the curriculum taxonomy (nested JSON, topics and subtopics are
    harvested) and plain-text glossaries (one term per line, ``#`` comments).
    Unreadable or missing files raise -- a silently empty vocabulary would make
    every term look invented.
    """
    vocabulary: set[str] = set()
    for path in paths:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"vocabulary file not found: {resolved}")
        text = resolved.read_text(encoding="utf-8")
        if resolved.suffix == ".json":
            for term in _harvest_json_terms(json.loads(text)):
                vocabulary.add(normalize_term(term))
        else:
            for line in text.splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    vocabulary.add(normalize_term(line))
    vocabulary.discard("")
    return vocabulary


def _harvest_json_terms(node: object) -> list[str]:
    """Every ``topic`` value and ``subtopics`` entry anywhere in the tree."""
    terms: list[str] = []
    if isinstance(node, dict):
        if isinstance(node.get("topic"), str):
            terms.append(node["topic"])
        for value in node.values():
            terms.extend(_harvest_json_terms(value))
    elif isinstance(node, list):
        for item in node:
            terms.extend(_harvest_json_terms(item))
    elif isinstance(node, str):
        terms.append(node)
    return terms


def candidate_terms(text: str) -> list[str]:
    """Technical-looking multi-word terms mentioned in ``text``."""
    seen: dict[str, None] = {}
    for match in _TERM_RE.finditer(text or ""):
        term = match.group(1).strip()
        first = term.split()[0]
        if first in _SENTENCE_STARTERS:
            # Drop the leading sentence starter and keep the rest if it still
            # looks like a compound term.
            remainder = term[len(first) :].strip()
            if len(remainder.split()) < 2:
                continue
            term = remainder
        seen.setdefault(term, None)
    return list(seen)


def unknown_terms(text: str, vocabulary: set[str]) -> list[str]:
    """Candidate terms absent from ``vocabulary``.

    **This is a triage aid, not a hallucination detector.** The vocabulary is the
    curriculum taxonomy plus a hand-written glossary, so it is far from complete:
    a real term it has never heard of is reported exactly like an invented one.
    Read the list, do not threshold on it. It is here because scanning twenty
    flagged phrases is tractable and reading four hundred responses is not.
    """
    return [
        term for term in candidate_terms(text) if normalize_term(term) not in vocabulary
    ]


# --------------------------------------------------------------------------
# tool trajectories
# --------------------------------------------------------------------------


def grade_trajectory(scenario: dict, calls: list[dict], final_text: str) -> dict:
    """Score one completed tool conversation against its scenario.

    ``scenario`` carries ``expected_calls`` (ordered tool names the task needs)
    and optionally ``expected_final`` (substrings the answer must contain, which
    is how a multi-step task proves it used the tool RESULTS rather than merely
    emitting calls).

    ``no_call`` scenarios invert the test: the correct behaviour is to answer
    directly, so any tool call is a failure. Those exist because a model trained
    only on call/answer pairs calls a tool for every question it is ever asked.
    """
    expected = [str(name) for name in scenario.get("expected_calls", [])]
    called = [str(call.get("name", "")) for call in calls]
    offered = {str(name) for name in scenario.get("offered_tools", [])}

    if scenario.get("no_call"):
        return {
            "kind": "no_call",
            "correct": not called,
            "n_calls": len(called),
            "hallucinated_tools": sorted({c for c in called if c not in offered}),
            "selected_expected": None,
            "arguments_valid": None,
            "completed_chain": None,
        }

    # Order-insensitive on purpose: a two-tool task can legitimately resolve its
    # dependencies either way round, and penalising order would measure the
    # scenario author's preference rather than the model.
    selected_expected = set(expected) <= set(called)
    arguments_valid = all(isinstance(call.get("arguments"), dict) for call in calls)
    final = (final_text or "").casefold()
    completed_chain = all(
        str(fragment).casefold() in final
        for fragment in scenario.get("expected_final", [])
    )
    return {
        "kind": "call",
        "correct": bool(selected_expected and arguments_valid and completed_chain),
        "n_calls": len(called),
        "hallucinated_tools": sorted({c for c in called if c not in offered}),
        "selected_expected": selected_expected,
        "arguments_valid": arguments_valid,
        "completed_chain": completed_chain,
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def summarize_open_ended(rows: list[dict]) -> dict:
    """Aggregate open-ended and calibration rows into the metrics block."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "exam_shape_rate": 0.0,
            "exam_shape_markers": {},
            "abstention_rate": 0.0,
            "mean_new_tokens": 0.0,
            "unknown_term_rate": 0.0,
            "unknown_terms": {},
        }
    marker_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}
    for row in rows:
        for marker in row.get("exam_shape_markers", []):
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        for term in row.get("unknown_terms", []):
            term_counts[term] = term_counts.get(term, 0) + 1
    return {
        "n": n,
        "exam_shape_rate": sum(1 for r in rows if r.get("exam_shape")) / n,
        "exam_shape_markers": dict(sorted(marker_counts.items())),
        "abstention_rate": sum(1 for r in rows if r.get("abstention")) / n,
        "mean_new_tokens": sum(float(r.get("new_tokens") or 0) for r in rows) / n,
        "unknown_term_rate": sum(1 for r in rows if r.get("unknown_terms")) / n,
        # Most frequent first: a term invented once is noise, a term invented in
        # thirty responses is a learned error.
        "unknown_terms": dict(
            sorted(term_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
        ),
    }


def summarize_agentic(rows: list[dict]) -> dict:
    """Aggregate tool-trajectory rows into the metrics block."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "no_call_precision": 0.0,
            "arguments_valid_rate": 0.0,
            "hallucinated_tool_rate": 0.0,
            "multi_step_accuracy": 0.0,
        }
    call_rows = [r for r in rows if r.get("kind") == "call"]
    no_call_rows = [r for r in rows if r.get("kind") == "no_call"]
    multi = [r for r in call_rows if r.get("n_expected", 0) > 1]
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r.get("correct")) / n,
        "no_call_precision": (
            sum(1 for r in no_call_rows if r.get("correct")) / len(no_call_rows)
            if no_call_rows
            else 0.0
        ),
        "arguments_valid_rate": (
            sum(1 for r in call_rows if r.get("arguments_valid")) / len(call_rows)
            if call_rows
            else 0.0
        ),
        "hallucinated_tool_rate": sum(
            1 for r in rows if r.get("hallucinated_tools")
        )
        / n,
        # Broken out because training contains exactly one round-trip per
        # example: this is the number that says whether chaining generalised.
        "multi_step_accuracy": (
            sum(1 for r in multi if r.get("correct")) / len(multi) if multi else 0.0
        ),
    }


def default_vocabulary_paths(cfg: dict) -> list[Path]:
    """The vocabulary files, resolved against the harness root."""
    return [
        config_mod.harness_path(path)
        for path in config_mod.get(cfg, "assistant.vocabulary_files", [])
    ]
