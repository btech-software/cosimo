"""Behavioural metrics for the assistant the harness is actually trying to build.

`cosimo_ft/grading.py` answers "was the number right". Nothing answered "is this
still an assistant", and the first full run showed why that gap matters: exam
accuracy rose while mean response length fell from ~750 tokens to 120, and the
served checkpoint answered an open-ended hedging question in `Step 1./Step 2.`
form while inventing a term ("Durbin-Watson duration") that does not exist.

Seven things are measured here, none of which need a gold answer:

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
``invented numbers``
    Figures the supplied facts do not explain. The v3 corpus refuses to publish
    a row that fails this gate, so the number here answers whether the student
    learned the constraint or only the prose around it (spec §8.4).
``must-mention coverage``
    Share of the terms a grounded answer was required to address. The corpus
    binds these per fact pack; a model that writes fluently around the point
    scores well on everything else and badly here.
``register match``
    Whether the answer is shaped like the voice it was asked for. Answering a
    one-line desk question with a four-heading memo is the same class of
    failure as answering it in exam form, and nothing else measured it.

The last three apply only to suite rows that declare the contract they measure,
so each carries its own denominator in the summary. An unlabelled row is
unscored, never scored zero.

Pure python and stdlib only, so the whole module is unit-testable on a CPU-only
machine alongside the rest of `tests/`.
"""

from __future__ import annotations

import json
import math
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
    re.compile(
        r"\bthis question (?:is|cannot)\b.{0,40}\b(?:ill-posed|be answered)", re.I
    ),
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
# fact grounding (spec §8.4)
# --------------------------------------------------------------------------

# Number tokens, thousands separators and decimals included, so "1,430,000.00"
# is one number rather than three. Deliberately the same shape as the corpus's
# own tokenizer (dataset/verification/nums.py): a figure the generation gate
# would have rejected must be a figure this metric counts, or the two halves of
# the system disagree about what a number is.
_NUMBER_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

#: A real minus, written properly. The tokenizer knows only ASCII ``-``, so a
#: dash that means minus is folded to one before a digit -- the corpus records
#: a live attribution memo that used U+2013 seventeen times, reading every
#: negative figure in it as positive and refusing seven pack values as
#: inventions. The em dash is deliberately left alone: "the cost -- 28 bp --
#: was high" is punctuation, not minus twenty-eight.
_ASCII_MINUS = re.compile(r"[\u2212\u2013](?=\d)")


def number_tokens(text: str) -> list[str]:
    """Number tokens in *text*, with hyphens that are not minus signs undone.

    The bare tokenizer treats a leading hyphen as part of the number, so any
    hyphen before a digit becomes that number's sign. In prose it usually is
    not one: "a 1-in-100 day" yields ``1`` and ``-100``, and the corpus records
    a VaR row dead-lettered three times over for an invented ``-100`` it never
    wrote. A leading sign counts as a sign only when what precedes it is not a
    word character.

    Mirrors ``verification/invented_numbers.read_tokens``. Reading the raw
    tokenizer here instead was this module's fourth drift from that gate, and
    the costly one: it inflated ``invented_number_rate`` -- the headline number
    for the corpus's whole purpose -- on any answer that hyphenated its prose.
    """
    out: list[str] = []
    raw = _ASCII_MINUS.sub("-", str(text or ""))
    for match in _NUMBER_TOKEN_RE.finditer(raw):
        token = match.group(0)
        start = match.start()
        if token[0] in "-+" and start > 0 and raw[start - 1].isalnum():
            token = token[1:]  # a hyphen inside a word, not a sign
        out.append(token)
    return out


# Relative, not absolute: packs mix 1e4 AUMs with 1e-4 spreads, and a fixed
# epsilon would either wave everything through or reject a pack's own figures.
NUMBER_REL_TOLERANCE = 0.005

# The two honest re-renderings of a pack value -- 0.03 written as "3.00%", 12.5
# written as a 0.125 ratio -- and nothing more permissive. A 1000x slip is not
# the same number, which is exactly what a unit hallucination looks like.
NUMBER_SCALE_FACTORS = (1.0, 100.0, 0.01)


def _number_value(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _decimals(token: str) -> int:
    """Decimal places the token as *written* claims, the way a reader reads it."""
    token = str(token).strip().lstrip("+-").replace(",", "")
    return len(token.split(".", 1)[1]) if "." in token else 0


def _number_allowed(value: float, allowed: frozenset[float], decimals: int = 0) -> bool:
    """Is *value* one of the permitted figures, spelled the way a desk spells it?

    Two paths, and the second is the one this was missing. The relative band
    catches a figure written at full precision; the rounding path catches a
    figure the answer *rounded for the reader* -- "about -2.2%" of a -2.24, or
    "0.43" of a 0.433. A desk rounds when it speaks, and a metric that scores
    the rounding as an invented number is measuring house style rather than
    grounding: a live bond answer, correct on duration and convexity both, was
    marked as inventing 2.2 and 2.3 for exactly this.

    The rounding path is exact at the precision the token itself carries, so it
    admits nothing a reader could tell apart from the permitted figure. It
    mirrors `verification/invented_numbers.py` on the corpus side deliberately:
    the generator and the evaluator disagreeing about what counts as the same
    number is how a corpus ends up certified by a rule it was not written to.
    """
    for scale in NUMBER_SCALE_FACTORS:
        scaled = value * scale
        for permitted in allowed:
            if permitted == 0.0:
                if scaled == 0.0:
                    return True
                continue
            if abs(scaled - permitted) <= NUMBER_REL_TOLERANCE * abs(permitted):
                return True
            if (
                decimals >= 1
                and abs(round(permitted / scale, decimals) - value) <= 1e-9
            ):
                return True
    return False


def conventions_cited(text: str, conventions) -> list[str]:
    """Which declared standards of the field this answer reached for.

    A counter, not a gate. The suites declare `conventions` -- the Sharpe
    bands, a Basel multiplier, long-run nominal GDP -- because an assistant
    that never places its figure against one is a calculator; this reports how
    often the model does it, so the permission granted in
    `invented_numbers` stays visible rather than becoming an unexamined hole.
    """
    if not conventions:
        return []
    # Read with this module's own number tokenizer, so "cited" means the same
    # thing here as "invented" does twenty lines down.
    seen = {
        value
        for value in (
            _number_value(token) for token in _NUMBER_TOKEN_RE.findall(text or "")
        )
        if value is not None
    }
    cited: list[str] = []
    for number in conventions:
        value = float(number)
        if any(abs(value - found) <= 1e-9 for found in seen):
            token = f"{value:g}"
            if token not in cited:
                cited.append(token)
    return cited


def invented_numbers(
    text: str, allowed: Iterable[float], whitelist: Iterable[str] = ()
) -> list[str]:
    """Number tokens in ``text`` that no allowed value explains, in order.

    This is the harness-side reading of the corpus's invented-number gate
    (``dataset/pipelines/v3/verification/invented_numbers.py``): same tokenizer,
    same 0.5% relative tolerance, same three scale factors. It is a
    reimplementation rather than an import because ``jobs`` must not import
    ``dataset`` -- the two trees are joined by published artifacts, not by
    Python (spec §3). What that costs is a policy that can drift; what it buys
    is the eval measuring the same thing the generator enforced, which is the
    only way "the student invents numbers the teacher was forbidden to invent"
    is a readable finding.

    Returns the *source strings* ("17.4%"), not floats: the operator reading the
    generations file needs what the model wrote, not what the metric parsed.

    ``whitelist`` holds token strings the caller declares arithmetic furniture
    rather than claims about the world -- "100" in a percentage, the as-of year.
    Matching is on the written form. Keep it tiny or it stops being a gate.

    An empty ``allowed`` set means the suite row carries no fact pack, and the
    metric does not apply; the caller skips those rows rather than scoring every
    number in them as invented.
    """
    permitted = frozenset(float(a) for a in allowed if a is not None)
    if not permitted:
        return []
    passed = frozenset(str(w).strip().strip(",") for w in whitelist)
    offenders: list[str] = []
    seen: set[str] = set()
    for token in number_tokens(text):
        if token in seen or token.strip(",") in passed:
            continue
        value = _number_value(token)
        if value is None or not _number_allowed(value, permitted, _decimals(token)):
            offenders.append(token)
            seen.add(token)
    return offenders


#: Suffixes stripped before an anchor is matched, longest first so "-ption" is
#: tried before the "-ion" inside it. English inflection only, and only enough
#: of it to stop a *correct* answer failing on grammar: an anchor on "scaling"
#: must accept "scales", one on "normality assumption" must accept "it
#: assumes". Mirrors ``dataset/pipelines/v3/verification/prose.py``.
_SUFFIXES = (
    "ingly",
    "edly",
    "ption",
    "ing",
    "ies",
    "ion",
    "ity",
    "ed",
    "es",
    "s",
    "y",
    "e",
)

#: Function words carry no analytical content, so requiring them would make an
#: anchor fail on grammar rather than on substance. Only these are dropped;
#: every content word of an anchor is still required.
_STOPWORDS = frozenset(
    """a an the and or nor but of in on to for with by at from as is are was
    were be been being it its this that these those than then so such not no
    versus vs against near over under into via per about""".split()
)


def _stem(word: str) -> str:
    """Strip suffixes until the word stops changing.

    One pass is not enough once "-ption" is in the table: "assumptions" would
    lose its "s" and stop, never reaching the "assum" that "assumes" reduces
    to. Looping is what lands a nominalisation and its verb in one place.
    """
    while True:
        for suffix in _SUFFIXES:
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        else:
            return word


def stems(text: str) -> set[str]:
    """Content-word stems of *text*, hyphens treated as spaces.

    Symmetric on both sides of the comparison, which is the point: an anchor
    written "square-root" has to match an answer that writes "square root",
    and an anchor written "decision" has to match "decision-price".
    """
    words = re.findall(
        r"[a-z0-9]+", " ".join(str(text).casefold().split()).replace("-", " ")
    )
    return {_stem(w) for w in words if w not in _STOPWORDS}


def mention_points(pack: dict, kind: str = "") -> list[str]:
    """The points a row of this record type must engage.

    ``must_mention`` describes a good answer to the pack's *own* question, so
    it is the wrong contract for a row refusing a different one: an abstention
    cannot engage "square-root impact scaling" while declining to price the
    shortfall, and requiring it would push the answer into answering. What an
    abstention owes is the name of the thing that is missing.
    """
    if kind == "abstention":
        missing = str((pack or {}).get("abstention_missing") or "")
        return [missing] if missing else []
    return list((pack or {}).get("must_mention") or [])


def must_mention_hits(text: str, terms: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(hit, missed)`` for the points an answer is required to engage.

    Every content word of the anchor must appear, compared on stems and in any
    order -- not the anchor as one verbatim run of characters.

    This was a substring test, and it had drifted: the corpus gate it mirrors
    moved to stems after verbatim matching "did not survive contact with a real
    teacher", and this copy did not follow. Measured against the corpus's own
    certified prose the two disagreed on 45 of 64 rows, the substring reading
    scoring 51.8% where the gate scored 98.2% -- so a correct answer that made
    every point in its own words was marked as having made none, and the
    headline coverage number understated the corpus by 46 points.
    """
    haystack = stems(text)
    hit, missed = [], []
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        (hit if stems(term) <= haystack else missed).append(term)
    return hit, missed


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------

# The five voices v3 writes in (dataset/pipelines/v3/config.VALID_REGISTERS),
# and the structural tell that separates each from the others. These are shape
# detectors, not judgements of quality: a desk reply carries no memo
# scaffolding, a memo reaches a decision. That is exactly the axis the model
# collapses -- answering a one-line desk question with a four-heading memo is
# the same failure as answering it in exam form, and nothing else measures it.
#: The five registers the corpus declares (``dataset/pipelines/v3/config.py``
#: ``VALID_REGISTERS``). Only the first three carry shape rules; ``auditor``
#: and ``code_review`` are declared and unwritten, so no family emits them and
#: the corpus gate returns clean for both -- which this mirrors rather than
#: scoring them unscored: inventing shape rules for a voice nobody has written
#: would pin down prose that has never been read.
VALID_REGISTERS = (
    "desk_chat",
    "ic_memo",
    "risk_committee",
    "auditor",
    "code_review",
)

#: The memo's scaffolding, anchored at a line start: "is this laid out like a
#: memo". Mirrors ``_MEMO_HEADINGS`` in the corpus gate.
_MEMO_HEADINGS = re.compile(
    r"^\s*(?:[-*#>\s]*)(?:\*\*)?\s*"
    r"(finding|evidence|call|recommendation)\s*(?:\*\*)?\s*:",
    re.IGNORECASE | re.MULTILINE,
)

#: The labelled call, *wherever* it appears -- a different question from the
#: one above, and the answer does not depend on where the label sits.
_CALL_LABEL = re.compile(r"\b(?:call|recommendation)\s*:", re.IGNORECASE)

#: The exam contract, matched the way the corpus's prose gate matches it.
_EXAM_TAG = re.compile(r"final answer\s*:", re.IGNORECASE)

#: Sentence terminators. Newlines break too: a desk reply written as twenty
#: one-line bullets is twenty sentences however few full stops it holds.
_SENTENCE_SPLIT = re.compile(r"[.!?]+[\s)\]]|\n+")

#: The moves that make a passage a decision rather than a survey.
_CALL_TERMS = (
    "recommend",
    "we would",
    "our call",
    "the call is",
    "initiate",
    "maintain",
    "add to",
    "trim",
    "reduce",
    "avoid",
    "prefer",
    "size at",
)

#: The one move a risk committee does not make. Matched on imperative openings
#: so "the desk should not add to the position" (a constraint) is not read as
#: "add to the position" (a call).
_TRADE_CALL = re.compile(
    r"\b(?:we\s+)?(?:recommend|advise)\s+(?:buying|selling|adding|shorting|"
    r"trimming)\b|\b(?:i|we)\s+would\s+(?:buy|sell|short|add|trim)\b",
    re.IGNORECASE,
)

#: What "the risk committee is speaking" looks like in vocabulary. Stems, so
#: "limits"/"limit" both land.
_RISK_TERMS = ("limit", "horizon", "assum", "breach", "exposure", "tolerance")

#: Record types an ``ic_memo`` row may write without reaching a decision: a
#: citation and a refusal legitimately end without one. Shared with
#: ``scripts/01_prepare_data.py``, which caps how much of a register's training
#: slice these kinds may be -- the exemption is correct per row and corrosive
#: in bulk.
CALL_EXEMPT_KINDS = frozenset({"grounded", "abstention"})

#: Desk chat's ceilings. The sentence count alone can be satisfied by writing
#: fewer, longer sentences -- the corpus measured a "desk_chat" memo of 334
#: words in 7 sentences -- so the mean-length ceiling closes that door.
DESK_CHAT_MAX_SENTENCES = 12
DESK_CHAT_WORDS_PER_SENTENCE = 14
DESK_CHAT_MAX_MEAN_SENTENCE = 2 * DESK_CHAT_WORDS_PER_SENTENCE

#: Sentence ceilings belonging to the *kind* rather than to the register: a
#: citation and a refusal are short by construction in a way an analysis is
#: not. Where a kind is stricter than the register, its number wins.
_KIND_SENTENCE_CAPS = {"grounded": 8, "abstention": 5}

#: The word bands each kind is written to, and the one place a register binds
#: tighter than its kind. Duplicated as literals for the reason
#: ``EXAM_SHARE_MAX`` is in ``01_prepare_data``: the trees are joined by
#: published artifacts, not by Python. They are read here only to derive the
#: desk_chat ceiling below -- never to judge an answer by its length, which is
#: the mistake this module is recovering from.
_WORD_BUDGETS = {
    "analysis": (60, 220),
    "memo": (100, 300),
    "grounded": (40, 90),
    "critique": (50, 180),
    "abstention": (25, 120),
}
_REGISTER_WORD_CAPS = {("analysis", "desk_chat"): 160}


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_SPLIT.split(text) if part.strip()])


def desk_chat_ceiling(kind: str) -> int:
    """The sentence ceiling for a desk_chat row of record type *kind*.

    Twelve, except where twelve is the wrong number in either direction.

    A kind may be *stricter* than its register -- ``grounded`` is eight
    sentences and ``abstention`` five wherever they are written -- and where it
    is, the kind's number is the ceiling.

    Twelve can also be *unreachable*. A register belongs to the family and a
    word budget to the record type, and one pack serves every record type, so a
    work type whose families all speak desk chat still has to write a ``memo``
    at a 100-word floor in the desk's voice. At twelve sentences that is an
    eight-word average and at the band's top a 25-word one; demanding both
    would refuse every memo of that work type forever. So the ceiling is twelve
    or what the lane's own band actually permits, whichever is larger.
    """
    kind_cap = _KIND_SENTENCE_CAPS.get(kind)
    if kind_cap:
        return kind_cap
    low, high = _WORD_BUDGETS.get(kind, (0, 0))
    high = min(high, _REGISTER_WORD_CAPS.get((kind, "desk_chat"), high))
    return max(
        DESK_CHAT_MAX_SENTENCES,
        math.ceil(low / DESK_CHAT_WORDS_PER_SENTENCE),
        math.ceil(high / DESK_CHAT_MAX_MEAN_SENTENCE),
    )


def _exam_violations(text: str) -> list[str]:
    """An exam item is worked, not written up, and closes on its tag.

    The kind outranks the voice here, and that is not a nicety: one pack serves
    every record type, so an exam item inherits whatever register its family
    speaks, and holding it to that register would demand an ``ic_memo`` exam
    close on a decision when its actual contract is to close on
    ``FINAL ANSWER:``. The corpus grades an exam row on the exam contract
    whatever its pack names; 84 of 188 shipped exam rows depend on it.
    """
    out: list[str] = []
    found = sorted({m.group(1).lower() for m in _MEMO_HEADINGS.finditer(text)})
    if found:
        out.append(
            "an exam answer carries memo headings ("
            + ", ".join(f"{h.title()}:" for h in found)
            + ") -- an item is worked, not written up"
        )
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines or not _EXAM_TAG.search(lines[-1]):
        out.append(
            "an exam answer must close on its 'FINAL ANSWER:' line and nothing after it"
        )
    return out


def _grounded_violations(text: str) -> list[str]:
    """A citation carries no scaffolding, whatever register it is written in.

    A ``grounded`` row is told "no section labels of any kind", and the rule
    belongs to the kind rather than to the voice: the corpus records that the
    first live grounded row on an ``ic_memo`` family closed on "Call: act on
    Financials allocation", obeying its register and breaking its kind. Checked
    on top of the per-register rule, not instead of it.
    """
    out: list[str] = []
    found = sorted({m.group(1).lower() for m in _MEMO_HEADINGS.finditer(text)})
    if found:
        out.append(
            "a grounded answer carries section headings ("
            + ", ".join(f"{h.title()}:" for h in found)
            + ") -- it cites, it does not write itself up"
        )
    elif _CALL_LABEL.search(text):
        out.append("a grounded answer labels a call -- cite the figures and stop")
    return out


def makes_a_call(text: str) -> bool:
    """Whether this passage decides something, in either form a desk uses.

    Two forms, and missing the second is not hypothetical: the corpus gate
    records that reading only a phrase list cost three of four live ``ic_memo``
    rows their place, because the model wrote its decision under a ``Call:``
    heading -- which is not merely *a* way to make a call but the form the
    register's own table licenses.
    """
    lowered = str(text or "").casefold()
    if any(term in lowered for term in _CALL_TERMS):
        return True
    return bool(_CALL_LABEL.search(str(text or "")))


def register_violations(register: str, text: str, *, kind: str = "") -> list[str]:
    """Every way *text* breaks the shape *register* promises, as sentences.

    The harness-side reading of the corpus's register gate
    (``dataset/pipelines/v3/verification/register.py``), and a reimplementation
    rather than an import for the reason ``invented_numbers`` above is: the two
    trees are joined by published artifacts, not by Python (spec §3).

    Argument order matches the corpus function deliberately, so the two stay
    diffable against each other by eye.

    What this replaced, and why: the previous reading tested a *word-count band
    plus a heading boolean* -- ``ic_memo`` demanded 250 words and a heading.
    The corpus's own budgets top out at 220 words for an ``analysis`` row and
    300 for a ``memo``, and its gate calls headings "permitted, not required",
    so the floor sat above the ceiling: 0 of 36 shipped ``ic_memo`` rows could
    pass, and 3 of the 7 human-certified gold-bar rows were scored as failures.
    A base model that rambled for 2553 tokens under markdown headings outscored
    a tuned one writing to the corpus's own contract, which is the measurement
    inverting the thing it was built to detect.
    """
    text = str(text or "")
    if kind == "exam":
        return _exam_violations(text)
    out: list[str] = _grounded_violations(text) if kind == "grounded" else []
    if register == "desk_chat":
        if _MEMO_HEADINGS.search(text):
            out.append("register desk_chat wears memo headings")
        if _CALL_LABEL.search(text):
            out.append("register desk_chat labels its call")
        sentences = _sentence_count(text)
        ceiling = desk_chat_ceiling(kind)
        if sentences > ceiling:
            out.append(
                f"register desk_chat runs {sentences} sentences, over the "
                f"{ceiling}-sentence ceiling for a {kind or 'unlabelled'} row"
            )
        if sentences:
            mean = len(text.split()) / sentences
            if mean > DESK_CHAT_MAX_MEAN_SENTENCE:
                out.append(
                    f"register desk_chat averages {mean:.0f} words a sentence, "
                    f"over the {DESK_CHAT_MAX_MEAN_SENTENCE} ceiling"
                )
    elif register == "ic_memo":
        if kind not in CALL_EXEMPT_KINDS and not makes_a_call(text):
            out.append("register ic_memo states no call")
    elif register == "risk_committee":
        if not any(term in text.casefold() for term in _RISK_TERMS):
            out.append("register risk_committee names no limit, horizon or assumption")
        if _TRADE_CALL.search(text):
            out.append("register risk_committee recommends a trade")
    return out


def register_match(text: str, register: str, kind: str = "") -> bool | None:
    """Whether ``text`` is shaped like the register it was asked for.

    Fidelity, not length: does a desk reply stay unscaffolded and inside its
    sentence ceiling, does a memo reach a decision, does a risk paper name its
    constraint without taking the desk's position.

    ``kind`` is the record type. It carries the one exemption the corpus grants
    -- a ``grounded`` citation or an ``abstention`` refusal is an ``ic_memo``
    that legitimately ends without a call -- and defaults to ``""``, which is
    in no exempt set, so an unlabelled row is held to the full contract.

    ``None`` only when the register is absent or is not one the corpus
    declares, so such a row stays out of the denominator entirely -- a typo
    would otherwise read as a model regression. A *declared* register with no
    shape rules yet (``auditor``, ``code_review``) scores ``True``, which is
    what the corpus gate returns for it.
    """
    register = str(register or "").strip()
    if register not in VALID_REGISTERS:
        return None
    return not register_violations(register, text, kind=str(kind or ""))


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


#: The v3 factory's own signature. A *generation* that carries it means the
#: student learned the labelling protocol -- it is reproducing the brief it was
#: never supposed to see -- and that is the single most direct measurement of
#: whether the two-surface change (amendment §A) actually held. Kept as a
#: literal for the same reason the one in `data_schema` is: `jobs` does not
#: import `dataset`.
TEACHER_FINGERPRINT = "Cosimo v3 teacher"

#: The shapes a leaked brief takes even when the exact phrase does not survive
#: paraphrase. Each is a token of the *contract*, not of finance: a desk answer
#: has no reason to write "allowed_numbers", and a model that does is quoting
#: the JSON it was trained on.
_PROTOCOL_TELLS = (
    "allowed_numbers",
    "must_mention",
    "forbidden_claim",
    "fact_pack",
    "word_budget",
    "register_hint",
    "number_policy",
)


def teacher_leak(text: str) -> list[str]:
    """The factory tells this generation reproduces, in order. Empty is clean."""
    lowered = str(text or "").casefold()
    found = [tell for tell in _PROTOCOL_TELLS if tell.casefold() in lowered]
    if TEACHER_FINGERPRINT.casefold() in lowered:
        found.insert(0, TEACHER_FINGERPRINT)
    return found


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
            "teacher_leak_rate": 0.0,
            "teacher_leak_tells": {},
        }
    marker_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}
    tell_counts: dict[str, int] = {}
    for row in rows:
        for marker in row.get("exam_shape_markers", []):
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        for term in row.get("unknown_terms", []):
            term_counts[term] = term_counts.get(term, 0) + 1
        for tell in row.get("teacher_leak", []) or ():
            tell_counts[tell] = tell_counts.get(tell, 0) + 1

    # The three v3 metrics apply only to rows that declare the contract they
    # measure, so each carries its own denominator. Scoring an unlabelled row as
    # a failure would make adding a suite row look like a model regression.
    grounded = [r for r in rows if r.get("invented_numbers") is not None]
    mentioned = [r for r in rows if r.get("must_mention_missed") is not None]
    registered = [r for r in rows if r.get("register_match") is not None]
    return {
        "n": n,
        "exam_shape_rate": sum(1 for r in rows if r.get("exam_shape")) / n,
        "exam_shape_markers": dict(sorted(marker_counts.items())),
        "abstention_rate": sum(1 for r in rows if r.get("abstention")) / n,
        "mean_new_tokens": sum(float(r.get("new_tokens") or 0) for r in rows) / n,
        # Amendment §G's fourth number. Unlike the three above it needs no
        # per-row contract to be scored against -- every generation either
        # quotes the factory or does not -- so its denominator is the whole
        # suite. A non-zero rate is not a style finding; it is the corpus
        # having trained the labelling protocol, and it is the number that
        # says whether §A worked.
        "teacher_leak_rate": sum(1 for r in rows if r.get("teacher_leak")) / n,
        "teacher_leak_tells": dict(
            sorted(tell_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "unknown_term_rate": sum(1 for r in rows if r.get("unknown_terms")) / n,
        # Most frequent first: a term invented once is noise, a term invented in
        # thirty responses is a learned error.
        "unknown_terms": dict(
            sorted(term_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
        ),
        # Share of fact-locked answers containing at least one figure the
        # supplied numbers do not explain. The corpus refuses to publish a row
        # that fails this; the number here says whether the student learned the
        # constraint or only the prose around it.
        "invented_number_rate": (
            sum(1 for r in grounded if r["invented_numbers"]) / len(grounded)
            if grounded
            else 0.0
        ),
        "invented_numbers_n": len(grounded),
        # Per-answer share of the required terms that were actually named --
        # a mean of ratios, not an all-or-nothing count, so an answer that
        # covers three of four reads differently from one that covers none.
        "must_mention_hit_rate": (
            sum(
                len(r.get("must_mention_hit") or [])
                / (len(r.get("must_mention_hit") or []) + len(r["must_mention_missed"]))
                for r in mentioned
                if (
                    len(r.get("must_mention_hit") or []) + len(r["must_mention_missed"])
                )
            )
            / len(mentioned)
            if mentioned
            else 0.0
        ),
        "must_mention_n": len(mentioned),
        # Whether the answer is shaped like the voice it was asked for. A desk
        # question answered as a four-heading memo fails here.
        "register_match_rate": (
            sum(1 for r in registered if r["register_match"]) / len(registered)
            if registered
            else 0.0
        ),
        "register_match_n": len(registered),
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
        "hallucinated_tool_rate": sum(1 for r in rows if r.get("hallucinated_tools"))
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
