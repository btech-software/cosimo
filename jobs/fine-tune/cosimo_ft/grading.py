"""Answer extraction and grading.

Pure python (stdlib only). Every model answer we train on and every model answer
we grade follows the same ``FINAL ANSWER: <value>`` protocol, which is also given
to the base model, so base-vs-tuned comparisons are fair.

Formatting must never decide correctness: ``"$1,234.00"``, ``"1234"`` and
``"1,234.0"`` are the same answer, and ``"12.5%"`` equals ``"12.5"``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DEFAULT_TAG = "FINAL ANSWER:"

# Characters that decorate a value without changing it.
_TRIM_CHARS = "*`\"' \t\u00a0"
_CURRENCY = "$\u20ac\u00a3\u00a5\u20b9"
# 1,805,579.56 | 1805579.56 | 1805579 | 1.2e-3
# The lookbehind stops a hyphen that joins two tokens from being read as a minus
# sign: "Between 5-10 bp" is [5, 10], not [5, -10], and "2024-03-05" is not negative.
_NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z)])[-+\u2212]?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
# A leading option marker: "C", "C.", "(C)", "C)" , "C:"
# Leading option marker. An upper-case letter may stand alone ("FINAL ANSWER: C"),
# but a lower-case one must be followed by option punctuation \u2014 otherwise the
# English article in "a portfolio worth 51" reads as option A. This is the same
# upper-case-only rule _MCQ_ANYWHERE_RE applies below.
_MCQ_RE = re.compile(r"^\s*\(?([A-D])\)?\s*(?:[.):\-\u2013]|\s|$)")
_MCQ_LOWER_RE = re.compile(r"^\s*(?:\(([a-d])\)|([a-d])\s*[.):\-\u2013])")
# A standalone upper-case option letter anywhere in the text: "the answer is C".
# Upper case only, so the English article "a" is not read as option A.
_MCQ_ANYWHERE_RE = re.compile(r"(?<![0-9A-Za-z])([A-D])(?![0-9A-Za-z])")
_YESNO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
# Tokens that flip the meaning of a phrase they precede. "Increase" must not be
# graded correct against "will not increase" or "no increase is expected".
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "neither",
        "nor",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "wont",
        "wouldnt",
        "shouldnt",
        "except",
        "unless",
    }
)


@dataclass
class Grade:
    """Outcome of grading one generation."""

    correct: bool
    format_ok: bool
    mode: str
    pred: str | None
    matched_distractor: bool


def extract_final_answer(text: str, tag: str = DEFAULT_TAG) -> str | None:
    """Return the value on the last ``FINAL ANSWER:`` line, or None if absent.

    Matching is case-insensitive via ``re``: case folding a string is not
    length-preserving (``ß`` -> ``SS``), so indices taken from ``text.upper()``
    cannot be used to slice ``text``.
    """
    if not text or not tag:
        return None
    match = None
    for match in re.finditer(re.escape(tag), text, flags=re.IGNORECASE):
        pass
    if match is None:
        return None
    line = text[match.end() :].split("\n", 1)[0]
    value = line.strip().strip(_TRIM_CHARS).strip()
    return value or None


def last_nonempty_line(text: str) -> str | None:
    """Fallback prediction when the model omitted the tag."""
    if not text:
        return None
    for line in reversed(text.splitlines()):
        value = line.strip().strip(_TRIM_CHARS).strip()
        if value:
            return value
    return None


def parse_number(s: str | None) -> float | None:
    """Parse a single numeric value, tolerating currency, commas, percent, parens."""
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    text = (
        text.replace("\u2212", "-")  # unicode minus
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .replace("\u00a0", " ")  # nbsp
        .replace("\u202f", " ")  # narrow nbsp
        .replace("\u00d7", " ")  # multiplication sign
    )
    negative = False
    paren = re.fullmatch(r"\((.*)\)", text.strip())
    if paren:  # accounting negative: (1,234) == -1234
        negative = True
        text = paren.group(1)
    for symbol in _CURRENCY:
        text = text.replace(symbol, "")
    for symbol in (",", "_", "%", " "):
        text = text.replace(symbol, "")
    text = text.rstrip(".")
    if text.startswith("+"):
        text = text[1:]
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return -value if negative else value


def numbers_in(s: str | None) -> list[float]:
    """Every numeric literal in a string, in order of appearance."""
    if not s:
        return []
    values = []
    for match in _NUMBER_RE.finditer(str(s)):
        value = parse_number(match.group(0))
        if value is not None:
            values.append(value)
    return values


def numeric_close(
    a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-6
) -> bool:
    """Relative comparison with an absolute floor so values near zero still match."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def mcq_letter(s: str | None) -> str | None:
    """Return the upper-case option letter a string starts with, if any."""
    if not s:
        return None
    text = str(s)
    match = _MCQ_RE.match(text)
    if match:
        return match.group(1).upper()
    match = _MCQ_LOWER_RE.match(text)
    if match:
        return (match.group(1) or match.group(2)).upper()
    return None


def _strip_mcq_letter(s: str) -> str:
    """Drop a leading option marker, leaving the option's value.

    Only strips what ``mcq_letter`` would have recognised, so a prediction that
    merely begins with the article "a" keeps its text intact.
    """
    if mcq_letter(s) is None:
        return s
    return re.sub(r"^\s*\(?[A-Da-d]\)?\s*[.):\-\u2013]?\s*", "", s, count=1)


def mcq_letter_anywhere(s: str | None) -> str | None:
    """Last standalone option letter in a string, e.g. "the answer is C" -> "C".

    Base models phrase the final line this way far more often than tuned ones,
    so reading only a *leading* letter would turn a formatting habit into an
    accuracy difference.
    """
    if not s:
        return None
    matches = _MCQ_ANYWHERE_RE.findall(str(s))
    return matches[-1] if matches else None


def _predicted_letter(pred: str | None) -> str | None:
    """The option letter a prediction selects, leading marker or prose phrasing."""
    return mcq_letter(pred) or mcq_letter_anywhere(pred)


def _predicted_values(pred: str | None) -> list[float]:
    """Every plausible numeric reading of a prediction, best reading first."""
    if pred is None:
        return []
    values: list[float] = []
    whole = parse_number(pred)
    if whole is not None:
        values.append(whole)
    if mcq_letter(pred) is not None:
        stripped = parse_number(_strip_mcq_letter(pred))
        if stripped is not None and stripped not in values:
            values.append(stripped)
    if not values:
        # No clean single value: read right-to-left, the final answer first.
        values = list(reversed(numbers_in(pred)))
    return values


def _predicted_value(pred: str | None) -> float | None:
    """Best single numeric reading of a prediction."""
    values = _predicted_values(pred)
    return values[0] if values else None


def _distractor_values(distractors) -> list[float]:
    values = []
    for item in distractors or []:
        value = _predicted_value(str(item))
        if value is not None:
            values.append(value)
    return values


def _distractor_letters(distractors) -> list[str]:
    letters = []
    for item in distractors or []:
        letter = mcq_letter(str(item))
        if letter is not None:
            letters.append(letter)
    return letters


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .;:")


def _tokens(s: str) -> list[str]:
    return _WORD_RE.findall(_normalize_text(s))


def _contains_phrase(pred_tokens: list[str], gold_tokens: list[str]) -> bool:
    """True when gold occurs as a token run in pred without a negation in front."""
    span = len(gold_tokens)
    for start in range(len(pred_tokens) - span + 1):
        if pred_tokens[start : start + span] != gold_tokens:
            continue
        if start > 0 and pred_tokens[start - 1] in _NEGATIONS:
            continue  # "will not increase" does not answer "Increase"
        return True
    return False


def _resolve_pred(generation: str, tag: str) -> tuple[str | None, bool]:
    """Extract the prediction and whether the required tag was present."""
    pred = extract_final_answer(generation, tag)
    if pred is not None:
        return pred, True
    return last_nonempty_line(generation), False


def _grade_prose(gold: str, pred: str, rel_tol: float) -> bool:
    gold_numbers = numbers_in(gold)
    pred_numbers = numbers_in(pred)
    gold_yesno = _YESNO_RE.match(gold)
    if gold_yesno:
        pred_yesno = _YESNO_RE.match(pred)
        if not pred_yesno or pred_yesno.group(1).lower() != gold_yesno.group(1).lower():
            return False
    if gold_numbers:
        return all(
            any(numeric_close(g, p, rel_tol) for p in pred_numbers)
            for g in gold_numbers
        )
    if gold_yesno:
        return True
    normalized_gold = _normalize_text(gold)
    if not normalized_gold:
        return False
    if normalized_gold == _normalize_text(pred):
        return True
    gold_tokens = _tokens(gold)
    return bool(gold_tokens) and _contains_phrase(_tokens(pred), gold_tokens)


def grade_cosimo(
    record: dict,
    generation: str,
    tag: str = DEFAULT_TAG,
    rel_tol: float = 1e-3,
) -> Grade:
    """Grade one Cosimo record.

    ``record`` needs ``answer``; ``question_type`` and ``distractors`` are used
    when present. A missing ``FINAL ANSWER:`` tag sets ``format_ok=False`` but
    the last non-empty line is still graded, so accuracy and format compliance
    are reported independently.
    """
    gold = str(record.get("answer") or "").strip()
    question_type = str(record.get("question_type") or "").strip()
    pred, format_ok = _resolve_pred(generation, tag)

    if question_type.upper() == "MCQ":
        mode = "mcq"
    elif parse_number(gold) is not None:
        mode = "numeric"
    else:
        mode = "prose"

    if pred is None:
        return Grade(False, format_ok, mode, None, False)

    pred_letter = None
    if mode == "mcq":
        gold_letter = mcq_letter(gold)
        pred_letter = _predicted_letter(pred)
        gold_value = (
            parse_number(_strip_mcq_letter(gold)) if gold_letter else parse_number(gold)
        )
        pred_value = _predicted_value(pred)
        correct = bool(gold_letter and pred_letter and gold_letter == pred_letter)
        if not correct and gold_value is not None and pred_value is not None:
            correct = numeric_close(pred_value, gold_value, rel_tol)
    elif mode == "numeric":
        gold_value = parse_number(gold)
        pred_value = _predicted_value(pred)
        correct = pred_value is not None and numeric_close(
            pred_value, gold_value, rel_tol
        )
    else:
        pred_value = _predicted_value(pred)
        correct = _grade_prose(gold, pred, rel_tol)

    distractors = record.get("distractors")
    matched_distractor = False
    if not correct and pred_value is not None:
        matched_distractor = any(
            numeric_close(pred_value, d, rel_tol)
            for d in _distractor_values(distractors)
        )
    if not correct and not matched_distractor and mode == "mcq" and pred_letter:
        # A bare-letter answer ("D") must count as falling for the pitfall exactly
        # like the spelled-out option ("D. 51"), or distractor_rate would improve
        # for free when a model merely changes how it phrases the final line.
        matched_distractor = pred_letter in _distractor_letters(distractors)
    return Grade(correct, format_ok, mode, pred, matched_distractor)


def _normalize_math_gold(s: str) -> str:
    """Light LaTeX normalisation for MATH-500 style answers."""
    text = str(s).strip()
    boxed = re.search(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if boxed:
        text = boxed.group(1)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right|!|,|;|\s)", "", text)
    text = text.replace("$", "").replace(" ", "")
    text = text.rstrip(".")
    return text.lower()


def grade_math(
    gold: str,
    generation: str,
    tag: str = DEFAULT_TAG,
    rel_tol: float = 1e-3,
) -> Grade:
    """Grade a GSM8K / MATH-500 item: numeric when both sides parse, else string."""
    pred, format_ok = _resolve_pred(generation, tag)
    if pred is None:
        return Grade(False, format_ok, "numeric", None, False)

    gold_value = parse_number(gold)
    if gold_value is not None:
        # A restated final line ("We need 18 eggs, so 3 remain") must not be graded
        # on its trailing number alone, so every numeric reading is considered.
        correct = any(
            numeric_close(value, gold_value, rel_tol)
            for value in _predicted_values(pred)
        )
        return Grade(correct, format_ok, "numeric", pred, False)

    normalized_gold = _normalize_math_gold(gold)
    normalized_pred = _normalize_math_gold(pred)
    correct = bool(normalized_gold) and normalized_gold == normalized_pred
    return Grade(correct, format_ok, "string", pred, False)
