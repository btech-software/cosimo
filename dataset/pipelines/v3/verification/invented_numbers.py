"""The invented-number gate (spec §6 axis 4; §5.6's teacher contract in code).

The teacher may *phrase*, never *quantify*: every number in a completion must
trace to the pack's ``allowed_numbers``. This module is that rule, executable,
and it is deliberately shared between the renderer's repair loop (PR2), the
verify board and the publish-time audit slice -- one policy, three call sites.

The rounding policy, stated once because a corpus is only as reproducible as
its definitions of "same number":

* a token is taken with :func:`verification.nums.val` -- thousands separators
  and decimals handled the way the v2 verifier already handles them;
* it passes when, after scaling by ``1``, ``100`` or ``1/100``, it lands
  within ``REL_TOLERANCE`` (0.5%) of an allowed value. The two scalings are
  the two honest ways prose renders a pack value: ``0.03`` as ``3.00%``,
  ``12.5`` as a ``0.125`` ratio; nothing more permissive, or the gate would
  stop being one -- a model cannot rescue a wrong figure by reformatting it;
* comparison is relative, not a fixed epsilon, because packs mix ``1e4``
  AUMs with ``1e-4`` spreads; one epsilon would let either class pass
  everything or reject its own outputs;
* and beside the relative test, a token also passes when it is an allowed
  value *correctly rounded to the precision the token itself carries* -- the
  desk writing ``-0.42%`` of a ``-0.416``. That path is exact, not a widened
  band, and needs at least one decimal: a relative test alone makes the
  standard a function of magnitude, so the same two-decimal rounding passed at
  3.309 and failed at 0.416 until this was added.

Fail-closed: an unparsable or non-finite allowed set is an error, never a
silent "nothing invented" -- the gate missing its data must look like the gate
failing, per the v1/v2 precedent in :mod:`verification.nums`.
"""

from __future__ import annotations

import math
import re
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verification import nums  # noqa: E402

REL_TOLERANCE = 0.005

#: Float-comparison slack for the desk-rounding test. Not a tolerance on
#: the *figure* -- that test is exact by construction -- only on the binary
#: representation of a rounding both sides computed.
_EXACT = 1e-9

#: Dashes a writer means as a minus, folded before tokenizing. U+2212 is the
#: true minus and U+2013 the en dash a model reaches for; the em dash (U+2014)
#: is deliberately absent, being punctuation far more often than arithmetic.
#: Only before a digit, so a dash between words stays a dash.
_ASCII_MINUS = re.compile("[\u2212\u2013](?=\\d)")
#: 1 (bare), 100 (fraction rendered as percent), 1/100 (percent rendered as
#: fraction). Any other factor is an invention, which is exactly what a unit
#: hallucination looks like numerically: a 1000x slip is *not* the same number.
SCALE_FACTORS = (1.0, 100.0, 0.01)


def decimal_places(token: str) -> int:
    """Decimal places a *written* number token carries, as the desk reads it.

    On the token as spelled, not on its float value: ``"0.472041725693"`` is
    twelve places whether or not the pack stores it that way, and ``"3.0"`` is
    one. Trailing zeros count, because a figure spelled ``0.10`` claims two
    places of precision and a reader is entitled to believe it.

    Shared so the two rules that care about precision cannot drift: the prose
    gate's absolute ceiling (:data:`config.PROSE_MAX_DECIMALS`) and the
    preference lane's relative over-quoting detector measure depth the same way.
    """
    token = str(token).strip().lstrip("+-").replace(",", "")
    if "." not in token:
        return 0
    return len(token.split(".", 1)[1])


def read_tokens(text: str) -> list[str]:
    """Number tokens in *text*, with hyphens that are not minus signs undone.

    The shared tokenizer treats a leading hyphen as part of the number, so any
    hyphen
    immediately before a digit becomes that number's sign. In prose it usually
    is not one: "a 1-in-100 day" yields ``1`` and ``-100``, and a live VaR row
    was dead-lettered three times over for an invented ``-100`` it never wrote.
    The same applies to "a 1-in-20 event" and to any written range.

    A leading sign counts as a sign only when what precedes it is not a word
    character. Fixed here rather than in :mod:`verification.nums`, which the
    v1 and v2 verifiers also read: their corpora are already scored, and a
    tokenizer change would silently restate those results.

    The other half is typography. A teacher writing properly uses a real
    minus, and the tokenizer only knows ASCII ``-``: one live attribution memo
    used U+2013 seventeen times, so *every* negative figure in it read as
    positive and seven pack values were refused as inventions. En dash and
    U+2212 are folded to ``-`` before a digit; the em dash is left alone,
    because the same row used five of those as punctuation and "the cost -- 28
    bp -- was high" does not mean minus twenty-eight.
    """
    out: list[str] = []
    raw = _ASCII_MINUS.sub("-", str(text))
    for match in nums.TOKEN.finditer(raw):
        token = match.group(0)
        start = match.start()
        if token[0] in "-+" and start > 0 and (raw[start - 1].isalnum()):
            token = token[1:]  # a hyphen inside a word, not a sign
        out.append(token)
    return out


def _allowed_set(allowed) -> frozenset[float]:
    values = frozenset(float(a) for a in allowed if a is not None)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(
            "invented_numbers needs a non-empty allowed set of finite values; "
            "refusing to grade with a missing or corrupt contract"
        )
    return values


def _matches(value: float, allowed: frozenset[float], decimals: int = -1) -> bool:
    """Does any allowed value explain *value*, written to *decimals* places?

    Two acceptances, and the second is why *decimals* is here.

    The relative test is the original policy and stays exactly as it was.
    Beside it sits the desk-rounding test: a token is the allowed value when it
    is that value *correctly rounded to the precision the token itself
    carries*. Both are needed because the relative test makes the standard a
    function of magnitude, and desk rounding is not.

    Measured on one live attribution memo: ``3.31`` for a ``3.309`` passed
    (0.03% relative), while ``-0.42`` for a ``-0.416`` and ``-0.08`` for a
    ``-0.084`` were refused (1% and 5%). All three are the same act -- a desk
    writing a percentage to two places -- and the corpus was refusing two of
    them for being small. A row cannot report a small figure in the register's
    own voice without tripping a gate, which is not a standard a writer can
    satisfy.

    Deliberately not a widening of the relative band: this path demands the
    rounding be *exact* at the written precision, so it admits no figure a
    reader could tell apart from the pack's. It requires at least one decimal,
    which leaves whole numbers where they were -- ``373`` for a ``372.6`` is
    :func:`verification.prose.rounding_drift`'s to refuse, and it still does.
    """
    for scale in SCALE_FACTORS:
        scaled = value * scale
        for a in allowed:
            if a == 0.0:
                if scaled == 0.0:
                    return True
                continue
            if abs(scaled - a) <= REL_TOLERANCE * abs(a):
                return True
            # Compared in the *token's* space, not the pack's: the written
            # decimals describe `value`, so the allowed figure has to be
            # brought back through the same scale before it is rounded.
            if decimals >= 1 and abs(round(a / scale, decimals) - value) <= _EXACT:
                return True
    return False


def invented_numbers(text: str, allowed, whitelist=()) -> list[str]:
    """Raw tokens in *text* no allowed value explains, in order of appearance.

    Returns the *source strings* ("17.4%"), not floats: the repair prompt and
    the dead-letter record must show the operator what the model wrote, not
    what the gate parsed it into.

    ``whitelist`` holds *token strings* the caller declares arithmetic
    furniture rather than pack facts ("100", the as-of year). Matching is on
    the written form, commas and all, because the mercy is about what the
    reader sees, not about what the value means; keep it tiny (see
    ``config.NUMBER_WHITELIST`` -- the analysis spec's "tiny whitelist" rule)
    or it stops being a gate.
    """
    allowed = _allowed_set(allowed)
    # Token edges are typography, not meaning: "31," at the end of a date and
    # "31" in mid-sentence are the same permission. Interior commas stay --
    # "1,000" is one token and must match as one.
    passed = frozenset(str(w).strip().strip(",") for w in whitelist)
    offenders: list[str] = []
    seen: set[str] = set()
    for tok in read_tokens(text):
        if tok in seen:
            continue
        if tok.strip(",") in passed:
            continue
        try:
            value = nums.val(tok)
        except ValueError:
            offenders.append(tok)  # unparsable number-ish text is not "no number"
            seen.add(tok)
            continue
        if not math.isfinite(value) or not _matches(
            value, allowed, decimal_places(tok)
        ):
            offenders.append(tok)
            seen.add(tok)
    return offenders


def clean(text: str, allowed, whitelist=()) -> bool:
    """Convenience for assertions and repair loops; see :func:`invented_numbers`."""
    return not invented_numbers(text, allowed, whitelist)
