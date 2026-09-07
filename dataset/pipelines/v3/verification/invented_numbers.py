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
* comparison is relative, not decimal-place based, because packs mix ``1e4``
  AUMs with ``1e-4`` spreads; a fixed epsilon would let either class pass
  everything or reject its own outputs.

Fail-closed: an unparsable or non-finite allowed set is an error, never a
silent "nothing invented" -- the gate missing its data must look like the gate
failing, per the v1/v2 precedent in :mod:`verification.nums`.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verification import nums  # noqa: E402

REL_TOLERANCE = 0.005
#: 1 (bare), 100 (fraction rendered as percent), 1/100 (percent rendered as
#: fraction). Any other factor is an invention, which is exactly what a unit
#: hallucination looks like numerically: a 1000x slip is *not* the same number.
SCALE_FACTORS = (1.0, 100.0, 0.01)


def _allowed_set(allowed) -> frozenset[float]:
    values = frozenset(float(a) for a in allowed if a is not None)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(
            "invented_numbers needs a non-empty allowed set of finite values; "
            "refusing to grade with a missing or corrupt contract"
        )
    return values


def _matches(value: float, allowed: frozenset[float]) -> bool:
    for scale in SCALE_FACTORS:
        scaled = value * scale
        for a in allowed:
            if a == 0.0:
                if scaled == 0.0:
                    return True
                continue
            if abs(scaled - a) <= REL_TOLERANCE * abs(a):
                return True
    return False


def invented_numbers(text: str, allowed) -> list[str]:
    """Raw tokens in *text* no allowed value explains, in order of appearance.

    Returns the *source strings* ("17.4%"), not floats: the repair prompt and
    the dead-letter record must show the operator what the model wrote, not
    what the gate parsed it into.
    """
    allowed = _allowed_set(allowed)
    offenders: list[str] = []
    seen: set[str] = set()
    for tok in nums.TOKEN.findall(str(text)):
        if tok in seen:
            continue
        try:
            value = nums.val(tok)
        except ValueError:
            offenders.append(tok)  # unparsable number-ish text is not "no number"
            seen.add(tok)
            continue
        if not math.isfinite(value) or not _matches(value, allowed):
            offenders.append(tok)
            seen.add(tok)
    return offenders


def clean(text: str, allowed) -> bool:
    """Convenience for assertions and repair loops; see :func:`invented_numbers`."""
    return not invented_numbers(text, allowed)
