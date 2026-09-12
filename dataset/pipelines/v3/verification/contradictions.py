"""Claims the pack's own numbers refute (amendment §D).

Every other axis in this package asks whether a figure came from the pack, or
whether a shape matches a register. None of them can ask the question a reader
asks first: *does this answer contradict the arithmetic it is quoting?* The
nine-row live sample is the argument. Every row was board-green, and:

* a 4.86% clip against a 10% cap was written up as a pacing problem -- "the
  schedule itself becomes the risk" -- in the same paragraph that said it was
  well under the cap;
* a book with a -0.10% daily mean was reported as carrying a positive drift
  worth $10M of expected gain, because the minus sign was read as a subtraction
  rather than as a loss;
* an attribution whose pieces missed the active return by 371 bp still named
  the effect worth acting on, with a story about the Carino factor to explain
  why the pieces looked wrong.

None of those invented a number. All three are refuted by a comparison the
pack can make against itself, and that is what this module does: read the
canonical figures, run the inequality, and if the answer says the opposite,
say so in a sentence the repair turn can act on.

Each violation opens with a tag (:data:`TAGS`) so the board and ``slice_audit``
can file it without re-parsing English, and so a shipped corpus can be swept
for the three failures that were invisible to sixteen axes.
"""

from __future__ import annotations

import re

#: ``tag`` -> what it means, for the board's report and for anyone reading a
#: dead letter six months from now.
TAGS = {
    "schedule_risk_below_cap": (
        "the answer calls the schedule the risk while participation sits under "
        "the pack's cap"
    ),
    "drift_sign": (
        "the answer reads a negative daily mean as a gain or a positive drift"
    ),
    "unreconciled_call": (
        "the answer picks an effect to act on while the pieces do not add to "
        "the active return"
    ),
    "ev_as_price": (
        "the answer makes an ownership call from a valuation that carries no "
        "market price to compare it against"
    ),
}

#: Schedule-as-risk language. Loose on the verb ("becomes", "is", "will be")
#: and tight on the pairing: a sentence has to put the schedule and the risk
#: together for this to fire, so "work the schedule patiently" and "the risk is
#: the impact" both pass.
_SCHEDULE_RISK = re.compile(
    r"\b(?:the\s+)?(?:schedule|pacing|timing)\b[^.;]{0,60}?\b"
    r"(?:is|becomes|turns into|as)\b[^.;]{0,30}?\b(?:the\s+)?"
    r"(?:risk|binding constraint|constraint|problem)\b",
    re.IGNORECASE,
)

#: The same claim written the other way round -- "the risk is the schedule".
_RISK_IS_SCHEDULE = re.compile(
    r"\b(?:the\s+)?(?:risk|binding constraint|constraint)\b[^.;]{0,40}?\bis\b"
    r"[^.;]{0,30}?\b(?:the\s+)?(?:schedule|pacing|timing)\b",
    re.IGNORECASE,
)

#: A negative mean read as something good. "Expected gain" and "positive drift"
#: are the amendment's two; "drift in your favour" is the same sentence with
#: the desk's words.
_GAIN_LANGUAGE = re.compile(
    r"\bpositive\s+drift\b|\bexpected\s+gain\b|\bdrift\s+(?:is\s+)?in\s+(?:your|its|the book's)\s+favou?r\b"
    r"|\bmean\s+(?:return\s+)?(?:is\s+)?positive\b",
    re.IGNORECASE,
)

#: Deciding which effect to act on. The pack's question asks for it, so the
#: words are the ones an answer to that question uses.
_ACTS_ON_AN_EFFECT = re.compile(
    r"\bskill\s+signal\b|\b(?:worth|effect)\s+acting\s+on\b|\bact\s+on\s+"
    r"(?:the\s+)?(?:allocation|selection|interaction)\b|\b(?:allocation|selection|"
    r"interaction)\s+is\s+the\s+(?:one|effect)\s+to\s+act\b",
    re.IGNORECASE,
)

#: How far the printed pieces may sit from the printed active before an answer
#: may not pick a winner among them. The amendment says 0.2 bp; the pieces are
#: each published to a tenth, so three of them plus the total can differ from
#: the arithmetic by 0.2 bp of pure rounding. Half a basis point is above every
#: rounding artefact a pack can produce (measured worst case: 0.1) and two
#: orders of magnitude below the failure it is looking for (371 bp).
RECONCILE_TOLERANCE_BPS = 0.5


#: Words that turn a claim into its denial. Kept local rather than imported
#: from :mod:`verification.prose`: that module imports *this* one, and the two
#: readings are not the same anyway -- there the negation must precede the
#: claim ("do not assume the mid is achievable"), here it usually sits inside
#: it ("the schedule is *not* the risk"), which is the sentence §F wants
#: written.
#:
#: True negators only, matched as *words*. An earlier draft also carried
#: "only", "would be" and "until" to excuse a conditional -- "only near the cap
#: does the schedule become the risk" -- and that was a hole rather than a
#: mercy: "the schedule is the only risk that matters" and "it would be a
#: mistake to think otherwise: the schedule is the risk here" both assert the
#: claim plainly and both were silently excused by a substring of a neighbour.
#:
#: The conditional readings do not need the excuse. The patterns below match
#: the assertive present tense ("is the risk", "becomes the risk"), so a
#: sentence written in the conditional -- "would become", "does ... become" --
#: does not match them in the first place. Where a conditional phrasing does
#: trip the gate, the cost is one repair turn on a row that survives; the cost
#: of the reverse is a contradicted row shipping board-green, which is the
#: whole reason this file exists.
_NEGATORS = frozenset(
    {
        "not",
        "never",
        "no",
        "nor",
        "neither",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        # The denial does not always carry a "not": "nothing says whether it is
        # worth owning near this EV" is the sentence §B.4 asks a valuation row
        # to write, and it says the claim is unsupported without once negating
        # a verb. "whether" earns its place the same way -- it marks an open
        # question, and an open question is not an assertion.
        "nothing",
        "none",
        "without",
        "whether",
        "unclear",
    }
)

#: Two-word denials, which a word set cannot hold.
_NEGATING_PHRASES = ("rather than", "n't")

#: How many words before a match are read for a denial, for the form that puts
#: it in front: "we would not say the schedule is the risk".
_NEGATION_WINDOW = 6


def _words(text: str) -> list[str]:
    return [word.strip(",.;:()'\"") for word in text.casefold().split()]


#: Where one clause ends and the next begins. The window stops here, because a
#: denial belongs to the clause it is spoken in: a live row wrote "the impact
#: is the cost being managed here, not the schedule; the schedule itself
#: becomes the risk only at or above the cap" and the ``not`` of the first
#: clause excused the claim in the second. The sentence happens to be true --
#: it is the conditional §F wants -- but it passed by laundering rather than by
#: saying so, and the next row phrased the same way will not be true.
_CLAUSE_BREAK = re.compile(r"[;:.!?,]")


def _negated(text: str, match: re.Match) -> bool:
    """Is this match a denial of the claim rather than the claim?"""
    span = match.group(0).casefold()
    before = text[: match.start()].casefold()
    clause = _CLAUSE_BREAK.split(before)[-1]
    window = " ".join(clause.split()[-_NEGATION_WINDOW:])
    if any(phrase in span or phrase in window for phrase in _NEGATING_PHRASES):
        return True
    return bool(_NEGATORS & set(_words(span) + _words(window)))


def _asserted(text: str, *patterns: re.Pattern) -> bool:
    """Does *text* make this claim plainly -- unnegated -- anywhere?"""
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not _negated(text, match):
                return True
    return False


def _canonical(pack: dict) -> dict:
    canonical = pack.get("canonical") or {}
    if canonical:
        return canonical
    # A fixture pack that declares no canonical map is graded on what it has:
    # the flattened inputs and computed figures are where these quantities live
    # in a real pack anyway.
    merged = dict(pack.get("inputs") or {})
    merged.update(pack.get("computed") or {})
    return merged


def _number(canonical: dict, name: str) -> float | None:
    value = canonical.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def schedule_risk_below_cap(pack: dict, text: str) -> str:
    """§D row one: the slogan, against the inequality that refutes it."""
    canonical = _canonical(pack)
    participation = _number(canonical, "participation")
    cap_pct = _number(canonical, "participation_cap_pct")
    if participation is None or cap_pct is None:
        return ""
    if participation >= cap_pct / 100.0:
        return ""  # at the cap the schedule *is* the risk; the pack says so
    if not _asserted(text, _SCHEDULE_RISK, _RISK_IS_SCHEDULE):
        return ""
    return (
        f"schedule_risk_below_cap: participation is {participation * 100:.2f}% "
        f"against a {cap_pct:.0f}% cap, so the schedule is not the risk here -- "
        "this is an impact bill, not a pacing problem; say which of the two the "
        "cost is and do not claim the other"
    )


def drift_sign(pack: dict, text: str) -> str:
    """§D row two: a negative mean is a negative drift, and never a gain."""
    mean = _number(_canonical(pack), "mu_daily")
    if mean is None or mean >= 0:
        return ""
    if not _asserted(text, _GAIN_LANGUAGE):
        return ""
    return (
        f"drift_sign: the daily mean is {mean}, a negative drift that adds to "
        "the expected loss -- it is not a gain and not a positive drift; the "
        "VaR formula subtracts it for that reason"
    )


def unreconciled_call(pack: dict, text: str) -> str:
    """§D row three: no naming a winner among pieces that do not add up."""
    canonical = _canonical(pack)
    active = _number(canonical, "active_bps")
    legs = [
        _number(canonical, key)
        for key in (
            "allocation_total_bps",
            "selection_total_bps",
            "interaction_total_bps",
        )
    ]
    if active is None or any(leg is None for leg in legs):
        return ""
    residual = sum(legs) - active  # type: ignore[arg-type]
    if abs(residual) <= RECONCILE_TOLERANCE_BPS:
        return ""
    if not _asserted(text, _ACTS_ON_AN_EFFECT):
        return ""
    return (
        f"unreconciled_call: allocation + selection + interaction miss the "
        f"{active} bp active return by {residual:.1f} bp, so no single effect "
        "can be the one to act on -- report that the figures do not reconcile "
        "and abstain on the call"
    )


#: Deciding to own, or not to own, a name. The move a valuation row is *asked*
#: for -- the DCF question opens "Should we own X?" -- and the one its figures
#: cannot support on their own.
_OWNERSHIP_CALL = re.compile(
    r"\bown\s+it\b|\bworth\s+owning\b|\bwe\s+would\s+(?:own|buy|sell)\b"
    r"|\b(?:looks|is)\s+(?:cheap|expensive|undervalued|overvalued)\b"
    r"|\brecommend\s+(?:owning|buying|selling)\b|\bbuy\s+the\s+name\b"
    # The position frame, which is the same call in the language a desk
    # actually uses. A live DCF analysis closed "Call: hold off on a position
    # until we can compare this EV to market capitalization" and the axis let
    # it through, because it was written to catch a slogan rather than a move.
    r"|\b(?:take|open|initiate|build|size|hold\s+off\s+on|wait\s+(?:on|for|before))"
    r"\s+(?:a|the|any)?\s*position\b"
    r"|\bposition\s+(?:until|once|when)\b",
    re.IGNORECASE,
)

#: Comparing an enterprise value with a market capitalisation. Not a call, and
#: worse than one: EV carries net debt and market cap does not, so the
#: comparison is wrong even when the price it wants exists. The honest sentence
#: names what is missing -- "no share price, so no ownership call" -- and does
#: not propose an arithmetic nobody should do.
_EV_AGAINST_PRICE = re.compile(
    r"\b(?:ev|enterprise\s+value)\b[^.;]{0,70}?\b(?:compare[d]?|against|versus|vs\.?|"
    r"relative\s+to)\b[^.;]{0,40}?\b(?:market\s+cap\w*|share\s+price|price)\b"
    r"|\bcompare\s+(?:this|the)\s+ev\b[^.;]{0,50}?\b(?:market\s+cap\w*|price)\b",
    re.IGNORECASE,
)

#: What a pack needs before an ownership call is possible: a market price to
#: compare the valuation against. An *implied* value per share is the answer's
#: own output, not a market quote, so it does not license the call -- that
#: confusion is the whole of "EV as price".
_PRICE_KEYS = ("market_price", "share_price", "price_per_share", "last_price", "px")


def ev_as_price(pack: dict, text: str) -> str:
    """§B.4's valuation rule: no market price, no ownership call.

    A live DCF analysis closed "**Call:** Own it, because the implied EV is
    reasonable" over a pack with no price and no share count -- board-green,
    and the exact error the work-type addendum was written to prevent. The
    addendum told the teacher; nothing checked it.

    The memo on the same pack got it right ("there is no ownership call here --
    with no share price or share count given, we cannot say whether it is worth
    owning near this EV"), which is the sentence this axis asks for.
    """
    work_type = str(pack.get("work_type") or "")
    if not work_type.startswith("valuation."):
        return ""
    canonical = _canonical(pack)
    if any(any(key in name for key in _PRICE_KEYS) for name in canonical):
        return ""
    if _asserted(text, _EV_AGAINST_PRICE):
        return (
            "ev_as_price: the answer proposes weighing the enterprise value "
            "against a market price or market capitalisation -- EV carries net "
            "debt and a market cap does not, so that comparison is wrong even "
            "where the price exists, and this pack carries no price at all"
        )
    if not _asserted(text, _OWNERSHIP_CALL):
        return ""
    return (
        "ev_as_price: this valuation carries no market price to compare, so "
        "nothing here supports owning, waiting on, or sizing a position -- an "
        "enterprise value is not a share price; say there is no ownership call"
    )


def contradiction_violations(pack: dict, text: str) -> list[str]:
    """Every claim this pack's own numbers refute, as tagged sentences.

    Cheap by design -- three comparisons and four regexes -- because it runs
    once per attempt inside the repair loop, again on the board, and again
    over a whole slice.
    """
    text = str(text or "")
    if not text.strip():
        return []
    found = [
        schedule_risk_below_cap(pack, text),
        drift_sign(pack, text),
        unreconciled_call(pack, text),
        ev_as_price(pack, text),
    ]
    return [line for line in found if line]


def tag_of(violation: str) -> str:
    """The tag a violation opens with, or ``""`` if it carries none."""
    head = str(violation or "").split(":", 1)[0].strip()
    return head if head in TAGS else ""


__all__ = [
    "TAGS",
    "RECONCILE_TOLERANCE_BPS",
    "contradiction_violations",
    "drift_sign",
    "ev_as_price",
    "schedule_risk_below_cap",
    "tag_of",
    "unreconciled_call",
]
