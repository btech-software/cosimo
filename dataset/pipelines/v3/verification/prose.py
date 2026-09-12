"""The prose contract, executable (analysis spec §6.3; arch spec §5.5, §6).

One function, :func:`gate_violations`, is the whole difference between "the
teacher wrote something" and "the corpus may carry it": invented numbers
(against the pack's ``canonical`` map, with the tiny structural whitelist),
rounding drift, integer spelling, ``must_mention`` coverage,
``forbidden_claims``, the exam-only ``FINAL ANSWER:`` tag, the word budget, and
-- since the amendment -- the shape the row's register promises, the length its
kind and register allow, and the claims its own pack refutes. The renderer's
repair loop calls it per attempt; ``verify_v3`` calls the same helpers over the
written shards; the publish-time audit slice calls the pack-aware ones again.
One policy, three call sites -- the v1/v2 failure of having the generator and
the auditor disagree about "clean" is exactly the bug class v3 was written to
kill, and it came back once already: the board graded the register's sentence
ceiling and not the kind's, and no word band at all, so a row repaired here for
running long was certified green there.

Coverage is *all* points, not "at least k": the pack's contract says cover
every ``must_mention``, and a k-threshold is a dial nobody would defend at a
review. A row missing a point is repaired, not bartered.

What changed after the first live run is *what a point is*. Matching was a
verbatim substring test over an English clause, and it failed correct answers
on articles and word order -- see :func:`missing_mentions`. Points are now
short **anchors** of content words, matched on stems, in any order, ignoring
function words. Still no synonym table: the gate does not guess at meaning it
was never given, it only stops insisting on one conjugation and one sentence
shape. The analytical *depth* an anchor stands for is driven by the brief and
checked by human review, not by a string test that cannot see it.
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # verification; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verification import nums  # noqa: E402
from verification.gates import FINAL_ANSWER_TAG  # noqa: E402

from .. import config  # noqa: E402
from ..config import NUMBER_WHITELIST  # noqa: E402
from ..teacher.prompts import KIND_SENTENCE_CAPS  # noqa: E402
from ..teacher.prompts import word_budget  # noqa: E402
from .contradictions import contradiction_violations  # noqa: E402
from .invented_numbers import decimal_places  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402
from .invented_numbers import read_tokens  # noqa: E402
from .register import register_violations, sentence_count  # noqa: E402


def whitelist_for(pack: dict) -> frozenset[str]:
    """The token strings the pack itself authorises outside ``allowed_numbers``.

    ``config.NUMBER_WHITELIST`` (percent denominators, the two of a two-way
    bridge, the trading-day convention) plus every number spelled in the
    pack's ``as_of`` date -- a row is allowed to *date* its analysis, and the
    date is the pack's own, drawn from the seeded pool, so leaking it through
    the whitelist re-authorises nothing the pack did not already state.
    """
    # Read with the same reader the answer is read with. `2026-06-30` is three
    # tokens, and which three depends on whether a hyphen counts as a sign --
    # so a whitelist built by one reader and checked by another authorises
    # spellings that never appear, and refuses the ones that do.
    as_of_tokens = read_tokens(str(pack.get("as_of") or ""))
    return frozenset(NUMBER_WHITELIST) | frozenset(as_of_tokens)


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


#: Suffixes stripped before an anchor is matched, longest first. English
#: inflection only, and only enough of it to stop a *correct* answer failing on
#: grammar: a pack anchored on "scaling" must accept "scales", one anchored on
#: "normality" must accept "normal". No stemmer library and no synonym table --
#: the gate still refuses to guess at meaning it was never given, it just stops
#: insisting on a particular conjugation.
#: Longest first, so "-ption" is tried before the "-ion" inside it: the desk
#: nominalises in an anchor ("normality assumption") and conjugates in the
#: prose ("it assumes"), and stripping only "-ion" leaves "assumpt" against
#: "assum" -- which cost a correct VaR answer its row on the first five-lane
#: live run.
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
    # Last, and the one that makes a verb match itself. "scaling" loses "ing"
    # and reaches "scal"; "scale" matched nothing and stayed whole, so an
    # anchor on "square-root horizon scaling" was refused three times over by
    # a live row that wrote "the 10-day figures scale by the square-root of
    # the horizon" -- every term present, one of them merely conjugated.
    # "assume"/"assumption" was the same story, which is the pairing this
    # module's own docstring already claims to handle.
    #
    # Safe because the loop keeps a three-character floor: "the", "are" and
    # "one" are shorter than that and come through untouched.
    "e",
)


def _stem(word: str) -> str:
    """Strip suffixes until the word stops changing.

    One pass is not enough once "-ption" is in the table: "assumptions" would
    lose its "s" and stop, never reaching the "assum" that "assumes" reduces
    to. Looping is what makes the singular and the plural of a nominalisation
    land in the same place.
    """
    while True:
        for suffix in _SUFFIXES:
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        else:
            return word


#: Function words carry no analytical content, so requiring them would make an
#: anchor fail on grammar rather than on substance -- "arrival versus decision
#: benchmark" must match an answer that writes "an arrival benchmark; a
#: decision-price benchmark would ...". Only these are dropped; every content
#: word of an anchor is still required.
_STOPWORDS = frozenset(
    """a an the and or nor but of in on to for with by at from as is are was
    were be been being it its this that these those than then so such not no
    versus vs against near over under into via per about""".split()
)


def stems(text: str) -> set[str]:
    """Content-word stems of *text*, hyphens treated as spaces.

    Public because the preference fixture maker has to silence a
    ``must_mention`` point the way this gate reads one; a second copy of the
    rule there would drift from this one, and a drifted copy makes the
    fixture claim a crime it no longer commits.

    Symmetric on both sides of the comparison, which is the point: an anchor
    written "square-root" has to match an answer that writes "square root",
    and an anchor written "decision" has to match "decision-price". Splitting
    only one side, or keeping the compound whole, fails one of those.
    """
    words = re.findall(r"[a-z0-9]+", _norm(text).replace("-", " "))
    return {_stem(w) for w in words if w not in _STOPWORDS}


def overprecise_numbers(text: str) -> list[str]:
    """Figures *written* past :data:`config.PROSE_MAX_DECIMALS`, in order.

    A presentation axis, not a correctness one. The invented-number gate already
    decides whether a value belongs to the pack; this decides whether the desk
    would have spelled it that way. They are genuinely separate: a figure can be
    impeccably sourced and still unpublishable, and the first live run on
    qwen3.8-flash-next produced exactly that -- a Brinson-Carino attribution
    quoting a portfolio weight as ``0.472041725693``, every digit of it straight
    from ``allowed_numbers``, which the four existing axes passed without
    comment.

    Absolute rather than relative to the pack's own print, because the packs are
    where the raw floats come from: 1,522 published figures carry 9-12 decimals,
    so "no deeper than the pack printed it" licenses the very thing this is for.
    Fixing the packs is the root cure (they should round at computation time);
    this is the gate that stops the symptom shipping in the meantime, and it
    stays useful afterwards as the rule that says what prose may look like.

    Deduplicated, because a figure repeated three times is one thing to fix.
    """
    seen: list[str] = []
    for token in nums.TOKEN.findall(str(text)):
        if decimal_places(token) > config.PROSE_MAX_DECIMALS and token not in seen:
            seen.append(token)
    return seen


#: The tag a rounding-drift violation opens with, so the board can file it on
#: the invented-number axis without re-parsing English.
ROUNDING_DRIFT_TAG = "rounding_drift: "

#: An integer wearing a decimal tail. ``430567.0`` is the share count the first
#: live TCA rows all printed, straight out of a pack that stores counts as
#: floats. It is not a *wrong* number, which is why four numeric axes passed it
#: without comment; it is a number no desk would publish, and one that tells a
#: reader the count is known to a tenth of a share.
_INTEGER_WITH_ZERO_TAIL = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)\.0+(?![\d])")


def canonical_numbers(pack: dict) -> list[float]:
    """The values an *answer* may claim, as opposed to those a question prints.

    ``allowed_numbers`` is a union and has to be: the oracle serves pack figures
    as tool results, the exam composer derives distractors from them, the
    implementation suite pins them into generated tests. What it cannot do is
    tell 372.6 from the 373 the question rounds it to, because it holds both as
    peers. ``canonical`` holds one value per named quantity, and that is what
    this returns -- falling back to the union for a pack that declares none, so
    a fixture pack is graded exactly as loosely as it was before, never looser.
    """
    canonical = pack.get("canonical") or {}
    if not canonical:
        return [float(x) for x in pack["allowed_numbers"]]
    values: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            values.append(round(float(node), 12))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(canonical)
    return sorted(set(values))


def _round_trips_exactly(token: str, value: object) -> bool:
    """True when *token* is the same number as *value*, only spelled differently.

    "Differently" covers a dropped sign and the two unit conversions the number
    gate already blesses -- a fraction written as a percent, a percent written
    as a fraction. What it does not cover is a loss of precision, and that is
    the whole distinction the drift axis rests on:

    * ``1.21`` against a canonical ``0.0121`` is the *percent spelling* of the
      same figure. Nothing was rounded away, so a desk that writes "1.21% daily
      vol" has drifted nowhere and must not be failed for it.
    * ``373`` against a canonical ``372.6`` threw away a decimal the answer was
      supposed to carry. That is drift, and it is the amendment's own example.

    Exact, not the number gate's 0.5% tolerance -- 373 sits comfortably inside
    that tolerance of 372.6, which is precisely why the tolerance cannot be the
    instrument here.
    """
    try:
        got = abs(nums.val(str(token)))
        want = abs(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    # The scale factors of `invented_numbers`, applied exactly. Compared with a
    # relative epsilon rather than `==` because 1.21 * 0.01 is 0.0121000000...2
    # in binary, and a gate that turned on float representation would fire at
    # random.
    for scale in (1.0, 100.0, 0.01):
        scaled = got * scale
        if scaled == want or (want and abs(scaled - want) <= 1e-9 * abs(want)):
            return True
    return False


def rounding_drift(pack: dict, text: str) -> list[str]:
    """Quantities the answer spelled with the question's rounding, in pack order.

    The amendment's worked case: an attribution pack computes ``active_bps`` as
    372.6 and its question opens "behind the policy mix by 373 bp", because
    that is how a memo opens. Both numbers used to sit in ``allowed_numbers``,
    so an answer could quote either and every axis stayed green -- which is how
    a corpus teaches a model that the official figure is whichever one it saw
    last.

    Matched on the token as *written*, with word boundaries, because the mercy
    and the crime are both about what the reader sees. A quantity with no
    canonical value contributes nothing: an alias of nothing is just a number,
    and the invented-number gate already has an opinion about those.

    Deliberately narrow: only a *whole-number* alias of a fractional canonical
    is reported. See the two skips in the loop for why -- the axis is worth
    having exactly as far as it can be defended against correct prose, and no
    further.
    """
    canonical = pack.get("canonical") or {}
    aliases = pack.get("aliases") or {}
    if not canonical or not aliases:
        return []
    haystack = str(text or "")
    official_values = canonical_numbers(pack)
    hits: list[str] = []
    for quantity in sorted(aliases):
        if quantity not in canonical:
            continue
        official = canonical[quantity]
        for token in (
            aliases[quantity]
            if isinstance(aliases[quantity], (list, tuple))
            else [aliases[quantity]]
        ):
            token = str(token)
            if not token:
                continue
            if _round_trips_exactly(token, official):
                # A spelling, not a rounding -- see the helper. `abs(round(x))`
                # on a negative figure and `x * 100` on a fraction both land
                # here, and neither is a claim about a different number.
                continue
            if any(_round_trips_exactly(token, value) for value in official_values):
                # The token is exactly *some* official figure of this pack, so
                # whatever it is, it is not a rounding of this one. Attribution
                # makes the case concrete: the question rounds a 24.6 bp linked
                # active to "25 bp", and the same pack's selection total is
                # 25.0 -- reporting that selection total as the question's
                # rounding would refuse a correct answer for quoting a number
                # the pack computed.
                continue
            if decimal_places(token) > 0:
                # A *finer* rounding than a whole number is desk practice, not
                # drift, and this axis will not argue with it. A Brinson memo
                # that writes "the book returned 3.31%" of a 3.309% figure has
                # rounded the way a desk rounds; failing it would push every
                # attribution answer to three decimals, which is the precision
                # complaint `overprecise_numbers` exists to make in reverse.
                #
                # What is left is the case the amendment actually names: the
                # question dropped the fraction entirely -- "behind the policy
                # mix by 373 bp" of a 372.6 -- and the answer is reporting the
                # figure, not opening a memo. That is a difference a filter can
                # see. Everything subtler is a judgement the numeric gate
                # cannot make, and §D says so: it stays a gold-bar problem, and
                # a gate that pretended otherwise would be the third instrument
                # in this file to be rewritten after it failed correct prose.
                continue
            # The token must be the *whole* number, not its head: an answer
            # that correctly writes 62.1 contains "62", and reading that as
            # the 62 the question rounds to would fail every correct answer
            # whose canonical figure happens to start with its own alias.
            if re.search(rf"(?<![\d.\-]){re.escape(token)}(?![\d])(?!\.\d)", haystack):
                hits.append(
                    f"{ROUNDING_DRIFT_TAG}{quantity} is {official}, and the answer "
                    f"writes {token!r} -- that spelling belongs to the question, "
                    "not to the figure you are reporting"
                )
                break
    return hits


def integer_format_offenders(pack: dict, text: str) -> list[str]:
    """Whole numbers written with a ``.0`` tail, each with the desk's spelling.

    A format rule, run *before* the repair turn is composed, which is the whole
    reason it is worth having as its own axis: the repair can then say "write
    430,567 not 430567.0" instead of asking the teacher to intuit house style
    from a violation about decimals it did not commit.

    The advice has to clear :func:`rounding_drift` as well as this rule, and it
    did not. A live attribution memo wrote ``373.0`` for an ``active_bps`` of
    372.6; this axis said "write 373", the model obeyed, and ``rounding_drift``
    refused 373 on the next attempt as the question's spelling of the figure.
    Two gates, opposite instructions, three attempts, one dead letter. So where
    the stripped whole is a known *alias* of a fractional canonical, the
    spelling offered is the canonical figure -- the only one both axes accept.
    """
    # Compared without the sign: the token regex never captures a leading
    # minus, so a pack that spells its active return "-271.0" would otherwise
    # fail to excuse the "271.0" an answer writes beside the word "behind".
    spellings = {str(v).lstrip("-") for v in (pack.get("display") or {}).values()}
    display = {s.replace(",", ""): s for s in spellings}
    out: list[str] = []
    seen: set[str] = set()
    for match in _INTEGER_WITH_ZERO_TAIL.finditer(str(text or "")):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        if token.lstrip("-") in spellings:
            # The pack's own spelling is never an offence. An attribution pack
            # prints its active return to a tenth -- `display: {active_bps:
            # "91.0"}` -- and "a count is not known to a tenth" is an argument
            # about `430567.0`, not about a basis-point figure whose precision
            # *is* a tenth. Refusing it here while §A.3 tells the brief to use
            # it is two rules for one token.
            continue
        whole = match.group(1)
        if _is_zero(whole):
            # Zero is not a count wearing false precision; it is zero. A
            # Brinson memo writes "allocation 0.0, selection -0.3, interaction
            # +0.2" because the column is read down, and demanding a bare 0 in
            # the middle of it makes the decomposition ragged for no gain.
            #
            # Measured: a live attribution row spent all three attempts at an
            # identical 256 words, refusing to drop a `0.0` that is the pack's
            # own canonical value for that effect. The gate was asking it to
            # misreport the pack -- "a count is not known to a tenth" is an
            # argument about 430567.0, and it does not reach this.
            continue
        spelled = (
            _canonical_for_alias(pack, whole)
            or display.get(whole.replace(",", ""))
            or whole
        )
        out.append(f"{token!r} -- write {spelled}")
    return out


def _is_zero(whole: str) -> bool:
    """Is *whole* a spelling of zero? ``-0`` and ``0`` both count."""
    try:
        return float(whole.replace(",", "")) == 0.0
    except ValueError:
        return False


def _canonical_for_alias(pack: dict, whole: str) -> str | None:
    """The official figure *whole* is a drifting alias of, if it is one.

    ``None`` when the token is nobody's alias, or when it round-trips to its
    canonical exactly (then it is a spelling, and both axes are content). Keeps
    :func:`integer_format_offenders` from advising a token that
    :func:`rounding_drift` will refuse.
    """
    canonical = pack.get("canonical") or {}
    aliases = pack.get("aliases") or {}
    bare = whole.replace(",", "")
    for quantity, declared in aliases.items():
        if quantity not in canonical:
            continue
        tokens = declared if isinstance(declared, (list, tuple)) else [declared]
        if not any(str(t).replace(",", "") == bare for t in tokens):
            continue
        official = canonical[quantity]
        if _round_trips_exactly(whole, official):
            return None
        return str(official)
    return None


def _plain(value: float) -> str:
    """*value* as a desk would type it: fixed point, no exponent, no tail."""
    text = f"{float(value):.12f}".rstrip("0")
    return text[:-1] if text.endswith(".") else text


def display_form_offenders(pack: dict, text: str) -> list[str]:
    """Figures written raw where the pack publishes a percent spelling (§A.3).

    ``display`` and ``desk_figures`` both existed before the amendment and
    neither was ever *enforced*, so a live TCA row wrote participation as
    ``0.048555`` beside a pack that spells it ``4.86%`` -- a number no desk
    publishes, impeccably sourced, and passed by every numeric axis. This is
    the axis that refuses it.

    Percent spellings only, and the narrowness is the point. A display like
    ``430,567`` or ``296.61`` names the same number the answer would write
    anyway -- §A.3 permits the bare integer, and the ``.0`` tail is already
    :func:`integer_format_offenders`' business -- so a rule that demanded the
    separators would be failing correct prose to enforce a house style. A
    percent display is different in kind: it says the quantity is *measured*
    in percent, and the fraction is the raw storage form leaking through.
    """
    canonical = pack.get("canonical") or {}
    haystack = str(text or "")
    out: list[str] = []
    for quantity, spelled in sorted((pack.get("display") or {}).items()):
        spelling = str(spelled).strip()
        if quantity not in canonical:
            continue
        raw = _plain(canonical[quantity])
        if spelling.endswith("%"):
            pass  # a percent display says the quantity is measured in percent
        elif decimal_places(spelling.replace(",", "")) >= decimal_places(raw):
            # The display names the same number at the same depth (430,567 for
            # 430567, 296.61 for 296.61). §A.3 permits the bare form, and
            # demanding the separators would be failing correct prose to
            # enforce a house style.
            continue
        if re.search(rf"(?<![\d.\-]){re.escape(raw)}(?![\d])", haystack):
            out.append(f"{raw!r} -- write {spelling}")
    return out


#: The machinery a student row must never name. The system turn says "never
#: mention the fact pack, the contract, the gate, or these instructions" and
#: nothing enforced it, so three of eight rows in the first v3.2 capture wrote
#: "this pack holds no decision price" or "the pack's reconciling residual" --
#: sentences that teach a student the labelling protocol exists, which is the
#: whole failure the two-surface split was built to end.
#:
#: ``\bpacks?\b`` is the fingerprint the reviewer asked for. It has one real
#: false positive in this domain -- a Eurodollar *pack* is a strip of futures
#: -- and no work type in the plan trades one; if one is ever added, this is
#: the line to revisit rather than the row to excuse.
_CONTRACT_WORDS = re.compile(
    r"\bpacks?\b|\bfact[- ]pack\b|\bmust[_ ]mention\b|\ballowed[_ ]numbers\b"
    r"|\bforbidden[_ ]claims?\b|\bcanonical (?:figures?|values?)\b"
    r"|\bthe gate\b|\bthis contract\b|\bthe brief\b|\bword budget\b",
    re.IGNORECASE,
)


def contract_leaks(text: str) -> list[str]:
    """Phrases that name the factory, in order, deduplicated.

    A leak axis rather than a style one: the row is the student's whole view of
    the world, and a desk note that says "the pack does not contain a decision
    price" is telling the student there is a pack. The desk sentence is "there
    is no decision price here", and it carries the same fact.
    """
    seen: list[str] = []
    for match in _CONTRACT_WORDS.finditer(str(text or "")):
        token = match.group(0)
        if token.casefold() not in {s.casefold() for s in seen}:
            seen.append(token)
    return seen


def missing_mentions(pack: dict, text: str) -> list[str]:
    """``must_mention`` anchors the text failed to carry, in pack order.

    Every word of the anchor must appear in the answer, compared on stems and
    in any order -- not the anchor as one verbatim run of characters.

    That used to be a verbatim substring test, and it did not survive contact
    with a real teacher. Asked for "the square-root impact scaling", a correct
    answer wrote "Square-root impact scaling drives the cost" and failed on the
    missing definite article; asked for "participation near the ADV cap as the
    schedule constraint" it wrote "the schedule itself becomes the risk only
    when participation approaches the ADV cap" and failed on word order. Both
    answers made the point. The gate scored 0 of 3 on prose whose every
    keyword was present, and the repair loop then spent two more calls asking
    the model to fix writing that was already right -- so the instrument was
    selecting for verbatim quotation of the pack's own notes, which is the
    opposite of the desk voice the register asks for.

    Anchors are therefore short and distinctive (see the pack modules), and
    matching is on content: all terms present, any order, any inflection.
    """
    haystack = stems(text)
    return [
        point
        for point in pack.get("must_mention") or []
        if not stems(point) <= haystack
    ]


#: Negations that flip a forbidden claim into a warning against it. A desk
#: answer that says "do not assume the printable mid is achievable" is doing
#: exactly what the pack wants; scoring it as having asserted the claim -- which
#: a plain substring test does -- rejects the best answers for being explicit.
_NEGATIONS = (
    "not",
    "never",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "don't",
    "doesn't",
    "avoid",
    "reject",
    "refuse",
    "beware",
    "wrong",
    "false",
    "myth",
    "mistake",
    "fallacy",
    "trap",
    "assume",
)

#: How many words before the claim are inspected for a negation. Wide enough
#: for "do not make the mistake of assuming that X", narrow enough that an
#: unrelated "not" two sentences earlier cannot launder an assertion.
_NEGATION_WINDOW = 8


def forbidden_hits(pack: dict, text: str) -> list[str]:
    """``forbidden_claims`` the text asserted anyway, in pack order.

    A claim counts as asserted only where it appears *without* a negation in
    the words immediately before it. The contract in the teacher's system turn
    says not to assert a forbidden claim "even as a hedge or a disclaimer", and
    that is still enforced -- an unnegated mention is a hit wherever it occurs.
    What is no longer a hit is the explicit warning, which is the form a senior
    desk answer actually takes.
    """
    haystack = _norm(text)
    hits: list[str] = []
    for claim in pack.get("forbidden_claims") or []:
        needle = _norm(claim).rstrip(".")
        if not needle:
            continue
        for match in re.finditer(re.escape(needle), haystack):
            window = haystack[: match.start()].split()[-_NEGATION_WINDOW:]
            if not any(word.strip(",.;:") in _NEGATIONS for word in window):
                hits.append(claim)  # asserted plainly at least once
                break
    return hits


#: A closing bracket welded to the next word. Always a defect: no desk writes
#: "(see below)The cost is", and every instance found in live prose has been
#: the same one.
_WELDED_BRACKET = re.compile(r"[)\]]\w")

#: A figure with a word fused to its tail: ``$39.248Mchers``, which shipped
#: board-green in the first v3.2 capture and is why this rule exists. The same
#: decoder artefact also produced ``30.8 bpches`` and ``expected losshol``.
_FUSED_FIGURE = re.compile(r"\d(?:[.,]\d+)?([A-Za-z]{1,12})\b")

#: No space in the pattern above, deliberately: "295.77 against" is prose and
#: "39.248Mchers" is wreckage, and the space is the whole difference. A unit
#: may follow a figure. Everything else fused to one is an artefact.
#: Kept small and explicit: the cost of a missing entry is one repair turn on a
#: correct row, the cost of a permissive rule is a corpus that teaches the
#: student to weld words onto numbers.
_UNITS = frozenset(
    """bp bps b bn m mm mn k x pct ppt pp e usd eur gbp jpy d dd day days wk
    wks mo mos y yr yrs h hr hrs min mins sec secs st nd rd th s tn t q""".split()
)

#: A word printed twice with no space: "shareshare", from a live valuation
#: abstention's "83.0M shareshare count". The figure rule above cannot see it
#: (the fusion is word-to-word, not word-to-number) and it is unmistakable:
#: English has a handful of self-doubled words and none of them is desk
#: vocabulary.
_DOUBLED_WORD = re.compile(r"\b(\w{4,})\1\b", re.IGNORECASE)

#: A unit with a word fused to *it*: "bpches". The figure-fusion rule cannot
#: see this one, because the space falls between the number and the wreckage.
#: Only units that are not the opening of an English word may appear here, and
#: the list is short for that reason: ``adv``, ``var`` and ``es`` were in the
#: first draft and between them they refuse "advance", "variance", "various"
#: and -- caught on a live valuation memo, three attempts and a dead letter --
#: "estimate". A gate that fails correct prose to catch a decoder artefact has
#: made the corpus worse, so the artefact goes uncaught in those spellings and
#: the figure-fusion rule above stays the general instrument.
_FUSED_UNIT = re.compile(r"\b(bps?|vwap|twap)([a-z]{2,})\b", re.IGNORECASE)


def malformed_prose(text: str) -> list[str]:
    """Text that is broken rather than merely wrong, as sentences.

    Every other rule here judges an argument. This one judges whether the
    bytes are prose at all, and it exists because a row can be flawless on all
    sixteen axes and still read:

        ... a tail thinner than this book carries)Skip the headline: the
        mechanism is square-root scaling ...

    Four of twenty-four live rows carried that, and three of nine in an
    earlier sample -- a sentence closing on an unopened bracket and running
    into the next without a break. It predates the record-type rules, so it is
    not an instruction being narrated back; with ``think_present`` false on
    every one of those rows it looks like a reasoning fragment reaching the
    content stream. Whatever its cause, it ships today, and at corpus scale it
    teaches the student to write it.

    This is the one rule the brief does not state, and the exception is
    principled rather than convenient: "a gate the writer cannot see is a
    trap" is an argument about *arbitrary* contracts -- a ceiling, a band, a
    format -- that a correct writer could breach in good faith. No correct
    writer emits an unmatched bracket. The repair turn names it, which is the
    only notice this kind of fault needs.
    """
    out: list[str] = []
    text = str(text or "")
    for match in _FUSED_FIGURE.finditer(text):
        tail = match.group(1)
        if tail.casefold() in _UNITS or len(tail) < 3:
            continue
        out.append(
            f"a word is fused to a figure ({match.group(0).strip()!r}) -- write "
            "the number, a space, and the word"
        )
        break
    doubled = _DOUBLED_WORD.search(text)
    if doubled:
        out.append(
            f"a word is printed twice with no space ({doubled.group(0)!r}) -- "
            "write it once"
        )
    fused_unit = _FUSED_UNIT.search(text)
    if fused_unit:
        out.append(
            f"a word is fused to a unit ({fused_unit.group(0)!r}) -- the unit "
            "ends where the number's meaning ends"
        )
    welded = _WELDED_BRACKET.search(text)
    if welded:
        start = max(0, welded.start() - 40)
        out.append(
            "a bracket runs straight into the next word "
            f"({text[start : welded.end() + 20].strip()!r}) -- close the aside "
            "and start the sentence"
        )
    for opener, closer in (("(", ")"), ("[", "]")):
        if text.count(opener) != text.count(closer):
            out.append(
                f"unbalanced {opener}{closer}: {text.count(opener)} {opener!r} "
                f"against {text.count(closer)} {closer!r}"
            )
    return out


def gate_violations(pack: dict, text: str, kind: str) -> list[str]:
    """Every way *text* breaks the prose contract for ``kind``, as sentences.

    Empty list means shippable. The returned strings are written to be read
    twice -- once by the repair prompt (the model must be able to find its
    own error in them) and once by whoever post-mortems a dead letter.

    Order is deliberate. The two *format* rules -- the integer tail and the
    rounding drift -- are stated before the coverage and length ones, because
    the repair turn quotes this list in order and a teacher that reads "write
    430,567 not 430567.0" first fixes the spelling rather than rewriting the
    paragraph that happened to contain it.
    """
    violations: list[str] = []
    # First: is this prose at all? A malformed row wastes every judgement
    # below it, and the repair turn reads in order.
    violations.extend(malformed_prose(text))
    # Answers are graded against `canonical`, not against the union: the union
    # necessarily holds the question's own roundings, and a gate that admits
    # both spellings of one quantity is not measuring the thing it names.
    offenders = invented_numbers(text, canonical_numbers(pack), whitelist_for(pack))
    if offenders:
        violations.append(
            "invented numbers not in the fact pack: "
            + ", ".join(repr(t) for t in offenders)
        )
    badly_spelled = integer_format_offenders(pack, text)
    if badly_spelled:
        violations.append(
            "whole numbers written with a decimal tail: "
            + "; ".join(badly_spelled)
            + " -- a count is not known to a tenth"
        )
    raw_figures = display_form_offenders(pack, text)
    if raw_figures:
        violations.append(
            "figures written in their raw form where the desk has a spelling: "
            + "; ".join(raw_figures)
        )
    violations.extend(rounding_drift(pack, text))
    overprecise = overprecise_numbers(text)
    if overprecise:
        violations.append(
            f"figures written past {config.PROSE_MAX_DECIMALS} decimals: "
            + ", ".join(repr(t) for t in overprecise)
            + " -- round them the way a desk would print them"
        )
    leaks = contract_leaks(text)
    if leaks:
        violations.append(
            "the answer names the machinery behind it ("
            + ", ".join(repr(t) for t in leaks)
            + ") -- write the fact, not where the fact came from: "
            '"there is no decision price here", never "the pack has no '
            'decision price"'
        )
    missing = missing_mentions(pack, text)
    if missing:
        violations.append(
            "points the answer does not engage: "
            + "; ".join(repr(p) for p in missing)
            + " -- make each of these points in your own words, using every "
            "content word of it at least once (the check is mechanical: "
            "'participation against ADV' needs both 'participation' and "
            "'ADV' to appear)"
        )
    # The claims this pack's own arithmetic refutes (§D). Beside the forbidden
    # claims rather than inside them: a forbidden claim is a sentence the pack
    # declares wrong for every variant, and these are wrong *because of the
    # numbers in this one*.
    violations.extend(contradiction_violations(pack, text))
    hits = forbidden_hits(pack, text)
    if hits:
        violations.append(
            "forbidden claims asserted: " + "; ".join(repr(c) for c in hits)
        )
    if FINAL_ANSWER_TAG.casefold() in _norm(text):
        violations.append(
            f"{FINAL_ANSWER_TAG!r} is an exam contract, not prose style "
            "(spec §6 axis 5); drop it"
        )
    # The band is the kind's *in this register*: a desk note answers in 160
    # words where a committee memo may take 220, and grading both against the
    # wider number is how the register stopped being a constraint (§B.2).
    low, high = word_budget(kind, str(pack.get("register") or ""))
    words = len(str(text).split())
    if not low <= words <= high:
        violations.append(
            f"length {words} words is outside the {kind} budget {low}-{high}"
        )
    # The kind's own sentence ceiling, where it has one. `grounded` is a
    # citation and `abstention` is a refusal; both are short by construction in
    # a way no register makes them, and both briefs state the number.
    cap = KIND_SENTENCE_CAPS.get(kind)
    if cap:
        sentences = sentence_count(text)
        if sentences > cap:
            violations.append(
                f"a {kind} answer runs {sentences} sentences, over its "
                f"{cap}-sentence ceiling -- merge or cut "
                f"{sentences - cap} of them"
            )
    # The §C axis. Last, because it is the only one that judges *shape* rather
    # than content, and a row that is still inventing numbers has a worse
    # problem than its section headings.
    violations.extend(register_violations(pack.get("register") or "", text, kind=kind))
    return violations
