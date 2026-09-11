"""The preference contract, verifiable (analysis spec §5.7; arch spec §5.6, §6 axis 11/12).

A DPO pair is three claims, not two texts: the chosen side is a *second
telling of the truth* (paraphrase, same pack, same figures, gate-clean), the
rejected side is *fluent prose of a named wrong model*, and the pair is
disjoint from the SFT corpus it was born from. The v1 disease the whole
design treats was the pair that was not a pair -- ``chosen`` byte-equal to
the SFT target, ``rejected`` the chosen with one number changed; DPO saw a
duplicate and the margin saturated at loss 0.0 (analysis spec §2.4). This
module is the pair's physics, executable:

* **per-kind probability** (§5.7): whether a shipped row is paired at all is
  a pure function of its coordinates -- ``pair_draw`` draws from the pair's
  own seed, so the pairing set is recomputible without the teacher;
* **the pitfall detectors** (§5.7 step 3b "must violate the pitfall"): each
  licensed pitfall carries a detector that reads the pack's own contract --
  ``must_mention`` for an ignored constraint, ``as_of`` for a look-ahead, the
  number table for a false precision or a figure from memory -- and answers
  whether the rejected side actually committed the crime it names. A
  rejected side that does not commit is a second chosen side, and a second
  chosen side is not training signal;
* **the disjointness gates** (acceptance criterion 5): ids disjoint by
  prefix, ``source_sft_id`` recorded and never equal, the chosen side not
  byte-identical to the SFT target and its 8-word shingle overlap with the
  target and with the rejected side below the set threshold -- v1's
  "chosen with a different last number" scores near 1.0 on that index and
  dies here.

``invented_number`` is deliberately *not* licensable: §5.7 caps the easy
negative control at a tenth of pairs and the taxonomy never lists it; a
work type that ever asked for it would be refused by the licence check,
which is the cap enforced by non-existence.
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
from ..seed import render_seed, rng_for  # noqa: E402
from .exam import LITURGY_MARKERS  # noqa: E402
from .invented_numbers import invented_numbers  # noqa: E402
from ..teacher.prompts import WORD_BUDGETS  # noqa: E402
from .prose import gate_violations, missing_mentions, whitelist_for  # noqa: E402

#: The record type the preference stage publishes. Not in ``BRIEF_KINDS``:
#: a pair is not a brief, and a row that could be rendered by the prose
#: stage could not be audited by this contract.
PREFER_KIND = "preference"

#: Violation tags, same idiom as the other lanes: the board routes each
#: prefix to the axis that owns it. ``TAG_COPY`` carries the v1 pathology --
#: the near-copy that made DPO's loss a constant -- and belongs to axis 12
#: whatever its cause, because a pair that is not two texts is not a pair.
TAG_SHAPE = "preference shape: "
TAG_TAMPER = "preference recomposition: "
TAG_PITFALL = "preference pitfall: "
TAG_COPY = "preference copy: "

#: What every pair row must carry before any text is compared. ``prompt`` is
#: the two-turn context both sides answer into (the conversation the pair
#: trains); ``source_sft_id`` is recorded, never equal to ``id`` (§5.7).
_PAIR_REQUIRED = (
    "id",
    "record_type",
    "source_sft_id",
    "work_type",
    "scenario_id",
    "variant",
    "parent_kind",
    "pitfall",
    "question",
    "prompt",
    "chosen",
    "rejected",
    "register",
    "verification",
)

#: The named wrong models (spec §5.7: "use a named wrong model -- ordinary
#: vs modified [duration]; additive duration across FX; Gaussian VaR on
#: options"). Each entry's phrases are the detector's evidence: the rejected
#: side must print one, casefolded and collapsed, or its crime is unproven
#: and the pair dies. A teacher that means the wrong model but words it
#: outside the list fails the detector -- fail loud is the correct direction
#: when the alternative is accepting a pair on vibes.
WRONG_MODEL_SIGNS: dict[str, tuple[str, ...]] = {
    "valuation.equity.dcf": ("without reinvestment", "reinvestment-free"),
    "valuation.equity.multiples": ("mean of the peers", "average peer multiple"),
    "risk.market.var_es": (
        "under the gaussian veil",
        "gaussian tails",
        "the book is gaussian",
    ),
    "portfolio.attribution.brinson_carino": (
        "unbridged brinson",
        "durations add across the curve",
    ),
    "execution.tca.arrival": ("linear with the order", "cost grows linearly"),
}

#: The brief the rejected author receives, per pitfall. The crime is named
#: twice on purpose: once as the model to adopt, once as the phrasing the
#: detector will look for. The figure discipline stays the pack's -- the
#: rejected side is *fluent and wrong*, not *wrong and sloppy*, and the
#: numbers it quotes are the pack's own except where the pitfall itself is
#: a number-crime (false precision, tool_skip), whose numbers the computer
#: can label (§5.7 step 3d).
PITFALL_INSTRUCTIONS: dict[str, str] = {
    "wrong_assumption": (
        "State your reading as if {concept}. Name the model you adopted in "
        "plain words so a reader can see it, and print the figures that "
        "model gives; keep every other figure the pack prints."
    ),
    "false_precision": (
        "Quote one figure of the pack to eight decimal places -- more "
        "precision than the pack's own print ever carried, and more than "
        "the pack supports. Keep every other figure as the pack prints it."
    ),
    "ignored_constraint": (
        "Write the desk's read of the pack, but leave one point of the "
        "pack's stated contract wholly unstated. The constraint you "
        "silence is part of the lesson; everything else keeps the pack's "
        "figures."
    ),
    "look_ahead": (
        "Anchor the read on a print dated after the pack's own as-of date "
        "and say so -- name the later date in full (YYYY-MM-DD). Everything "
        "else keeps the pack's figures."
    ),
    "overconfident_abstention_fail": (
        "The pack is incomplete and you will answer anyway: commit, out "
        "loud, to the figure the computer would have printed were the pack "
        "complete. Name no caution, state no missing item."
    ),
    "wrong_register": (
        "Close with the exam liturgy on a brief that is not an exam: an "
        "ASSUMPTIONS block, numbered steps beginning 'Step 1.', and a "
        "'FINAL ANSWER:' line. Keep the body's figures the pack's own."
    ),
    "tool_skip": (
        "Produce the desk's figures from memory rather than from the pack "
        "or the tools: quote at least one number the pack never printed. "
        "Everything else keeps the pack's figures."
    ),
}

#: The concept each wrong-assumption brief substitutes, per work type. The
#: wording must contain one of that row's detector phrases; the pair is
#: dead-lettered, not judged, when a side cannot manage both.
_WRONG_MODEL_CONCEPTS: dict[str, str] = {
    "valuation.equity.dcf": "a reinvestment-free perpetuity, the strip running on its own",
    "valuation.equity.multiples": "the arithmetic mean of the peers, not the median",
    "risk.market.var_es": "a Gaussian book where the tails are not Gaussian",
    "portfolio.attribution.brinson_carino": "unbridged Brinson arithmetic, no Carino bridge",
    "execution.tca.arrival": "cost linear with the order, not the square root",
}

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for token in nums.TOKEN.findall(str(text)):
        try:
            values.append(nums.val(token))
        except ValueError:
            continue
    return values


def _agrees(value: float, table, tolerance: float) -> bool:
    return any(
        abs(value - float(allowed)) <= tolerance * max(1.0, abs(float(allowed)))
        for allowed in table
    )


def _printed_depth(value: float) -> int:
    """The decimal places the *brief* spells *value* to.

    Measured on the figure as the model is shown it, not as the pack stores it:
    ``teacher.prompts.desk_figures`` rounds the brief to
    ``config.PROSE_MAX_DECIMALS`` before the teacher reads a byte, and that is
    the only spelling a rejected side can be over-quoting. The stored pack is
    deeper -- ``assemble_numbers`` cleans to 12 dp to kill binary noise, and
    1,522 figures on disk sit at 9-12 places. Measuring against *those* made
    this detector unsatisfiable for them: a rejected side would have had to write
    thirteen decimals to out-print the pack, while the brief asks for eight.

    Every figure travels the fixed-point, zero-trimmed spell the prose lane
    prints in (trailing zeros and a bare point stripped): ``26.66 -> "26.66"``
    is two places, ``3.0 -> "3"`` none, ``0.048555 -> "0.048555"`` six. This count is one half of the
    false-precision detector's test: a rejected side that writes a pack figure
    with *more* decimals than the pack printed it has quoted past the print --
    the crime -- while one that writes it at or under that depth has merely
    quoted the figure the pack owns. Depth alone is not the whole test, for
    the desk spells an integer figure with a conventional trailing zero (a
    three written ``3.0``, one place past the pack's print of ``3``) and that
    is not a claim of precision; the detector therefore asks for that
    conventional overshoot to reach the brief's depth (:data:
    ``_FALSE_PRECISION_PLACES``) before it calls, which no pack figure's own
    print ever does.
    """
    value = round(float(value), config.PROSE_MAX_DECIMALS)
    text = f"{value:.10f}".rstrip("0")
    if text.endswith("."):
        text = text[:-1]
    return len(text.split(".", 1)[1]) if "." in text else 0


#: The depth the false-precision brief commands -- "quote one figure of the
#: pack to eight decimal places" -- and the depth the fixture's over-quoting
#: rejected side actually writes (``:.8f``). No figure the brief shows reaches
#: it, and that is enforced now rather than assumed: ``desk_figures`` caps the
#: brief at ``config.PROSE_MAX_DECIMALS`` (six, a participation). The
#: detector calls the crime only where a token reaches this depth *and* runs
#: deeper than the pack's print of that figure, so neither the pack's honest
#: six-place figures nor a conventional trailing zero are mistaken for it.
_FALSE_PRECISION_PLACES = 8


def shingle_overlap(left: str, right: str) -> float:
    """The Jaccard index of the two texts' word-shingle sets (§criterion 5).

    Eight-word shingles (config, ``PREF_SHINGLE_WORDS``): long enough that
    a paraphrase of honest effort survives only where the sentences really
    were retaken, short enough that v1's "same paragraph, different last
    number" -- a near copy at any shorter grain -- still scores near 1.0
    and is caught. Texts shorter than one shingle share nothing measurable.
    """
    from .. import config  # late: one read of the two constants, both places

    n = config.PREF_SHINGLE_WORDS

    def grams(text: str) -> set:
        words = str(text).casefold().split()
        return {
            " ".join(words[i : i + n]) for i in range(0, max(0, len(words) - n + 1))
        }

    left_set, right_set = grams(left), grams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def pitfall_violations(pack: dict, text: str, pitfall: str) -> list[str]:
    """Did the rejected side commit the crime it names? Empty list: it did.

    Each detector reads the pack's own contract, never a template: the
    ignored constraint is a point of ``must_mention`` the text leaves
    uncovered, the look-ahead is a full date beyond ``as_of``, the false
    precision is a pack figure quoted past the digit the computer printed,
    the figure from memory is a token no allowed number explains, the
    overconfident answer is a computed figure committed to by a text that
    names no missing item, the wrong register is the exam's closing on a
    prose body, and the wrong model must be *named* -- one of
    :data:`WRONG_MODEL_SIGNS` present in the prose. An unlicensed or
    unknown pitfall name is a defect of the caller, and is raised, not
    shrugged.
    """
    low = _norm(text)
    if pitfall == "wrong_assumption":
        signs = WRONG_MODEL_SIGNS.get(pack["work_type"], ())
        if signs and not any(sign in low for sign in signs):
            return [
                f"{TAG_PITFALL}the named wrong model is absent: the rejected "
                f"side says none of {list(signs)!r}"
            ]
        return []
    if pitfall == "false_precision":
        for token in nums.TOKEN.findall(str(text)):
            if "." not in token:
                continue
            fraction = token.split(".", 1)[1]
            if len(fraction) < _FALSE_PRECISION_PLACES:
                continue
            value = nums.val(token)
            for allowed in pack["allowed_numbers"]:
                deeper = len(fraction) > _printed_depth(allowed)
                if deeper and _agrees(value, (allowed,), 1e-6):
                    return []
        return [
            f"{TAG_PITFALL}no pack figure was quoted to {_FALSE_PRECISION_PLACES} "
            "decimals past its print: no token is both that deep and deeper "
            "than the pack's own spelling of the figure"
        ]
    if pitfall == "ignored_constraint":
        if not missing_mentions(pack, text):
            return [
                f"{TAG_PITFALL}nothing was ignored: the rejected side still "
                "covers every point of the pack's stated contract"
            ]
        return []
    if pitfall == "look_ahead":
        as_of = str(pack.get("as_of") or "")
        if not any(
            date > as_of for date in _DATE.findall(str(text)) if len(date) == 10
        ):
            return [
                f"{TAG_PITFALL}no look-ahead was taken: no full date beyond "
                f"the pack's as-of {as_of!r} appears"
            ]
        return []
    if pitfall == "overconfident_abstention_fail":
        computed = (pack.get("computed") or {}).values()
        if not any(_agrees(v, computed, 1e-9) for v in _numbers(text)):
            return [
                f"{TAG_PITFALL}the pack was left unanswered after all: the "
                "rejected side commits to no figure of the computer"
            ]
        return []
    if pitfall == "wrong_register":
        closing = FINAL_ANSWER_TAG.casefold() in low or any(
            marker.casefold() in low for marker in LITURGY_MARKERS
        )
        if not closing:
            return [
                f"{TAG_PITFALL}no exam closing on the prose body: neither "
                f"{FINAL_ANSWER_TAG!r} nor {list(LITURGY_MARKERS)!r} appears"
            ]
        return []
    if pitfall == "tool_skip":
        if not invented_numbers(text, pack["allowed_numbers"], whitelist_for(pack)):
            return [
                f"{TAG_PITFALL}nothing was free-handed: every figure the "
                "rejected side prints is the pack's own"
            ]
        return []
    raise ValueError(
        f"no detector for pitfall {pitfall!r}; the licensed names are "
        + ", ".join(sorted(PITFALL_INSTRUCTIONS))
        + " (config's ladders and the taxonomy's licences share that list)"
    )


def _pair_seed(work_type: str, family: str, kind: str, variant: int) -> int:
    """The pair's own seed domain, salted with the parent kind.

    ``analysis`` and ``memo`` rows of one coordinate are two pairs, and the
    ids must say so: unsalted they would draw the same seed, collide on the
    shard, and the second would silently never be written.
    """
    return render_seed(work_type, family, f"{PREFER_KIND}:{kind}", variant)


def pair_id(work_type: str, family: str, kind: str, variant: int) -> str:
    """The id one (work_type, family, parent kind, variant) pair is *always* called.

    The pair draws from its own record-type seed, disjoint from both sides'
    supervised seeds: the id is a coordinate, and the coordinate names the
    pair whether or not the teacher was ever asked (the draw is the same
    function -- a replay can recompute which pairs the plan entails).
    """
    from .. import config  # late, with the other control-plane reads

    return config.preference_id(_pair_seed(work_type, family, kind, variant))


def pair_draw(
    work_type: str, family: str, kind: str, variant: int, licensed: list[str]
) -> tuple:
    """``(paired, pitfall)`` for one shipped row -- a pure function of coordinates.

    §5.7's probability table read as a dice roll of the pair's own seed, and
    the crime drawn from the *sorted* licence list of the work type, so the
    pairing set and the crime per pair are recomputible from the plan
    without asking the teacher a second time. ``licensed`` arrives sorted
    and non-empty; an empty licence is not a draw but a planning error, and
    it is raised.
    """
    from .. import config  # late, with the other control-plane reads

    probability = config.PREF_PROBABILITIES.get(kind)
    if probability is None:
        raise ValueError(f"no pairing probability for parent kind {kind!r}")
    if not licensed:
        raise ValueError(f"work type {work_type!r} licenses no pitfall to draw")
    rng = rng_for(_pair_seed(work_type, family, kind, variant))
    if rng.random() >= probability:
        return False, None
    return True, rng.choice(list(licensed))


CHOSEN_ASK = (
    "Restate the desk's answer above in your own words, for a preference "
    "pair: the same figures, the same points, a different telling. Change "
    "the wording and the order of the sentences; keep every number exactly "
    "as the pack prints it. Output the passage alone."
)

_REJECTED_FRAMING = (
    "Write the desk's answer to the brief above, in fluent prose, under one "
    "deliberate defect -- the pair teaches by contrast, so the defect must "
    "be real and it must be this one: "
)


def chosen_brief(prompt: list[dict], answer: str, kind: str = "") -> list[dict]:
    """The second-telling exchange: the parent context, the parent answer,
    and the ask. The assistant turn is the target the model must *not* copy.

    ``kind`` names the parent lane, and naming it closes a gap that only
    became visible once the prompt stopped being the parent's whole brief.
    The chosen side is graded against ``WORD_BUDGETS[kind]`` -- an analysis
    paraphrase under 120 words is dead-lettered -- and the ask never said so,
    so the model was being marked against a band it had not been shown. It
    also makes the request distinguishable again: with the prompt reduced to
    the pack's question, two lanes whose parent answers happen to coincide
    would otherwise send byte-identical bodies and be owed two different
    word bands by the same reply.
    """
    ask = CHOSEN_ASK
    if kind in WORD_BUDGETS:
        low, high = WORD_BUDGETS[kind]
        ask += f" Write between {low} and {high} words, as a {kind} answer does."
    return [
        *prompt,
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ask},
    ]


def rejected_brief(prompt: list[dict], work_type: str, pitfall: str) -> list[dict]:
    """The contrastive exchange: the parent context and the named crime.

    ``think`` is forced off the rejected call at the routing table, not
    here: a thinking trace narrating a deliberate error teaches the wrong
    lesson twice (arch spec §5.4).
    """
    if pitfall not in PITFALL_INSTRUCTIONS:
        raise ValueError(f"no brief for pitfall {pitfall!r}")
    instruction = PITFALL_INSTRUCTIONS[pitfall]
    if pitfall == "wrong_assumption":
        concept = _WRONG_MODEL_CONCEPTS.get(work_type)
        if concept is None:
            raise ValueError(f"no wrong model on file for {work_type!r}")
        instruction = instruction.format(concept=concept)
    return [
        *prompt,
        {
            "role": "user",
            "content": _REJECTED_FRAMING + instruction,
        },
    ]


def pair_gate_violations(pack: dict, parent_row, row: dict) -> list[str]:
    """Every way a stored pair fails its contract, tagged for the board.

    ``parent_row`` is the SFT row ``source_sft_id`` names (``None`` when the
    board could not find it -- itself a finding). The chosen side is graded
    as the full prose it claims to be; the rejected side is graded as
    rhetoric and for one thing: that it committed the crime it names. The
    copy checks are the acceptance criterion, read in the literal: the
    chosen side may not be the SFT target, byte for byte, nor a near copy
    of it, nor a near copy of the rejected side it is supposed to contrast.
    """
    from .. import config  # late, with the other control-plane reads

    violations: list[str] = []
    missing = [field for field in _PAIR_REQUIRED if field not in row]
    if missing:
        return [f"{TAG_SHAPE}missing fields: {', '.join(missing)}"]
    kind = row["parent_kind"]
    if not str(row["id"]).startswith(f"{config.PREFERENCE_ID_PREFIX}_"):
        violations.append(
            f"{TAG_COPY}id {row['id']!r} is outside the preference namespace "
            f"{config.PREFERENCE_ID_PREFIX!r} -- a pair that shares the SFT "
            "id space is the v1 disease recidivist"
        )
    if not str(row["source_sft_id"]).startswith(f"{config.SUPERVISED_ID_PREFIX}_"):
        violations.append(
            f"{TAG_SHAPE}source_sft_id {row['source_sft_id']!r} is not a supervised id"
        )
    if row["source_sft_id"] == row["id"]:
        violations.append(
            f"{TAG_COPY}source_sft_id equals the pair id: the pair is "
            "masquerading as its own SFT row (v1 nested the pair; v2 made the "
            "ids disjoint; both mistakes are this shape)"
        )
    if kind not in config.PREF_PROBABILITIES:
        violations.append(
            f"{TAG_SHAPE}parent_kind {kind!r} carries no pairing probability"
        )
    if row["pitfall"] not in PITFALL_INSTRUCTIONS:
        violations.append(f"{TAG_SHAPE}pitfall {row['pitfall']!r} is not known")
    if row["record_type"] != PREFER_KIND:
        violations.append(
            f"{TAG_SHAPE}record_type {row['record_type']!r} does not claim "
            f"{PREFER_KIND!r}"
        )
    if row["register"] not in config.VALID_REGISTERS:
        violations.append(f"{TAG_SHAPE}register {row['register']!r} is not a register")
    prompt = row["prompt"]
    # The student's question, one user turn, and nothing else. It used to be
    # the parent row's ``messages[:2]`` -- the factory's system turn and its
    # JSON contract -- so the preference config carried the labelling protocol
    # into DPO exactly as faithfully as the SFT config carried it into
    # training. §A closes both doors, and this is the second one.
    if not isinstance(prompt, list) or [m.get("role") for m in prompt] != ["user"]:
        violations.append(
            f"{TAG_SHAPE}prompt must be the single user turn the pair trains "
            "on; roles read "
            + repr([m.get("role") for m in prompt or [] if isinstance(m, dict)])
        )
    elif str((prompt[0] or {}).get("content") or "") != str(pack.get("question") or ""):
        violations.append(
            f"{TAG_SHAPE}prompt is not the pack's question -- a pair is trained "
            "against the question the desk asked, not a paraphrase of it"
        )
    chosen, rejected = row["chosen"], row["rejected"]
    if not isinstance(chosen, str) or not chosen.strip():
        violations.append(f"{TAG_SHAPE}chosen side is empty")
        return violations
    if not isinstance(rejected, str) or not rejected.strip():
        violations.append(f"{TAG_SHAPE}rejected side is empty")
        return violations

    # The chosen side: the full prose gate. The number/coverage/forbidden
    # sentences arrive worded as the prose board already files them (it maps
    # them to 3/4/4/5 by those clauses); everything the gate says besides is
    # structure of a side, and belongs to the shape tag.
    if kind in config.PREF_PROBABILITIES:
        for problem in gate_violations(pack, chosen, kind):
            text = f"chosen side: {problem}"
            if "budget" in problem:
                violations.append(f"{TAG_SHAPE}{text}")
            else:
                violations.append(text)
    if parent_row is None:
        violations.append(
            f"{TAG_TAMPER}no SFT row answers to source_sft_id "
            f"{row['source_sft_id']!r} -- the pair outlived its own parent"
        )
        return violations
    answer = str((parent_row or {}).get("answer") or "")
    if chosen == answer:
        violations.append(
            f"{TAG_COPY}chosen side is the SFT target byte for byte (v1's "
            "loss-0.0 pathology; the paraphrase did not paraphrase)"
        )
    overlap_target = shingle_overlap(chosen, answer)
    if overlap_target > config.PREF_MAX_SHINGLE_OVERLAP:
        violations.append(
            f"{TAG_COPY}chosen side is a near copy of the SFT target: shingle "
            f"overlap {overlap_target:.2f} exceeds {config.PREF_MAX_SHINGLE_OVERLAP} "
            "(a different last number is not a second telling)"
        )
    overlap_pair = shingle_overlap(chosen, rejected)
    if overlap_pair > config.PREF_MAX_SHINGLE_OVERLAP:
        violations.append(
            f"{TAG_COPY}the two sides are a near copy of each other: shingle "
            f"overlap {overlap_pair:.2f} exceeds {config.PREF_MAX_SHINGLE_OVERLAP} "
            "(DPO reads a margin; this pair has none)"
        )
    words = len(rejected.split())
    if not config.PREF_MIN_REJECTED_WORDS <= words <= config.PREF_MAX_REJECTED_WORDS:
        violations.append(
            f"{TAG_PITFALL}rejected side is not fluent prose of the desk: "
            f"{words} words, outside the band "
            f"{config.PREF_MIN_REJECTED_WORDS}-{config.PREF_MAX_REJECTED_WORDS}"
        )
    if row["pitfall"] in PITFALL_INSTRUCTIONS:
        violations += pitfall_violations(pack, rejected, row["pitfall"])
    return violations


__all__ = [
    "PREFER_KIND",
    "PITFALL_INSTRUCTIONS",
    "TAG_COPY",
    "TAG_PITFALL",
    "TAG_SHAPE",
    "TAG_TAMPER",
    "WRONG_MODEL_SIGNS",
    "chosen_brief",
    "pair_draw",
    "pair_gate_violations",
    "pair_id",
    "pitfall_violations",
    "rejected_brief",
    "shingle_overlap",
]
