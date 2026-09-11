"""The prose gate in isolation: every way a completion can be wrong, tested.

A *synthetic* pack drives these cases, not a real one: a unit test of the
gate must control every field the gate reads (``allowed_numbers``,
``must_mention``, ``forbidden_claims``, ``as_of``) -- real packs would make
the assertions hostage of the computers' authoring. Real-pack compliance is
the committed fixture's job and the PR2 50-row gate test carries it.
"""

from __future__ import annotations

import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config  # noqa: E402
from pipelines.v3.verification.prose import (  # noqa: E402
    ROUNDING_DRIFT_TAG,
    canonical_numbers,
    forbidden_hits,
    gate_violations,
    integer_format_offenders,
    missing_mentions,
    overprecise_numbers,
    rounding_drift,
    stems,
    whitelist_for,
)
from pipelines.v3.verification.register import (  # noqa: E402
    DESK_CHAT_MAX_SENTENCES,
    DESK_CHAT_WORDS_PER_SENTENCE,
    REGISTER_MIN_SEPARATION,
    desk_chat_ceiling,
    profile_distance,
    register_profile,
)
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402

PACK = {
    "allowed_numbers": [0.03, 2500.0, 1.7],
    "as_of": "2026-03-31",
    "must_mention": ["The dcf value sits below price.", "Capex is rising."],
    "forbidden_claims": ["The forecast is certain."],
    "question": "What do the figures say?",
    "register": "desk_chat",
}


def _clean_text() -> str:
    """Gate-clean prose for PACK within the abstention budget (50-140 words);
    the length test below pins that, so editing this cannot silently drift."""
    return (
        "The dcf value sits below price. Capex is rising. "
        # `2500`, not `2500.0`: §D's format rule says a whole number does not
        # carry a decimal tail, and this helper has to stay gate-clean.
        "Measured against 2500 revenue the ratio 1.7 holds, and the "
        "sensitivity is 0.03. As of 2026-03-31, split 100 to 2 over 252 "
        "sessions. The bridge opens at 2500 and closes at 2500, the "
        "coverage ratio reads 1.7 twice over, and the drift stays near 0.03 "
        "for every desk that checks the file against the pack it was drawn from."
    )


def test_clean_text_itself_sits_in_the_abstention_band():
    low, high = 50, 140  # the abstention budget; if this fails the helpers drift
    words = len(_clean_text().split())
    assert low <= words <= high, words


def test_clean_produce_no_violations():
    assert gate_violations(PACK, _clean_text(), "abstention") == []


def test_invented_number_is_named_not_numbered():
    violations = gate_violations(
        PACK, _clean_text() + " The terminal multiple is 8153.7729.", "abstention"
    )
    assert len(violations) == 1
    assert "invented numbers" in violations[0]
    assert "8153.7729" in violations[0], "the repair prompt must name the token"


def test_missing_mention_is_reported_point_by_point():
    missing = missing_mentions(PACK, "Capex is rising.")
    assert missing == ["The dcf value sits below price."]
    violations = gate_violations(PACK, "Capex is rising. " * 40, "abstention")
    assert any("must_mention points not covered" in v for v in violations)


def test_mention_matching_ignores_case_and_runs_of_spaces():
    text = "capex   IS  rising. and the   DCF value SITS below PRICE."
    assert missing_mentions(PACK, text) == []


def test_a_nominalised_anchor_matches_the_verb_the_desk_actually_writes():
    """ "normality assumption" must match "normality ... it assumes ...".

    Live evidence: a correct VaR answer wrote "The square-root scaling is the
    first thing normality hides: it assumes daily returns are independent and
    identically distributed", and the gate reported the "normality assumption"
    point as not covered -- "assumption" stemmed to "assumpt", "assumes" to
    "assum". Nobody writes "the normality assumption assumption"; the desk
    nominalises in the anchor and conjugates in the prose, and an instrument
    that cannot see through that is selecting for quotation again.
    """
    pairs = [
        ("normality assumption", "normality hides: it assumes iid returns"),
        ("assumption", "the answer assumed a flat curve"),
        ("assumptions", "what it assumes"),
        ("consumption", "the model consumes the whole budget"),
    ]
    for anchor_text, prose in pairs:
        assert stems(anchor_text) <= stems(prose), f"{anchor_text!r} vs {prose!r}"


def test_stemming_does_not_collapse_words_that_mean_different_things():
    """The mercy above must not become a matcher that agrees with anything."""
    for anchor_text, prose in [
        ("normality assumption", "the option is priced off a normal curve"),
        ("liquidity", "the position is liquidated"),
        ("participation", "the participants disagreed"),
    ]:
        assert not stems(anchor_text) <= stems(prose), f"{anchor_text!r} vs {prose!r}"


def test_a_figure_spelled_past_desk_precision_is_a_violation():
    """Sourced and unpublishable are different verdicts.

    The first live run on qwen3.8-flash-next wrote a Brinson-Carino attribution
    quoting a portfolio weight as ``0.472041725693``. Every digit came from
    ``allowed_numbers``, so the invented-number axis passed it -- correctly. It is
    still not a figure a desk would print, and a reader handed twelve decimals is
    being told the book is known to a picometre.
    """
    assert overprecise_numbers("the weight is 0.472041725693 of the book") == [
        "0.472041725693"
    ]
    # Deduplicated: a figure repeated is one thing to fix.
    once = overprecise_numbers("0.13874892827 versus 0.13874892827 again")
    assert once == ["0.13874892827"]


def test_precision_that_the_desk_actually_needs_is_left_alone():
    """Six places is a participation rate, not sloppiness -- the ceiling has to
    sit above the deepest figure the corpus legitimately carries."""
    assert overprecise_numbers("participation is 0.017381 of ADV") == []
    assert overprecise_numbers("impact 26.66 bp on 8,867,535 shares at 296.61") == []
    assert config.PROSE_MAX_DECIMALS == 6


def test_the_precision_axis_reaches_the_gate():
    """Wired into gate_violations, not merely available beside it."""
    pack = dict(PACK)
    pack["allowed_numbers"] = list(PACK["allowed_numbers"]) + [0.472041725693]
    violations = gate_violations(pack, "the weight is 0.472041725693 here", "analysis")
    assert any("past 6 decimals" in v for v in violations), violations


def test_forbidden_claim_is_caught_even_buried():
    text = "Between the lines, obviously, the forecast is certain. for anyone reading."
    assert forbidden_hits(PACK, text) == ["The forecast is certain."]
    violations = gate_violations(PACK, text, "abstention")
    assert any("forbidden claims asserted" in v for v in violations)


def test_final_answer_tag_is_a_prose_violation():
    violations = gate_violations(
        PACK, _clean_text() + " FINAL ANSWER: sell.", "abstention"
    )
    assert any("FINAL ANSWER:" in v for v in violations)


def test_word_budget_is_enforced_both_ways():
    short = gate_violations(PACK, _clean_text()[:80], "abstention")
    assert any("outside the abstention budget" in v for v in short)
    long = gate_violations(PACK, _clean_text() + " plain prose " * 90, "abstention")
    assert any("outside the abstention budget" in v for v in long)


def test_whitelist_covers_structural_tokens_and_the_as_of_date():
    wl = whitelist_for(PACK)
    assert {"100", "2", "252"} <= wl
    # TOKEN keeps the sign that precedes a digit: the date's own tokens are
    # "-03"/"-31", and a written "2026-03-31," must survive its trailing comma
    assert {"2026", "-03", "-31"} <= wl, "the pack's own as-of date may be cited"
    assert invented_numbers("as of 2026-03-31", [0.03], wl) == []
    assert invented_numbers("as of 2027-03-31", [0.03], wl) == ["2027"], (
        "a date the pack never stated is an invented number, whitelisted or not"
    )


def test_harness_compliant_text_passes_for_every_kind():
    """The dummy that writes the fixture obeys the same law the teacher faces."""
    for kind, (low, high) in _kinds():
        text = prose_harness.compliant_text(_wide_pack(), kind)
        assert low <= len(text.split()) <= high, kind
        assert gate_violations(_wide_pack(), text, kind) == [], kind


def _kinds():
    from pipelines.v3.teacher.prompts import BRIEF_KINDS, WORD_BUDGETS

    return [(kind, WORD_BUDGETS[kind]) for kind in BRIEF_KINDS]


def _wide_pack() -> dict:
    """A pack rich enough that every word budget is reachable -- like the
    plan's real ones, which is the situation the harness must survive."""
    return {
        "allowed_numbers": [float(i) for i in range(1, 40)] + [0.031, 0.5],
        "as_of": "2025-12-31",
        # The harness's opener names the work type and the as-of instead of
        # quoting the question -- see make_prose_fixture.compliant_text -- so a
        # pack that omits it is a pack the harness cannot write against.
        "work_type": "synthetic.wide",
        "question": "How do the 39 engineered figures hold together?",
        "must_mention": [
            f"The figure numbered {i} anchors the bridge." for i in range(1, 9)
        ],
        "forbidden_claims": ["Nothing is guaranteed."],
        "computed": {f"metric_{i}": float(i) for i in range(1, 40)},
        "register": "ic_memo",
    }


# --------------------------------------------------------------------------
# Amendment §D: one official number per quantity, spelled the way a desk spells
# it. All three rules grade the *answer*; the question stays free to print
# whatever rounding reads well, which is exactly why the union of
# `allowed_numbers` could not be the authority for both.
# --------------------------------------------------------------------------

#: A pack that declares the contract, as every shipped computer now does.
#: `active_bps` is the amendment's own worked example: the arithmetic says
#: 372.6, the question opens "behind the policy mix by 373 bp".
CONTRACT_PACK = {
    "allowed_numbers": [372.6, 373.0, 430567.0],
    "canonical": {"active_bps": 372.6, "shares": 430567.0},
    "aliases": {"active_bps": ["373"]},
    "display": {"shares": "430,567"},
    "as_of": "2026-03-31",
    "must_mention": [],
    "forbidden_claims": [],
    "register": "",
}


def test_an_answer_may_claim_the_canonical_figure():
    assert rounding_drift(CONTRACT_PACK, "Active return is 372.6 bp.") == []
    assert integer_format_offenders(CONTRACT_PACK, "We work 430,567 shares.") == []


def test_the_questions_rounding_in_an_answer_is_named_drift():
    hits = rounding_drift(CONTRACT_PACK, "Active return is 373 bp on the period.")
    assert len(hits) == 1
    assert hits[0].startswith(ROUNDING_DRIFT_TAG)
    assert "active_bps" in hits[0] and "372.6" in hits[0]


def test_drift_reads_whole_numbers_not_prefixes_of_them():
    """The canonical figure contains its own alias as a prefix: 62.1 holds 62.

    Without the boundary this rejected every correct answer whose official
    number happened to begin with the digits of its own rounding -- which is
    most of them, and would have made the axis unusable on its first live run.
    """
    pack = {**CONTRACT_PACK, "canonical": {"x": 62.1}, "aliases": {"x": ["62"]}}
    assert rounding_drift(pack, "The figure is 62.1 exactly.") == []
    assert rounding_drift(pack, "The figure is 62 exactly.") != []


def test_a_sign_stripped_alias_is_a_spelling_not_a_drift():
    """`abs(round(-164.0))` is 164: the same number, not a different one."""
    pack = {**CONTRACT_PACK, "canonical": {"x": -164.0}, "aliases": {"x": ["164"]}}
    assert rounding_drift(pack, "The book ran 164 bp behind.") == []


def test_an_integer_with_a_decimal_tail_is_quoted_with_its_desk_spelling():
    """The repair turn must be able to say what to write, not only what is wrong."""
    assert integer_format_offenders(CONTRACT_PACK, "We work 430567.0 shares.") == [
        "'430567.0' -- write 430,567"
    ]


def test_the_answer_gate_grades_canonical_while_the_union_keeps_the_alias():
    """The split §D exists for, in one assertion pair."""
    assert canonical_numbers(CONTRACT_PACK) == [372.6, 430567.0]
    assert 373.0 in CONTRACT_PACK["allowed_numbers"]


def test_a_pack_without_a_contract_grades_exactly_as_it_used_to():
    """Backward compatibility as a property, not a hope: no canonical, no change."""
    assert canonical_numbers(PACK) == [float(x) for x in PACK["allowed_numbers"]]
    assert rounding_drift(PACK, _clean_text()) == []


# --------------------------------------------------------------------------
# Amendment §C: register is a gate. Driven through `gate_violations` rather
# than the register module directly -- the point of §C is that the *repair
# loop* sees these, and the repair loop calls this one function.
# --------------------------------------------------------------------------


def _desk_pack() -> dict:
    return {**PACK, "register": "desk_chat"}


def test_a_desk_chat_row_wearing_memo_headings_is_a_violation():
    violations = gate_violations(
        _desk_pack(), "Finding: the book is long.\n" + _clean_text(), "abstention"
    )
    assert any("memo headings" in v for v in violations)


def test_a_desk_chat_row_in_plain_prose_clears_the_register_axis():
    assert not any(
        "register desk_chat" in v
        for v in gate_violations(_desk_pack(), _clean_text(), "abstention")
    )


def test_a_risk_committee_row_that_names_no_constraint_is_a_violation():
    pack = {**PACK, "register": "risk_committee"}
    violations = gate_violations(pack, _clean_text(), "abstention")
    assert any("names no limit, horizon or assumption" in v for v in violations)


def test_an_ic_memo_may_wear_headings_but_must_make_a_call():
    """§C's table plus the one requirement it leaves out.

    Headings are permitted and not required -- a memo that leads with its call
    is a memo. What is required is the call itself: an investment-committee
    memo exists to produce a decision, and one that surveys the evidence and
    stops is a research note with the wrong label. It is also the only
    row-level lever against register collapse, since it gives ic_memo a habit
    desk_chat does not have.
    """
    pack = {**PACK, "register": "ic_memo"}
    call = " Our call is to hold the position."
    headed = "Finding: the book is long.\n" + _clean_text() + call
    assert not any(
        "register ic_memo" in v for v in gate_violations(pack, headed, "abstention")
    )
    # No call: a survey, not a memo.
    assert any(
        "states no call" in v
        for v in gate_violations(pack, _clean_text(), "abstention")
    )
    # The exam's closing is still not a memo's call.
    closed = _clean_text() + call + " FINAL ANSWER: sell."
    assert any("FINAL ANSWER" in v for v in gate_violations(pack, closed, "abstention"))


def test_registers_that_read_alike_are_measurably_close():
    """Axis 16's instrument, in isolation.

    A per-row gate cannot see register collapse -- no single row is wrong when
    four voices become one -- so the distinctiveness question is asked over a
    profile of the slice. This pins that the profile actually separates prose
    that differs and fails to separate prose that does not, which is the only
    property that makes the axis worth reading.
    """
    terse = ["Cost is 12 bp. Work it patiently. Done."] * 10
    memo = [
        "Finding: the book is long duration against its policy weight, and the "
        "carry no longer compensates for the convexity being given up. "
        "Evidence: the bridge shows the whole gap in allocation. "
        "Our call is to trim the overweight into the next print."
    ] * 10
    near = register_profile(terse)
    far = register_profile(memo)
    assert profile_distance(near, far) >= REGISTER_MIN_SEPARATION
    # A voice compared with itself is, correctly, not separated at all.
    assert profile_distance(near, register_profile(terse)) == 0.0


def test_the_desk_chat_ceiling_is_reachable_inside_every_lanes_word_floor():
    """A register and a word budget must not be contracts that cannot both close.

    ``execution.tca.arrival`` speaks desk_chat in every family and still emits
    `memo` records at a 200-word floor. At a flat twelve sentences that is a
    17-word average and at the band's top a 46-word one, so the ceiling scales
    -- and this is the assertion that says by how much, rather than leaving it
    to be discovered by a lane that dead-letters every row forever.
    """
    from pipelines.v3.teacher.prompts import WORD_BUDGETS

    for kind, (low, _high) in WORD_BUDGETS.items():
        ceiling = desk_chat_ceiling(kind)
        assert ceiling >= DESK_CHAT_MAX_SENTENCES
        assert low / ceiling <= DESK_CHAT_WORDS_PER_SENTENCE + 1e-9, kind
