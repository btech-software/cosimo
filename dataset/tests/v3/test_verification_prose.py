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
    makes_a_call,
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
        "for every desk that checks the file against the figures it was drawn from."
    )


def test_clean_text_itself_sits_in_the_abstention_band():
    """Read from the table rather than pinned beside it.

    The bands moved when §B.2 split kind from register, and a helper pinned to
    two literals would have gone on claiming a budget that no longer exists.
    """
    from pipelines.v3.teacher.prompts import KIND_SENTENCE_CAPS, word_budget
    from pipelines.v3.verification.register import sentence_count

    low, high = word_budget("abstention", PACK["register"])
    words = len(_clean_text().split())
    assert low <= words <= high, words
    assert sentence_count(_clean_text()) <= KIND_SENTENCE_CAPS["abstention"]


def test_clean_produce_no_violations():
    assert gate_violations(PACK, _clean_text(), "abstention") == []


def test_invented_number_is_named_not_numbered():
    # Folded into the last sentence rather than added as a sixth: an
    # abstention has a five-sentence ceiling now, and a test about naming a
    # token should not be measuring that.
    violations = gate_violations(
        PACK,
        _clean_text().rstrip(". ") + ", on a terminal multiple of 8153.7729.",
        "abstention",
    )
    assert len(violations) == 1
    assert "invented numbers" in violations[0]
    assert "8153.7729" in violations[0], "the repair prompt must name the token"


def test_missing_mention_is_reported_point_by_point():
    missing = missing_mentions(PACK, "Capex is rising.")
    assert missing == ["The dcf value sits below price."]
    # `analysis`, not `abstention`: an abstention answers the question the pack
    # *cannot*, so its points come from `abstention_missing` rather than from
    # the list describing a good answer to the pack's own question.
    violations = gate_violations(PACK, "Capex is rising. " * 40, "analysis")
    assert any("points the answer does not engage" in v for v in violations)


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
    # The date's own tokens, read the way an answer is read. They used to be
    # "-03"/"-31" -- the shared tokenizer takes a hyphen before a digit as a
    # sign -- and this test asserted that shape, which is how the reading
    # survived long enough to dead-letter a live VaR row for an invented
    # "-100" it had written as "a 1-in-100 day". The behaviour below is the
    # actual contract and is unchanged either way.
    assert {"2026", "03", "31"} <= wl, "the pack's own as-of date may be cited"
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
        # Four, not eight. A `grounded` row is 90 words now, and eight anchors
        # are ~60 words before a single figure is quoted -- a pack whose points
        # alone overflow the tightest band is an authoring error, and the
        # fixture should not model one. The real plan's packs carry one to
        # three; four keeps the stress above reality without inventing a pack
        # nobody could answer.
        "must_mention": [
            f"The figure numbered {i} anchors the bridge." for i in range(1, 5)
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
        "register ic_memo" in v for v in gate_violations(pack, headed, "analysis")
    )
    # No call: a survey, not a memo.
    assert any(
        "states no call" in v for v in gate_violations(pack, _clean_text(), "analysis")
    )
    # The exam's closing is still not a memo's call.
    closed = _clean_text() + call + " FINAL ANSWER: sell."
    assert any("FINAL ANSWER" in v for v in gate_violations(pack, closed, "analysis"))


def test_the_two_kinds_whose_brief_forbids_a_call_are_not_asked_for_one():
    """§B.2, where the kind and the register would otherwise contradict.

    `grounded` is told "no call unless the question asked for one" and
    `abstention` is told to name what is missing and stop. Every attribution
    family is `ic_memo`, so without this exemption both kinds would be briefed
    not to decide and gated for not deciding -- a teacher given two instructions
    it cannot both obey, which is the failure this file's own history is made
    of.
    """
    pack = {**PACK, "register": "ic_memo"}
    survey = _clean_text()
    assert not makes_a_call(survey)
    for kind in ("grounded", "abstention"):
        assert not any(
            "states no call" in v for v in gate_violations(pack, survey, kind)
        ), kind
    # The kinds that *are* documents ending in a decision keep the requirement.
    for kind in ("analysis", "memo", "critique"):
        assert any(
            "states no call" in v for v in gate_violations(pack, survey, kind)
        ), kind


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

    from pipelines.v3.teacher.prompts import KIND_SENTENCE_CAPS

    for kind, (low, _high) in WORD_BUDGETS.items():
        ceiling = desk_chat_ceiling(kind)
        # Twelve unless the *kind* is tighter: a citation is eight sentences
        # and a refusal is five in every register, and the register may not
        # quietly offer four more than the kind's own brief asks for.
        cap = KIND_SENTENCE_CAPS.get(kind)
        assert ceiling == cap if cap else ceiling >= DESK_CHAT_MAX_SENTENCES
        assert low / ceiling <= DESK_CHAT_WORDS_PER_SENTENCE + 1e-9, kind


def test_a_memo_that_states_its_call_as_a_heading_is_making_a_call():
    """The form the teacher actually uses, which the first version missed.

    Three of four live ic_memo rows were dead-lettered for "states no call"
    while every one of them ended `Call: <decision>`. The heading *is* the
    call -- it is the form §C's own table licenses for this register -- and
    the gate was already matching it in `_MEMO_HEADINGS` to permit memo
    scaffolding. One regex said yes and one phrase list said no about the same
    four characters, and the phrase list won.
    """
    pack = {**PACK, "register": "ic_memo"}
    heading = _clean_text() + "\nCall: use 2500 as the central case."
    assert makes_a_call(heading)
    assert not any(
        "states no call" in v for v in gate_violations(pack, heading, "abstention")
    )
    # The phrase form still counts, and so does Recommendation:.
    assert makes_a_call("We would trim the overweight.")
    assert makes_a_call("Recommendation: hold at the current weight.")
    # A survey that reaches no decision is still a survey.
    assert not makes_a_call("Finding: the book is long. Evidence: the bridge shows it.")


def test_desk_chat_may_not_label_its_call_even_mid_paragraph():
    """The memo speech act, wherever it sits. Found in a live capture.

    Two live desk_chat rows ended `... Call: execute under current
    participation, but cap the schedule` on the same line as the prose before
    it, and both scored `register_ok: true`. The heading regex is anchored at
    a line start -- rightly, since "the finding: we are long" mid-sentence is
    prose -- so the label escaped, and the register field was decoration for
    exactly the rows it exists to judge.
    """
    desk = {**PACK, "register": "desk_chat"}
    labelled = "Work it patiently at this size. Call: execute under the cap."
    assert any(
        "labels its call" in v for v in gate_violations(desk, labelled, "abstention")
    )
    # A line-start label is caught too -- by the heading rule, which still owns
    # "this is scaffolded like a memo".
    assert gate_violations(desk, "Work it patiently.\nCall: execute.", "abstention")
    # The same decision, unlabelled, is what the desk actually writes.
    plain = _clean_text() + " Work it patiently and cap the schedule."
    assert not any(
        "labels its call" in v for v in gate_violations(desk, plain, "abstention")
    )
    # ic_memo is unaffected: the label is its licensed form (§C's table).
    memo = {**PACK, "register": "ic_memo"}
    assert not any(
        "labels its call" in v
        for v in gate_violations(memo, _clean_text() + " Call: hold.", "abstention")
    )


# --------------------------------------------------------------------------
# the brief and the gate must describe the same shape
# --------------------------------------------------------------------------


def test_the_brief_states_every_shape_rule_the_register_gate_enforces():
    """A gate the writer cannot see is a trap, and it is billed per row.

    ``desk_chat`` was briefed as "short sentences, no preamble, no sign-off"
    and refused for memo headings, for a labelled ``Call:``, and for passing a
    sentence ceiling it was never shown -- five of the first nine rows of a
    live think-off render died on rules nobody had told the model. This pins
    the repair: whatever the gate refuses, the brief says out loud.
    """
    from pipelines.v3.teacher.prompts import render_brief
    from pipelines.v3.verification.register import (
        _REGISTER_SHAPE,
        desk_chat_ceiling,
        register_shape,
    )

    for register in _REGISTER_SHAPE:
        pack = {
            "question": "q",
            "register": register,
            "allowed_numbers": [1.0],
            "must_mention": [],
            "forbidden_claims": [],
        }
        user = render_brief(pack, kind="analysis")[-1]["content"]
        assert register_shape(register, "analysis") in user, (
            f"{register}: the gate's shape rules are not in the brief"
        )

    desk = render_brief(
        {
            "question": "q",
            "register": "desk_chat",
            "allowed_numbers": [1.0],
            "must_mention": [],
            "forbidden_claims": [],
        },
        kind="analysis",
    )[-1]["content"]
    assert str(desk_chat_ceiling("analysis")) in desk, (
        "the sentence ceiling is a number the gate measures against; the brief "
        "has to carry that same number, not a synonym for 'be brief'"
    )


def test_a_register_with_no_shape_rules_adds_nothing_to_the_brief():
    """``auditor``/``code_review`` are declared but unwritten -- §C invents no
    rule for them, and neither may the brief."""
    from pipelines.v3.verification.register import register_shape

    assert register_shape("auditor", "analysis") == ""
    assert register_shape("code_review", "analysis") == ""


def test_the_repair_turn_does_not_tell_a_too_long_draft_to_keep_its_length():
    """Two instructions a draft cannot both obey is a wasted attempt, x3.

    "Do not shorten what was compliant" is the right advice for a numbers or
    an anchor fault and the exact opposite of the fix for a length one. The
    first row of a corrected live render spent all three attempts at 13
    sentences against a ceiling of 12, told each time to cut and to keep.
    """
    from pipelines.v3.teacher.prompts import render_repair
    from pipelines.v3.verification.register import register_violations

    messages = [{"role": "user", "content": "brief"}]
    long_draft = ". ".join(["a b c"] * 20) + "."
    too_long = register_violations("desk_chat", long_draft, kind="analysis")
    assert too_long, "the fixture draft must actually trip the ceiling"

    repair = render_repair(messages, long_draft, too_long)[-1]["content"]
    assert "Do not shorten" not in repair
    assert "cut at least" in repair, (
        "the repair quotes the violation, so the violation has to name a target"
    )

    other = ["gate: invented numbers not in the fact pack: '9.8'"]
    kept = render_repair(messages, "a short draft.", other)[-1]["content"]
    assert "Do not shorten what was compliant." in kept, (
        "the clause is dropped only for the fault it contradicts"
    )


def test_an_empty_draft_still_gets_the_empty_draft_turn():
    """The length branch must not shadow the empty-draft branch."""
    from pipelines.v3.teacher.prompts import render_repair

    repair = render_repair(
        [{"role": "user", "content": "brief"}],
        "",
        ["register desk_chat runs 13 sentences, over the 12-sentence ceiling"],
    )[-1]["content"]
    assert "You returned an empty answer" in repair


def test_the_decimal_tail_advice_does_not_name_a_token_rounding_drift_refuses():
    """Two gates, opposite instructions -- the live case, pinned.

    An attribution memo wrote ``373.0`` for an ``active_bps`` of 372.6. The
    format axis said "write 373"; the model obeyed; ``rounding_drift`` refused
    373 on the next attempt as the question's spelling of the figure. Three
    attempts, one dead letter, and neither axis was wrong on its own.
    """
    from pipelines.v3.verification.prose import (
        integer_format_offenders,
        rounding_drift,
    )

    pack = {
        "canonical": {"active_bps": 372.6},
        "aliases": {"active_bps": ["373"]},
        "display": {},
    }
    advice = integer_format_offenders(pack, "active return of 373.0 basis points")
    assert advice == ["'373.0' -- write 372.6"], advice

    # And the figure it now names is one the other axis accepts.
    assert rounding_drift(pack, "active return of 372.6 basis points") == []
    assert rounding_drift(pack, "active return of 373 basis points"), (
        "the premise: the stripped whole is exactly what the drift axis refuses"
    )


def test_a_plain_decimal_tail_still_gets_the_desk_spelling():
    """The alias lookup must not swallow the ordinary case it was built around."""
    from pipelines.v3.verification.prose import integer_format_offenders

    pack = {"display": {"shares": "430,567"}, "canonical": {}, "aliases": {}}
    assert integer_format_offenders(pack, "430567.0 shares") == [
        "'430567.0' -- write 430,567"
    ]


def test_a_zero_written_to_a_tenth_is_not_a_count_with_false_precision():
    """The gate was refusing the pack's own figure, and the model knew it.

    ``effects_Industrials.allocation_bps`` is canonically ``0.0``. A live
    attribution row wrote it as ``0.0``, was told "write 0", and returned an
    identical 256 words three times rather than misreport the pack. Zero in a
    decomposition column is read down against ``-0.3`` and ``+0.2``; the
    "a count is not known to a tenth" argument is about ``430567.0`` and does
    not reach it.
    """
    from pipelines.v3.verification.prose import integer_format_offenders

    pack = {"display": {}, "canonical": {"alloc": 0.0}, "aliases": {}}
    assert integer_format_offenders(pack, "allocation 0.0, selection -0.3") == []
    assert integer_format_offenders(pack, "interaction -0.0 bps") == []

    # The case the axis exists for is untouched.
    counts = {"display": {"shares": "430,567"}, "canonical": {}, "aliases": {}}
    assert integer_format_offenders(counts, "430567.0 shares") == [
        "'430567.0' -- write 430,567"
    ]


def test_the_sentence_ceiling_cannot_be_met_by_writing_longer_sentences():
    """A live desk_chat memo ran 334 words in 7 sentences and scored clean.

    48 words a sentence is not desk chat by any reading; the ceiling counts
    full stops and a memo can simply use fewer of them.
    """
    from pipelines.v3.verification.register import (
        DESK_CHAT_MAX_MEAN_SENTENCE,
        register_violations,
    )

    long_ones = " ".join(["word"] * 60 + ["."]) + " " + " ".join(["word"] * 60) + "."
    found = register_violations("desk_chat", long_ones, kind="analysis")
    assert any("words a sentence" in v for v in found), found

    terse = ". ".join(" ".join(["word"] * 8) for _ in range(6)) + "."
    assert register_violations("desk_chat", terse, kind="analysis") == []
    assert DESK_CHAT_MAX_MEAN_SENTENCE == 28


def test_the_mean_sentence_rule_is_in_the_brief_like_every_other_one():
    """The session's own lesson, applied to the rule it just added."""
    from pipelines.v3.verification.register import (
        DESK_CHAT_MAX_MEAN_SENTENCE,
        register_shape,
    )

    assert str(DESK_CHAT_MAX_MEAN_SENTENCE) in register_shape("desk_chat", "analysis")


# --------------------------------------------------------------------------
# record type: the move, as distinct from the register's voice
# --------------------------------------------------------------------------


def test_every_record_type_asks_for_something_different():
    """Before this, an `analysis` brief and a `grounded` brief differed by the
    literal task string and two overlapping word budgets. Nothing else."""
    from pipelines.v3.teacher.prompts import BRIEF_KINDS, kind_shape

    shapes = {kind: kind_shape(kind) for kind in BRIEF_KINDS}
    assert all(shapes.values()), [k for k, v in shapes.items() if not v]
    assert len(set(shapes.values())) == len(BRIEF_KINDS), "two kinds share a rule"


def test_the_task_rules_reach_the_brief():
    """And they reach it *in this register*: the caps are part of the job."""
    from pipelines.v3.teacher.prompts import BRIEF_KINDS, kind_shape, render_brief

    pack = {
        "question": "q",
        "register": "desk_chat",
        "work_type": "execution.tca.arrival",
        "allowed_numbers": [1.0],
        "must_mention": [],
        "forbidden_claims": [],
    }
    for kind in BRIEF_KINDS:
        user = render_brief(pack, kind=kind)[-1]["content"]
        assert kind_shape(kind, pack["register"]) in user, kind
    # A desk analysis is 160 words where a committee analysis is 220, and the
    # brief says which one this row is being written to.
    desk = render_brief(pack, kind="analysis")[-1]["content"]
    memo = render_brief({**pack, "register": "ic_memo"}, kind="analysis")[-1]["content"]
    assert "60-160 words" in desk and "60-220 words" in memo


def test_the_work_type_addendum_reaches_the_brief_and_only_its_own():
    """§B.4: four arithmetic disciplines, each paid for on its own rows."""
    from pipelines.v3.teacher.prompts import render_brief, work_type_rules

    base = {
        "question": "q",
        "register": "desk_chat",
        "allowed_numbers": [1.0],
        "must_mention": [],
        "forbidden_claims": [],
    }
    tca = render_brief({**base, "work_type": "execution.tca.arrival"}, kind="analysis")
    user = tca[-1]["content"]
    assert "impact bill" in user and "not a pacing problem" in user
    assert work_type_rules("risk.market.var_es") not in user
    # A work type with no addendum gets no filler.
    assert work_type_rules("no.such.work_type") == ""


def test_the_number_policy_quotes_this_packs_own_spellings():
    """§A.3: `display` was in the pack and never in the brief."""
    from pipelines.v3.teacher.prompts import number_policy

    policy = number_policy({"display": {"shares": "430,567"}})
    assert "shares is written 430,567" in policy
    assert "430567.0" in policy, "the rule names the spelling it refuses"


def test_the_system_turn_still_carries_the_fingerprint_the_leak_gate_matches():
    """Shortening the system turn must not delete what proves a row is clean.

    `row.carries_teacher_brief` and the harness's prepare gate both search for
    `config.TEACHER_FINGERPRINT`. A system turn that stopped containing it
    would turn every leak check green by removing what they look for, which is
    the worst way for a gate to pass.
    """
    from pipelines.v3.teacher.prompts import AGENTIC_SYSTEM, TEACHER_SYSTEM

    assert config.TEACHER_FINGERPRINT in TEACHER_SYSTEM
    assert config.TEACHER_FINGERPRINT in AGENTIC_SYSTEM
    # §B.1 asks for eight lines at the outside; it was seventeen.
    assert len(TEACHER_SYSTEM.splitlines()) <= 8


def test_no_task_rule_contradicts_the_register_it_will_be_written_in():
    """One pack serves every record type, so the two briefs must compose.

    The older form of this test forbade a kind rule from mentioning headings,
    sentences or length at all. §B.2 moved exactly those decisions onto the
    kind on purpose -- a `grounded` row carries no headings in any register, an
    `abstention` is five sentences wherever it is written -- so the rule that
    survives is the one that mattered: a kind may not *demand* what a register
    *refuses*, because a teacher told both has been given no brief.

    The two pairings that could contradict are checked directly, and the third
    is checked by being illegal: `memo` names the headings its register allows,
    and `memo` x `desk_chat` is no longer a row the inventory will emit.
    """
    from pipelines.v3 import config as v3config
    from pipelines.v3 import inventory
    from pipelines.v3.teacher.prompts import BRIEF_KINDS, kind_shape
    from pipelines.v3.verification.register import _REGISTER_SHAPE

    for kind in BRIEF_KINDS:
        rule = kind_shape(kind).casefold()
        assert rule, kind
        assert "final answer:" not in rule, f"{kind} reaches for the exam contract"

    # `memo` is the one kind that legislates headings, and it may only be
    # written where headings are licensed.
    plan = inventory.load_plan(v3config.taxonomy_path())
    for work_type, spec in plan.items():
        if "memo" not in spec["record_types"]:
            continue
        for family in spec["families"]:
            assert not inventory.illegal_triple(spec, family, "memo"), (
                f"{work_type}/{family} emits a memo in a register that refuses "
                "the headings the memo brief asks for"
            )

    # `grounded` and `abstention` are told not to decide; the register that
    # requires a decision exempts exactly those two.
    for kind, refusal in (
        ("grounded", "no call"),
        ("abstention", "missing quantity"),
    ):
        assert refusal in kind_shape(kind).casefold(), kind
    assert "call" in _REGISTER_SHAPE["ic_memo"].casefold()
    # ...and the brief for those two kinds must not ask for the call the gate
    # has already excused them from. A live grounded row spent three attempts
    # writing the "Call:" its register demanded and its kind refused.
    from pipelines.v3.verification.register import register_shape

    for kind in ("grounded", "abstention"):
        shape = register_shape("ic_memo", kind).casefold()
        assert "end on an explicit call" not in shape, kind
        assert "do not close on a labelled call" in shape, kind
    assert "explicit call" in register_shape("ic_memo", "memo").casefold()
    assert not any(
        "states no call" in v
        for v in gate_violations(
            {**PACK, "register": "ic_memo"}, _clean_text(), "grounded"
        )
    )


def test_the_two_desk_chat_length_rules_are_jointly_satisfiable_for_every_kind():
    """A row must never face two rules it cannot both obey.

    When the mean-sentence rule was first added it was derived independently of
    the sentence ceiling, and for four of the five kinds the pair had no
    solution at the top of the word band: a 341-word memo came back at 16
    sentences against a ceiling of 15, having broken its sentences up exactly
    as instructed. Both bounds now come off the same band, and this asserts
    they close.
    """
    from pipelines.v3.teacher.prompts import WORD_BUDGETS
    from pipelines.v3.verification.register import (
        DESK_CHAT_MAX_MEAN_SENTENCE,
        desk_chat_ceiling,
    )

    for kind in WORD_BUDGETS:
        low, high = WORD_BUDGETS[kind]
        assert low <= high, kind
        ceiling = desk_chat_ceiling(kind)
        assert high <= ceiling * DESK_CHAT_MAX_MEAN_SENTENCE, (
            f"{kind}: {high} words in {ceiling} sentences needs "
            f"{high / ceiling:.0f} words a sentence, over the "
            f"{DESK_CHAT_MAX_MEAN_SENTENCE} mean"
        )


def test_a_verb_matches_its_own_conjugation():
    """ "scaling" and "scale" are one word, and the matcher said otherwise.

    A live VaR row wrote "the 10-day figures scale by the square-root of the
    horizon" and was refused three times against an anchor of "square-root
    horizon scaling" -- every term present, one of them merely conjugated.
    """
    from pipelines.v3.verification.prose import _stem, missing_mentions

    assert len({_stem(w) for w in ("scale", "scaling", "scales", "scaled")}) == 1
    assert _stem("assume") == _stem("assumption")

    pack = {"must_mention": ["square-root horizon scaling"]}
    covered = "the 10-day figures scale by the square-root of the horizon"
    assert missing_mentions(pack, covered) == []
    assert missing_mentions(pack, "the ten day figures are larger") == [
        "square-root horizon scaling"
    ]


def test_short_function_words_survive_the_new_suffix():
    """The three-character floor is what makes stripping a trailing "e" safe."""
    from pipelines.v3.verification.prose import _stem

    for word in ("the", "are", "one", "use", "due"):
        assert _stem(word) == word, word


# --------------------------------------------------------------------------
# text that is broken rather than wrong
# --------------------------------------------------------------------------


def test_a_bracket_welded_to_the_next_word_is_a_defect():
    """Four of twenty-four live rows shipped carrying this, clean on 16 axes.

        ... a tail thinner than this book carries)Skip the headline: the
        mechanism is square-root scaling ...

    It predates the record-type rules, so it is not an instruction being
    narrated back, and `think_present` was false on every row that carried it.
    """
    from pipelines.v3.verification.prose import malformed_prose

    found = malformed_prose("a tail thinner than this book carries)Skip the headline")
    assert found and "runs straight into the next word" in found[0]
    assert malformed_prose("the total was [a]nd then")


def test_well_formed_prose_is_not_flagged():
    """The gate must be free of opinions about style."""
    from pipelines.v3.verification.prose import malformed_prose

    assert malformed_prose("The cost is 28 bp (impact, not spread). Work it.") == []
    assert (
        malformed_prose("Impact (26.66 bp) dominates the half-spread (1.5 bp).") == []
    )
    assert malformed_prose("") == []


def test_unbalanced_brackets_are_reported():
    from pipelines.v3.verification.prose import malformed_prose

    assert any("unbalanced" in v for v in malformed_prose("the cost (impact is 26 bp"))
    assert malformed_prose("the cost (impact) is 26 bp") == []


def test_the_malformed_check_runs_before_the_judgements_below_it():
    """The repair turn reads in order, and a broken row wastes the rest."""
    from pipelines.v3.verification.prose import gate_violations

    pack = {
        "question": "q",
        "register": "desk_chat",
        "allowed_numbers": [1.0],
        "must_mention": [],
        "forbidden_claims": [],
        "canonical": {},
        "aliases": {},
        "display": {},
    }
    found = gate_violations(
        pack, "a broken aside)Skip the headline and 9.8 too", "analysis"
    )
    assert "runs straight into the next word" in found[0], found


# --------------------------------------------------------------------------
# the v3.2 capture's review: four rules the first live sample asked for
# --------------------------------------------------------------------------


def test_a_row_may_not_name_the_machinery_that_produced_it():
    """Three of eight rows in the first v3.2 capture did.

    The system turn already said "never mention the fact pack, the contract,
    the gate"; nothing enforced it, and a student row that says "the pack holds
    no decision price" is teaching the labelling protocol, which is the failure
    the two-surface split exists to end.
    """
    from pipelines.v3.verification.prose import contract_leaks

    for leaked in (
        "This pack holds no decision price.",
        "The pack's reconciling residual is 0.1 bp.",
        "Every must_mention point is covered.",
        "The figures come from allowed_numbers.",
    ):
        assert contract_leaks(leaked), leaked
        assert any(
            "names the machinery" in v
            for v in gate_violations(PACK, _clean_text() + " " + leaked, "abstention")
        ), leaked
    # The same facts, said the way a desk says them.
    for clean in (
        "There is no decision price here.",
        "The residual is 0.1 bp.",
        "A packed order book is not the point.",
    ):
        assert contract_leaks(clean) == [], clean


def test_a_word_fused_to_a_figure_is_refused():
    """`$39.248Mchers` shipped board-green; a gate that misses it is a gate."""
    from pipelines.v3.verification.prose import malformed_prose

    assert malformed_prose("the 95% VaR is $39.248Mchers today")
    assert malformed_prose("interaction totals 30.8 bpches")
    # A live valuation abstention shipped "83.0M shareshare count".
    assert malformed_prose("implied equity over 83.0M shareshare count")
    # Prose that merely puts a word after a number is prose.
    for fine in (
        "fill at 295.77 against 296.61 arrival",
        "28.16bp of arrival on a 10-day horizon",
        "participation of 4.86% of ADV",
        # The first draft of this rule read "es" + "timate" and dead-lettered a
        # correct valuation memo three attempts running. Every unit that opens
        # an English word is out of the pattern, and these are the words that
        # bought that lesson.
        "the central estimate is 21.98 against the peer set",
        "an advance on the variance of various names",
        "the ADV is the anchor and the VAR is a quantile",
    ):
        assert malformed_prose(fine) == [], fine


def test_the_desk_spelling_wins_on_a_rounded_display_too():
    """§A.3 beyond the percent case: a price is quoted to the cent."""
    from pipelines.v3.packs import compute_pack
    from pipelines.v3.verification.prose import display_form_offenders

    pack = compute_pack("execution.tca.arrival", "large_cap_intraday", 0).to_dict()
    assert display_form_offenders(pack, "the fill is 295.7747") == [
        "'295.7747' -- write 295.77"
    ]
    assert display_form_offenders(pack, "the fill is 295.77 on 296.61 arrival") == []
    # An integer keeps its bare form: the separators are a house style, not a
    # precision claim, and §A.3 permits either.
    assert display_form_offenders(pack, "430567 shares against 8867535 ADV") == []


def test_grounded_is_materially_shorter_than_the_analysis_beside_it():
    """A citation that can only be told from an argument by reading it is not
    yet a second record type: the first v3.2 capture put an 80-word grounded
    row beside an 81-word analysis of the same pack."""
    from pipelines.v3.teacher.prompts import word_budget

    for register in ("desk_chat", "ic_memo", "risk_committee"):
        _, grounded = word_budget("grounded", register)
        _, analysis = word_budget("analysis", register)
        assert grounded * 1.5 <= analysis, register
