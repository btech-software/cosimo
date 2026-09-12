"""§D: the three claims a pack's own numbers refute, and where they are read.

The nine-row live sample is the argument for this file. Every row cleared
sixteen axes; three of them said something the pack contradicts, and none of
the three invented a number -- which is why nothing caught them. The checks are
cheap, so they run everywhere a row is judged: inside the render's repair loop,
on the verify board, and over a whole slice before anybody scales from it.
"""

from __future__ import annotations

import os

from pipelines.v3 import row as rowlib, write
from pipelines.v3.packs import compute_pack
from pipelines.v3.render.prose import row_id_from_coords
from pipelines.v3.verification.contradictions import (
    RECONCILE_TOLERANCE_BPS,
    TAGS,
    contradiction_violations,
    tag_of,
)
from pipelines.v3.verification.prose import gate_violations
from pipelines.v3.verify_v3 import _check_row

TCA = ("execution.tca.arrival", "large_cap_intraday", 0)
VAR = ("risk.market.var_es", "rates_book", 0)
ATTRIBUTION = ("portfolio.attribution.brinson_carino", "global_equity_long", 0)


def _pack(coords) -> dict:
    return compute_pack(*coords).to_dict()


def _tags(pack: dict, text: str) -> list[str]:
    return [tag_of(v) for v in contradiction_violations(pack, text)]


# --------------------------------------------------------------------------
# 1. the schedule is not the risk below the cap
# --------------------------------------------------------------------------


def test_the_schedule_is_not_the_risk_under_the_cap():
    """The live desk note said it both ways in one paragraph, on a 4.86% clip."""
    pack = _pack(TCA)
    assert pack["canonical"]["participation"] < (
        pack["canonical"]["participation_cap_pct"] / 100.0
    )
    for claim in (
        "At this size the schedule itself becomes the risk.",
        "The binding constraint is the schedule, not the spread.",
        "Here the risk is the schedule rather than the impact.",
    ):
        assert _tags(pack, claim) == ["schedule_risk_below_cap"], claim


def test_naming_the_impact_as_the_cost_is_not_a_contradiction():
    """The axis has to be silent on the answer §F actually wants."""
    pack = _pack(TCA)
    good = (
        "Implementation shortfall is 28.16 bp against arrival, 26.66 of it "
        "square-root impact. Participation is 4.86% of ADV, under the 10% cap, "
        "so this is an impact bill and not a pacing problem. Work the schedule "
        "at this participation."
    )
    assert contradiction_violations(pack, good) == []


def test_denying_the_claim_is_not_making_it():
    """The sentence §F asks for must not be the sentence §D refuses.

    "The schedule is not the risk at this participation" contains every word
    the pattern looks for, and a gate that read it as the claim would push the
    teacher into never mentioning the schedule at all -- which is not the
    answer to a question that asks which of the two costs is being managed.
    """
    pack = _pack(TCA)
    for denial in (
        "The schedule is not the risk at 4.86% of ADV.",
        "Only near the 10% cap does the schedule become the risk.",
        "The binding constraint is the impact, not the schedule.",
        "The schedule would become the risk as participation approached the cap.",
    ):
        assert contradiction_violations(pack, denial) == [], denial


def test_an_emphatic_assertion_is_not_mistaken_for_a_denial():
    """The hole an over-generous negation list opened, closed by a test.

    An earlier draft excused a match whenever "only", "would be" or "until"
    appeared near it, on the theory that those mark a conditional. They also
    appear in plain assertions -- "the schedule is the only risk that matters"
    -- and a gate that reads emphasis as denial lets through exactly the rows
    §D exists to stop, which is the worse of the two errors: a false positive
    costs one repair turn, a false negative ships.
    """
    tca = _pack(TCA)
    for claim in (
        "At this size the schedule is the only risk that matters.",
        "It would be a mistake to think otherwise: the schedule is the risk here.",
    ):
        assert _tags(tca, claim) == ["schedule_risk_below_cap"], claim
    var = _pack(VAR)
    assert _tags(var, "The only reading here is a positive drift.") == ["drift_sign"]
    assert (
        contradiction_violations(var, "There is no positive drift on this book.") == []
    )


def test_a_denial_in_the_previous_clause_does_not_excuse_the_next():
    """Found in the first live render after the amendment.

    "the impact is the cost being managed here, not the schedule; the schedule
    itself becomes the risk only at or above the cap" is a true sentence, and
    it cleared the gate for the wrong reason -- the ``not`` of the first clause
    sat inside the six-word window of the second. A denial belongs to the
    clause it is spoken in.
    """
    pack = _pack(TCA)
    laundered = (
        "The impact is the cost being managed here, not the schedule; the "
        "schedule itself becomes the risk at this size."
    )
    assert _tags(pack, laundered) == ["schedule_risk_below_cap"]
    # The denial in its own clause still reads as one.
    assert (
        contradiction_violations(
            pack, "The impact is the bill; the schedule is not the risk here."
        )
        == []
    )


def test_a_pack_at_the_cap_is_allowed_the_claim():
    """The rule is the inequality, not the sentence."""
    pack = _pack(TCA)
    at_cap = {
        **pack,
        "canonical": {**pack["canonical"], "participation": 0.11},
    }
    assert contradiction_violations(at_cap, "The schedule is the risk now.") == []


# --------------------------------------------------------------------------
# 2. a negative mean is a negative drift
# --------------------------------------------------------------------------


def test_a_negative_daily_mean_is_never_a_gain():
    """The live memo turned a -0.10% mean into $10M of expected gain."""
    pack = _pack(VAR)
    assert pack["canonical"]["mu_daily"] < 0
    for claim in (
        "The positive drift offsets part of the quantile.",
        "Over ten days that is an expected gain against the VaR.",
        "The drift is in the book's favour.",
    ):
        assert _tags(pack, claim) == ["drift_sign"], claim
    honest = "The daily mean is negative, a drift that adds to the expected loss."
    assert contradiction_violations(pack, honest) == []


def test_a_book_with_a_positive_mean_may_say_so():
    pack = _pack(VAR)
    positive = {**pack, "canonical": {**pack["canonical"], "mu_daily": 0.0004}}
    assert contradiction_violations(positive, "There is a positive drift.") == []


# --------------------------------------------------------------------------
# 3. no winner among pieces that do not add up
# --------------------------------------------------------------------------


def test_an_unreconciled_pack_forbids_naming_an_effect_to_act_on():
    """The failure that produced a theory about a negative Carino factor."""
    pack = _pack(ATTRIBUTION)
    # As shipped, this pack reconciles, so the axis is silent even on the
    # sentence that answers the question.
    call = "Allocation is the skill signal worth acting on."
    assert contradiction_violations(pack, call) == []
    broken = {
        **pack,
        "canonical": {
            **pack["canonical"],
            "allocation_total_bps": 0.8,
            "selection_total_bps": 0.7,
            "interaction_total_bps": 0.1,
        },
    }
    assert _tags(broken, call) == ["unreconciled_call"]
    # Reporting the failure instead of deciding is exactly what §B.2 asks for.
    abstains = "The pieces do not reconcile with the active return; I would not act."
    assert contradiction_violations(broken, abstains) == []


def test_print_rounding_is_not_a_reconciliation_failure():
    """Three totals rounded to a tenth may miss the active by a tenth."""
    pack = _pack(ATTRIBUTION)
    canonical = pack["canonical"]
    drift = (
        canonical["allocation_total_bps"]
        + canonical["selection_total_bps"]
        + canonical["interaction_total_bps"]
        - canonical["active_bps"]
    )
    assert abs(drift) < RECONCILE_TOLERANCE_BPS


# --------------------------------------------------------------------------
# where they are read
# --------------------------------------------------------------------------


def test_every_violation_carries_a_tag_the_board_can_file():
    pack = _pack(TCA)
    violations = contradiction_violations(pack, "The schedule is the risk.")
    assert violations and all(tag_of(v) in TAGS for v in violations)
    assert tag_of("not a tag: something") == ""


def test_the_render_gate_refuses_a_contradiction():
    """Inside the repair loop, where a fix still costs one call."""
    pack = _pack(TCA)
    text = (
        "Shortfall is 28.16 bp against arrival with 26.66 bp of impact and a "
        "1.5 bp half-spread, implying a fill at 295.7747 on arrival 296.61. "
        "Participation is 4.86% of the 8,867,535 ADV, so the schedule itself "
        "becomes the risk. Square-root impact scaling against ADV is what "
        "drives it, measured against the arrival versus decision benchmark."
    )
    violations = gate_violations(pack, text, "analysis")
    assert any(v.startswith("schedule_risk_below_cap") for v in violations)


def test_the_board_files_a_contradiction_on_the_claims_axis(tmp_path):
    """And on a *shipped* row, graded against the recomputed pack."""
    work_type, family, variant = TCA
    pack = compute_pack(work_type, family, variant)
    rid = row_id_from_coords("analysis", work_type, family, variant)
    row = rowlib.student_row(
        row_id=rid,
        kind="analysis",
        pack=pack.to_dict(),
        holdout=False,
        answer=(
            "Cost is 28.16 bp of arrival. At 4.86% of ADV the schedule itself "
            "becomes the risk."
        ),
        stamp={"pack_seed": f"{pack.seed:016x}"},
        teacher={"model": "test", "finish_reason": "stop", "usage": {}},
        render={"kind": "analysis", "attempts": 1},
        attempts=1,
        invented=[],
        missing=[],
        register_ok=True,
    )
    write.write_jsonl(write.path_for("sft", "analysis", str(tmp_path)), [row])
    failures = _check_row(row, frozenset(), quick=True)
    claims = failures["must_mention / forbidden_claims"]
    assert any(str(problem).startswith("schedule_risk_below_cap") for problem in claims)
    assert os.path.isfile(write.path_for("sft", "analysis", str(tmp_path)))


def test_the_board_and_the_render_gate_agree_about_length_and_shape(tmp_path):
    """ "One policy, three call sites" has to be true of the length axis too.

    The board used to grade the register's sentence ceiling and not the kind's,
    and no word band at all -- so a row repaired at render time for running
    long was certified green by the auditor that came after it. A generator and
    an auditor that disagree about clean is the bug class v3 was written to
    make impossible, and it is worth a row-level test rather than a promise in
    a docstring.
    """
    work_type, family, variant = TCA
    pack = compute_pack(work_type, family, variant)
    # Over the desk's 160-word ceiling, and carrying every must_mention
    # anchor so the length rule is the only thing either side can object to.
    long_answer = (
        "Square-root impact scaling against ADV sets the cost, measured on "
        "the arrival versus decision benchmark. " * 14
    ).strip()
    assert any(
        "outside the analysis budget" in v
        for v in gate_violations(pack.to_dict(), long_answer, "analysis")
    )
    row = rowlib.student_row(
        row_id=row_id_from_coords("analysis", work_type, family, variant),
        kind="analysis",
        pack=pack.to_dict(),
        holdout=False,
        answer=long_answer,
        stamp={"pack_seed": f"{pack.seed:016x}"},
        teacher={"model": "test", "finish_reason": "stop", "usage": {}},
        render={"kind": "analysis", "attempts": 1},
        attempts=1,
        invented=[],
        missing=[],
        register_ok=True,
    )
    failures = _check_row(row, frozenset(), quick=True)
    assert any("outside the analysis budget" in p for p in failures["schema"])
