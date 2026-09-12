"""§A.2: a point whose truth is an inequality is a test, not a slogan.

``must_mention`` used to be a list the answer had to enact whatever the pack's
own numbers said. So a TCA pack whose participation is half its cap still
produced answers declaring that the schedule itself becomes the risk -- the
gate demanded the concept, the teacher supplied it, and the row shipped
board-green contradicting its own arithmetic.

The declarations live on the pack, are evaluated when the pack is built, and
reach the teacher only as points its own numbers make true.
"""

from __future__ import annotations

import pytest

from pipelines.v3.packs import compute_pack
from pipelines.v3.packs.base import PackError, fold_conditionals, holds

TCA = ("execution.tca.arrival", "large_cap_intraday", 0)
VAR = ("risk.market.var_es", "rates_book", 0)


def test_a_condition_reads_this_packs_own_quantities():
    quantities = {"participation": 0.048, "mu_daily": -0.0001}
    assert holds("participation < 0.1", quantities)
    assert not holds("participation >= 0.1", quantities)
    assert holds("mu_daily < 0", quantities)
    assert holds("participation != mu_daily", quantities)


def test_a_condition_nobody_can_read_is_a_pack_error_not_a_silent_false():
    """A forbid that quietly evaporates is worse than no forbid at all."""
    with pytest.raises(PackError, match="neither a quantity"):
        holds("particpation < 0.1", {"participation": 0.048})
    with pytest.raises(PackError, match="is not"):
        holds("participation is small", {"participation": 0.048})


def test_folding_appends_only_what_the_numbers_make_true():
    folded = fold_conditionals(
        {"participation": 0.048},
        {},
        must_mention=["square-root impact scaling"],
        forbidden_claims=["the printable mid is achievable"],
        conditional_mentions=[
            {"when": "participation >= 0.1", "mention": "schedule risk at the cap"}
        ],
        conditional_forbids=[
            {
                "when": "participation < 0.1",
                "claim": "the schedule itself becomes the risk",
            }
        ],
    )
    assert folded["must_mention"] == ["square-root impact scaling"]
    assert folded["forbidden_claims"] == [
        "the printable mid is achievable",
        "the schedule itself becomes the risk",
    ]
    # The declarations survive the fold: a row audited months later should be
    # able to say why it was told what it was told.
    assert folded["conditional_forbids"][0]["when"] == "participation < 0.1"


def test_a_tca_pack_under_its_cap_forbids_the_slogan_and_does_not_require_it():
    pack = compute_pack(*TCA)
    assert pack.computed["participation"] < 0.10
    assert "the schedule itself becomes the risk" in pack.forbidden_claims
    assert "schedule risk at the cap" not in pack.must_mention
    # The question no longer solicits it either: it asks which of the two costs
    # is being managed, which is a question this pack can answer.
    assert "does the schedule itself become" not in pack.question


def test_every_tca_pack_carries_the_conditional_because_every_one_is_under_the_cap():
    """Participation above the cap is a PackError, so the forbid is universal.

    Worth asserting rather than assuming: if the cap ever moves, this says out
    loud that the corpus contains no row where the schedule *is* the risk.
    """
    seen = 0
    for family in ("large_cap_intraday", "mid_cap_swing", "etf_rebalance"):
        for variant in range(8):
            try:
                pack = compute_pack("execution.tca.arrival", family, variant)
            except PackError:
                continue
            seen += 1
            assert "the schedule itself becomes the risk" in pack.forbidden_claims
    assert seen >= 12


def test_a_var_pack_with_a_negative_mean_forbids_both_readings_of_the_sign():
    pack = compute_pack(*VAR)
    assert pack.inputs["mu_daily"] < 0
    assert {"positive drift", "expected gain"} <= set(pack.forbidden_claims)


def test_a_var_pack_with_a_positive_mean_forbids_neither():
    positive = None
    for family in ("rates_book", "equity_longonly", "credit_focused"):
        for variant in range(20):
            try:
                pack = compute_pack("risk.market.var_es", family, variant)
            except PackError:
                continue
            if pack.inputs["mu_daily"] > 0:
                positive = pack
                break
        if positive:
            break
    assert positive is not None, "no positive-drift book in the plan's variants"
    assert "positive drift" not in positive.forbidden_claims
