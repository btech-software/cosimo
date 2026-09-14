"""Named constants of the discipline: the one thing an answer may know.

Every other number rule in this package answers "did this figure come from
this scenario", and the answer has to stay absolute -- an invented-number rate
of zero is what v3 is for. But a corpus where *every* number must come from the
pack can only teach arithmetic and refusal. It cannot teach the sentence a desk
actually wants, which places the figure against what the field expects:

    0.43 is weak against the ~1 a diversified book targets

Before ``FactPack.conventions`` that half-sentence was unwritable, and worse,
unpredictably so: a teacher writing "Basel multipliers of 3 to 4 apply" was
dead-lettered, while one writing "a Sharpe above 1.5 is strong" survived --
not by rule, but because 1.5 happened to round near a figure that pack held.

So conventions are declared, per work type, in advance. They are legal in that
work type's rows and nowhere else, they are stated in the brief, and they may
never be restated as facts about the entity.
"""

from __future__ import annotations

import pytest

from pipelines.v3.packs import compute_pack
from pipelines.v3.packs.base import convention_numbers
from pipelines.v3.teacher.prompts import conventions_policy, render_brief
from pipelines.v3.verification.prose import (
    canonical_numbers,
    gate_violations,
    gradeable_numbers,
)

VAR = ("risk.market.var_es", "rates_book", 0)
ATTRIBUTION = ("portfolio.attribution.brinson_carino", "global_equity_long", 0)
DCF = ("valuation.equity.dcf", "mature_consumer", 0)
TCA = ("execution.tca.arrival", "large_cap_intraday", 0)


def _pack(coords) -> dict:
    return compute_pack(*coords).to_dict()


def test_the_work_types_that_declare_conventions_declare_named_ones():
    """Constants of the field, with the sentence that makes each usable."""
    for coords, expected in (
        (VAR, {3.0, 4.0, 0.5, 1.0, 2.0}),
        (ATTRIBUTION, {0.5, 0.75, 1.0}),
        (DCF, {2.5, 3.0}),
    ):
        pack = _pack(coords)
        assert set(convention_numbers(pack)) == expected, pack["work_type"]
        for entry in pack["conventions"]:
            assert entry["says"].strip(), pack["work_type"]
            assert entry["numbers"], pack["work_type"]


def test_a_work_type_with_nothing_to_declare_declares_nothing():
    """The field is opt-in. TCA's own cap is a figure of the scenario, so it
    belongs in `canonical` where it already is, not here."""
    pack = _pack(TCA)
    assert pack["conventions"] == []
    assert convention_numbers(pack) == []
    assert conventions_policy(pack) == ""


def test_conventions_widen_what_may_be_quoted_and_nothing_else():
    """`canonical` does not move by a digit; the conventions sit beside it."""
    pack = _pack(VAR)
    canonical = set(canonical_numbers(pack))
    gradeable = set(gradeable_numbers(pack))
    assert canonical <= gradeable
    assert gradeable - canonical == set(convention_numbers(pack)) - canonical


def test_a_judgement_against_the_standard_passes_the_gate():
    """The sentence the corpus exists to make trainable."""
    pack = _pack(VAR)
    computed = pack["computed"]
    answer = (
        f"The 1-day 95% VaR is {computed['var95_1d_m']} and the 95% expected "
        f"shortfall is {computed['es95_1d_m']}, on the normality assumption and "
        "square-root horizon scaling that hide the expected shortfall tail. A "
        "Basel multiplier starting at 3 and rising toward 4 would size the "
        "capital against that figure. On the Sharpe scale a diversified book "
        "targets about 1, so a book nearer 0.5 is not being paid for its risk. "
        "The negative daily mean adds to the expected loss rather than "
        "offsetting it, and the limit stands on that assumption."
    )
    assert gate_violations(pack, answer, "analysis") == []


def test_a_convention_is_legal_in_its_own_work_type_and_nowhere_else():
    """The scoping that keeps this from being a global hole in the gate.

    Tested with 0.75 -- attribution's information-ratio band -- rather than
    with the Basel multiplier, and the reason is worth writing down: bare
    integers 0-10 are in ``config.NUMBER_WHITELIST`` as arithmetic furniture, so
    "a multiplier of 4" was always legal in any row and still is. The scoping
    this asserts therefore covers every convention value that is *not* a small
    whole number, which is where the risk of laundering a figure actually
    lives.
    """
    borrowed = (
        "Shortfall is 28.16 bp against arrival with 26.66 bp of square-root "
        "impact scaling and participation against ADV of 4.86%, on the arrival "
        "versus decision benchmark. An information ratio near 0.75 is very good "
        "and this desk is not there."
    )
    violations = gate_violations(_pack(TCA), borrowed, "analysis")
    assert any("invented numbers" in v and "0.75" in v for v in violations)
    # ...and the work type that declared it may say exactly the same thing.
    attribution = _pack(ATTRIBUTION)
    assert 0.75 in gradeable_numbers(attribution)
    assert 0.75 not in canonical_numbers(attribution)


def test_the_brief_states_the_conventions_and_asks_for_the_judgement():
    """A gate the writer cannot see is a trap -- and this one is an invitation,
    not a fence, so it has to be read to be acted on."""
    pack = _pack(DCF)
    policy = conventions_policy(pack)
    assert "nominal GDP" in policy
    assert "judging it" in policy
    user = render_brief(pack, kind="analysis")[-1]["content"]
    assert policy in user


@pytest.mark.parametrize("coords", [VAR, ATTRIBUTION, DCF])
def test_a_convention_may_not_be_spelled_in_a_way_the_format_gate_refuses(coords):
    """The brief quotes these strings; a teacher that copies one must not fail.

    `integer_format_offenders` refuses a whole number with a decimal tail, so a
    convention that said "about 1.0" would be the contract instructing a
    violation of itself.
    """
    pack = _pack(coords)
    for entry in pack["conventions"]:
        from pipelines.v3.verification.prose import integer_format_offenders

        assert integer_format_offenders(pack, entry["says"]) == [], entry["says"]
