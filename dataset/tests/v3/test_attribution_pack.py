"""The attribution pack adds up, and says so in figures an answer can quote.

The amendment names this file (§A.1) because the failure it guards is not a
prose failure. A pack whose allocation, selection and interaction effects miss
the active return by 371 bp still renders, still passes every numeric gate --
every figure in it came off the computer -- and still asks its question: "show
that the pieces sum to the active number". The teacher's only way out was to
invent a mechanism, and it did: a live memo explained that a negative Carino k
flips every sign. That row was board-green.

So the reconciliation is an assert in the computer and a test here, and the
prose gates are left to judge prose.
"""

from __future__ import annotations

import math

import pytest

from pipelines.v3.packs import PackError, compute_pack
from pipelines.v3.packs.portfolio_attribution import (
    ADDITIVITY_TOLERANCE_BPS,
    FAMILIES,
    WORK_TYPE,
    _carino_k,
)

#: Every variant the plan reserves for a family, walked. The bug this file
#: exists for was a property of the *arithmetic*, not of a seed, so a spot
#: check on one coordinate would have passed it just as happily.
VARIANTS = range(20)


def _packs():
    for family in FAMILIES:
        for variant in VARIANTS:
            try:
                yield compute_pack(WORK_TYPE, family, variant)
            except PackError:
                continue  # a guard's verdict, not a failure: see stage skips


def test_the_factor_is_the_textbook_one_not_the_benchmark_wearing_its_name():
    """``Rb * k_carino`` is what shipped, and it is what made 372.6 into 1.6."""
    rp, rb = 0.0331, -0.0042
    k = _carino_k(rp, rb)
    assert k == pytest.approx(
        (math.log(1 + rp) - math.log(1 + rb)) / (rp - rb), rel=1e-12
    )
    assert 0.9 < k < 1.1, "a period factor sits near one; it is not a return"


def test_a_degenerate_period_has_no_factor_and_no_pack():
    with pytest.raises(PackError, match="nothing to attribute"):
        _carino_k(0.02, 0.02)
    with pytest.raises(PackError, match="not defined"):
        _carino_k(-1.4, 0.01)


def test_every_pack_reconciles_to_its_active_return():
    """§A.1's assert, from the outside: printed pieces, printed active."""
    seen = 0
    for pack in _packs():
        seen += 1
        computed = pack.computed
        pieces = (
            computed["allocation_total_bps"]
            + computed["selection_total_bps"]
            + computed["interaction_total_bps"]
        )
        residual = pieces - computed["active_bps"]
        assert residual == pytest.approx(
            computed["reconciling_residual_bps"], abs=1e-9
        ), pack.scenario_id
        # The printed figures are each rounded to a tenth, so the sum of three
        # of them may sit a tenth from the printed active. The *arithmetic*
        # residual is nine orders of magnitude smaller and is what the
        # computer asserts.
        assert abs(residual) <= 0.2, f"{pack.scenario_id}/{pack.variant}: {residual}"
    assert seen >= 50, "the walk stopped finding packs; the guard is too tight"


def test_the_sector_rows_add_to_the_totals_they_are_summarised_by():
    """Per-sector effects and their totals are one decomposition, not two."""
    for pack in _packs():
        computed = pack.computed
        for leg, total_key in (
            ("allocation_bps", "allocation_total_bps"),
            ("selection_bps", "selection_total_bps"),
            ("interaction_bps", "interaction_total_bps"),
        ):
            legs = sum(
                value[leg]
                for key, value in computed.items()
                if key.startswith("effects_")
            )
            # Five sector rows each rounded to a tenth, and a total rounded
            # the same way: three tenths is the arithmetic worst case, and
            # anything past it is a decomposition disagreeing with itself.
            assert legs == pytest.approx(computed[total_key], abs=0.3), (
                f"{pack.scenario_id}/{pack.variant}: {leg}"
            )


def test_the_two_active_returns_are_both_named():
    """A reader who subtracts the question's two percentages is not inventing.

    ``active_bps`` is the linked figure the k-scaled pieces add to; the plain
    difference of the printed returns is beside it under its own name, so an
    answer that quotes either is quoting the pack.
    """
    for pack in _packs():
        computed = pack.computed
        linked = computed["active_bps"]
        arithmetic = computed["arithmetic_active_bps"]
        assert arithmetic == pytest.approx(
            (computed["portfolio_return"] - computed["benchmark_return"]) * 1e4, abs=0.6
        )
        assert linked == pytest.approx(arithmetic * computed["carino_k"], abs=0.6)
        assert {linked, arithmetic} <= set(pack.allowed_numbers)


def test_the_pack_no_longer_demands_a_slogan():
    """§A.1: one anchor, and none that a non-reconciling pack would force."""
    for pack in _packs():
        assert pack.must_mention == ["allocation versus selection"]


def test_the_tolerance_is_the_amendments_and_bites_on_the_old_arithmetic():
    """A pack built the old way would have failed by three orders of magnitude."""
    assert ADDITIVITY_TOLERANCE_BPS == 0.15
    pack = compute_pack(WORK_TYPE, "global_equity_long", 0)
    old_style = pack.computed["active_bps"] * pack.computed["benchmark_return"]
    assert abs(old_style - pack.computed["active_bps"]) > ADDITIVITY_TOLERANCE_BPS
