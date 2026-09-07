"""The DCF computer's arithmetic and its three PackError gates.

``test_answer_recomputes_independently`` is the load one: it re-derives the
enterprise value with a separate loop written beside -- not calling
``_dcf`` -- because a bug in the module's own helper cannot be caught by
re-running that helper. The verify gate recomputes through the module; this
test is the module's second opinion.
"""

import random

import pytest

from pipelines.v3.packs import valuation_fcff as m
from pipelines.v3.packs.base import PackError

GOOD = {
    "rev_m": 1000.0,
    "ebit_margin": 0.15,
    "capex_pct": 0.06,
    "nwc_pct": 0.10,
    "wacc": 0.09,
    "g": 0.02,
    "tax": 0.21,
    "n": 5,
    "name": "Northwind Consumer",
}


def _build(**over):
    params = dict(GOOD)
    params.update(over)
    return m._build(m.WORK_TYPE, "mature_consumer", 0, params, random.Random(42))


def test_builds_a_coherent_pack():
    pack = _build()
    assert pack.work_type == m.WORK_TYPE
    assert pack.computed["fcff_year1_m"] > 0
    assert pack.computed["enterprise_value_m"] > 0


def test_wacc_not_clear_of_terminal_growth_is_packerror():
    with pytest.raises(PackError, match="perpetuity"):
        _build(g=0.086)  # wacc - g = 0.004 < 0.005


def test_sensitivity_crossing_wacc_is_packerror():
    with pytest.raises(PackError, match="sensitivity crosses WACC"):
        _build(g=0.072)  # 1.2g = 0.0864 >= wacc - 0.005


def test_negative_first_fcff_is_packerror():
    with pytest.raises(PackError, match="reinvestment exceeds NOPAT"):
        _build(ebit_margin=0.05, capex_pct=0.10)


def test_answer_recomputes_independently():
    """A second, hand-written DCF must agree with the module's to the cent."""
    pack = _build()
    p = dict(GOOD)
    run, pv, fcff_last = p["rev_m"], 0.0, 0.0
    for t in range(1, p["n"] + 1):
        nxt = run * (1.0 + p["g"])
        fcff = (
            nxt * (p["ebit_margin"] * (1.0 - p["tax"]) - p["capex_pct"])
            - p["nwc_pct"] * run * p["g"]
        )
        pv += fcff / (1.0 + p["wacc"]) ** t
        fcff_last, run = fcff, nxt
    tv = fcff_last * (1.0 + p["g"]) / (p["wacc"] - p["g"])
    pv_tv = tv / (1.0 + p["wacc"]) ** p["n"]
    assert pack.computed["pv_explicit_m"] == pytest.approx(pv, abs=1e-2)
    assert pack.computed["terminal_value_m"] == pytest.approx(tv, abs=1e-2)
    assert pack.computed["enterprise_value_m"] == pytest.approx(pv + pv_tv, abs=1e-2)


def test_sensitivity_brackets_the_base_read():
    pack = _build()
    c = pack.computed
    assert (
        c["ev_terminal_minus_20pct_growth_m"]
        <= c["enterprise_value_m"]
        <= c["ev_terminal_plus_20pct_growth_m"]
    )


def test_unknown_family_is_packerror():
    with pytest.raises(PackError, match="unknown scenario family"):
        m.compute("no_such_family", 0)
