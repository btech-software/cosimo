"""valuation.equity.dcf -- FCFF enterprise-value fact computer.

Pure Python, no network, no clock: ``compute(family, variant)`` draws the
scenario from ``seed = pack_seed(work_type, family, variant)`` and returns a
:class:`FactPack`, or raises :class:`PackError` when the drawn parameters do
not describe a coherent DCF (``WACC <= g`` first of all -- the v2 generator
that shipped perpetuities with a negative denominator is exactly what this gate
replaces). Verification re-imports this module and recomputes from the seed,
so the file *is* the definition of its own ground truth.

The FCFF identity is the textbook one, with the sign v2's analysis file got
wrong: working capital is a *build*, so it is subtracted as an increase
``nwc_pct * (Rev_t - Rev_{t-1})``, not added as a flat share of revenue.
"""

from __future__ import annotations

import random

from ..seed import pack_seed, rng_for
from .base import (
    FactPack,
    PackError,
    assemble_contract,
    make_ticker,
    pick_as_of,
)
from .registers import pick_register

WORK_TYPE = "valuation.equity.dcf"
FAMILIES = ("mature_consumer", "cyclical_industrial", "fade_required")

# Per family: revenue (in $M), EBIT margin, capex as % of revenue, working
# capital as % of the revenue build, WACC, terminal growth, and the entity
# name pool the scenario is dressed in. The bands are the *scenario*, and the
# holdout family (fade_required) is drawn from the same structural space --
# held out by family, unseen by the student, never by luck of the seed.
#: Issuer names, composed rather than listed. Three per family was the old
#: pool, so twenty variants of a family were twenty write-ups of the same three
#: companies -- the numbers moved, the scenario did not, which is the stem
#: repaint v1 was built out of. Cross-multiplying a stem with a suffix widens
#: the pool without a wall of literals, and the draw count is unchanged (one
#: `rng.choice`, whatever the pool holds), so every figure these packs compute
#: is byte-identical to what it computed before.
_ISSUER_SUFFIXES = ("Group", "Holdings", "Co", "Industries", "Partners")


def _issuers(*stems: str) -> tuple[str, ...]:
    return tuple(f"{stem} {suffix}" for stem in stems for suffix in _ISSUER_SUFFIXES)


_BANDS = {
    "mature_consumer": {
        "rev_m": (400.0, 6000.0, 1),
        "ebit_margin": (0.12, 0.25, 4),
        "capex_pct": (0.04, 0.09, 4),
        "nwc_pct": (0.05, 0.20, 4),
        "wacc": (0.070, 0.115, 4),
        "g": (0.015, 0.030, 4),
        "names": _issuers(
            "Northwind Consumer",
            "Hearthstone Foods",
            "Copperfield Retail",
            "Amberlee Provisions",
            "Fairbanks Grocery",
            "Silverbrook Staples",
        ),
    },
    "cyclical_industrial": {
        "rev_m": (200.0, 4000.0, 1),
        "ebit_margin": (0.06, 0.16, 4),
        "capex_pct": (0.07, 0.15, 4),
        "nwc_pct": (0.08, 0.24, 4),
        "wacc": (0.085, 0.130, 4),
        "g": (0.020, 0.050, 4),
        "names": _issuers(
            "Ironvale Foundry",
            "Blackridge Machinery",
            "Kestrel Industrial",
            "Stonemarch Castings",
            "Redalloy Works",
            "Halbrook Engineering",
        ),
    },
    "fade_required": {
        # The margin that "looks perpetual" is the teaching point of the holdout
        # family: forbidden_claims exists so the read cannot just ratify it.
        "rev_m": (150.0, 2500.0, 1),
        "ebit_margin": (0.18, 0.32, 4),
        "capex_pct": (0.03, 0.07, 4),
        "nwc_pct": (0.04, 0.14, 4),
        "wacc": (0.075, 0.110, 4),
        "g": (0.025, 0.060, 4),
        "names": _issuers(
            "Meridian Luxe",
            "Aurelian Brands",
            "Gilded Arc",
            "Veranda Maison",
            "Lumiere Atelier",
            "Cassini Couture",
        ),
    },
}

_TAX_RATES = (0.19, 0.21, 0.24, 0.25)
_MIN_WACC_SPREAD = 0.005


def _draw(rng: random.Random, family: str, variant: int) -> dict:
    bands = _BANDS.get(family)
    if bands is None:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")
    params = {}
    for key in ("rev_m", "ebit_margin", "capex_pct", "nwc_pct", "wacc", "g"):
        lo, hi, dp = bands[key]
        params[key] = round(rng.uniform(lo, hi), dp)
    params["tax"] = rng.choice(list(_TAX_RATES))
    params["n"] = rng.randint(4, 8)
    # Indexed, not drawn -- see ``_issuers``. ``variant`` is threaded into the
    # draw for this one field only; every figure below still comes off ``rng``.
    issuers = bands["names"]
    params["name"] = issuers[variant % len(issuers)]
    return params


def _dcf(
    rev: float,
    ebit_margin: float,
    tax: float,
    capex_pct: float,
    nwc_pct: float,
    wacc: float,
    g: float,
    n: int,
) -> dict:
    """PV of the explicit FCFF strip and the terminal value, textbook form."""
    run = rev
    pv_explicit = 0.0
    fcff_last = 0.0
    for t in range(1, n + 1):
        nxt = run * (1.0 + g)
        fcff = nxt * (ebit_margin * (1.0 - tax) - capex_pct) - nwc_pct * run * g
        pv_explicit += fcff / (1.0 + wacc) ** t
        fcff_last = fcff
        run = nxt
    tv = fcff_last * (1.0 + g) / (wacc - g)
    pv_tv = tv / (1.0 + wacc) ** n
    return {
        "pv_explicit": pv_explicit,
        "tv": tv,
        "pv_tv": pv_tv,
        "ev": pv_explicit + pv_tv,
    }


def _build(
    work_type: str, family: str, variant: int, params: dict, rng: random.Random
) -> FactPack:
    rev = params["rev_m"]
    ebit = params["ebit_margin"]
    tax = params["tax"]
    capex_pct = params["capex_pct"]
    nwc_pct = params["nwc_pct"]
    wacc = params["wacc"]
    g = params["g"]
    n = params["n"]

    if wacc - g < _MIN_WACC_SPREAD:
        raise PackError(
            f"{work_type}/{family}: WACC {wacc} within {_MIN_WACC_SPREAD:.3f} of "
            f"g {g}: the perpetuity is not valued"
        )
    if g * 1.2 >= wacc - _MIN_WACC_SPREAD:
        raise PackError(
            f"{work_type}/{family}: the +20% terminal-growth sensitivity crosses "
            f"WACC ({g * 1.2:.4f} vs {wacc}): variant has no bounded read"
        )

    hi = _dcf(rev, ebit, tax, capex_pct, nwc_pct, wacc, g * 1.2, n)
    lo = _dcf(rev, ebit, tax, capex_pct, nwc_pct, wacc, g * 0.8, n)
    base = _dcf(rev, ebit, tax, capex_pct, nwc_pct, wacc, g, n)
    fcff_1 = rev * (1.0 + g) * (ebit * (1.0 - tax) - capex_pct) - nwc_pct * rev * g
    if fcff_1 <= 0 or base["ev"] <= 0:
        raise PackError(
            f"{work_type}/{family}: reinvestment exceeds NOPAT (FCFF1={fcff_1:.2f}):"
            " the scenario contradicts itself"
        )

    inputs = {
        "revenue_m": rev,
        "ebit_margin": ebit,
        "cash_tax_rate": tax,
        "capex_pct_revenue": capex_pct,
        "nwc_pct_revenue_build": nwc_pct,
        "wacc": wacc,
        "terminal_growth": g,
        "explicit_years": n,
    }
    computed = {
        "fcff_year1_m": round(fcff_1, 2),
        "pv_explicit_m": round(base["pv_explicit"], 2),
        "terminal_value_m": round(base["tv"], 2),
        "pv_terminal_m": round(base["pv_tv"], 2),
        "enterprise_value_m": round(base["ev"], 2),
        "ev_terminal_plus_20pct_growth_m": round(hi["ev"], 2),
        "ev_terminal_minus_20pct_growth_m": round(lo["ev"], 2),
    }
    ticker = make_ticker(rng)
    question = (
        f"Should we own {params['name']} ({ticker})? Revenue ${rev:,.0f}M, EBIT "
        f"margin {ebit * 100:.1f}%, cash tax {tax * 100:.0f}%, capex "
        f"{capex_pct * 100:.1f}% of revenue, working capital "
        f"{nwc_pct * 100:.1f}% of the revenue build. Over {n} years at "
        f"{wacc * 100:.2f}% WACC and {g * 100:.2f}% terminal growth: what "
        f"enterprise value does the FCFF model imply, and how does a +/- 20% "
        f"change in terminal growth move it?"
    )
    return FactPack(
        schema_version="v3.0",
        scenario_id=f"{work_type}.{family}",
        work_type=work_type,
        seed=pack_seed(work_type, family, variant),
        variant=variant,
        entities=[{"name": params["name"], "ticker": ticker, "currency": "USD"}],
        inputs=inputs,
        computed=computed,
        formulas=[
            "FCFF_t = EBIT_t (1 - t) - capex_t - NWC_t",
            "NWC_t = nwc% * (Rev_t - Rev_{t-1})",
            "TV = FCFF_n (1 + g) / (WACC - g)",
            "EV = sum PV(FCFF_1..n) + PV(TV)",
        ],
        # Every *printed* form still joins the allow-list next to its source
        # value -- "{:.1f}" renders 21.17 as 21.2, and a token the gate cannot
        # match is an invented number no matter which rounding produced it --
        # but it joins as a declared *alias* of the quantity it rounds. That is
        # the difference the amendment is after: the question may print 21.2,
        # the answer must claim 0.2117 (or its percent), and the gate can now
        # tell those two apart instead of admitting both as peers.
        **assemble_contract(
            inputs,
            computed,
            aliases={
                "revenue_m": [f"{round(rev)}"],
                "ebit_margin": [f"{ebit * 100:.1f}"],
                "cash_tax_rate": [f"{round(tax * 100)}"],
                "capex_pct_revenue": [f"{capex_pct * 100:.1f}"],
                "nwc_pct_revenue_build": [f"{nwc_pct * 100:.1f}"],
                "wacc": [f"{wacc * 100:.2f}"],
                "terminal_growth": [f"{g * 100:.2f}"],
            },
            display={"revenue_m": f"{rev:,.0f}"},
            # The sensitivity band is the question's own parameter rather than
            # a figure about the company -- it names how far terminal growth is
            # flexed -- but it is still a quantity an answer must be able to
            # state, so it is canonical under its own name.
            canonical_extra={"growth_sensitivity_pct": 20.0},
        ),
        forbidden_claims=[
            "point EV without the sensitivity",
            "perpetual margin uplift without fade",
        ],
        must_mention=[
            "terminal growth versus WACC",
            "reinvestment capex working-capital",
            "terminal-growth sensitivity",
        ],
        register=pick_register(work_type, family, rng),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    """Deterministically build the scenario; raises PackError, never guesses."""
    seed = pack_seed(WORK_TYPE, family, variant)
    rng = rng_for(seed)
    params = _draw(rng, family, variant)
    return _build(WORK_TYPE, family, variant, params, rng)


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
