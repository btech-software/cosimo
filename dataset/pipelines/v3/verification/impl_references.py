"""The instruments the implementation slice asks the student to rebuild.

One static reference per work type, authored once and reviewed once: the
textbook form of the computer's own arithmetic, operation for operation, read
from ``inputs`` alone. Two rules hold this table accountable:

* an instrument never embeds the expected answer -- the tests pin the pack's
  figures, and a reference that carried them would make the suite vacuous
  (v2's self-confirming exam ceremony, analysis spec §2);
* each mirrors :mod:`pipelines.v3.packs` exactly -- same order of operations,
  same rounding, so a correct implementation reproduces ``computed`` to the
  last printed digit and the tolerance in :mod:`verification.implementation`
  has a meaning.

``solve(inputs) -> dict`` is the entrypoint the student learns. The guard the
dirty-data policy demands (§5.9): every reading arrives through ``_reading``,
which turns *any* unreadable value -- a missing one, a non-numeric string, a
non-finite one -- into ``ValueError``, so the instrument rejects dirt instead
of returning a plausible figure for an impossible input.
"""

from __future__ import annotations

#: Prepended to every reference: the coercion the policy is enforced by.
#: Written as source because the reference ships and executes as source.
_GUARD = """def _reading(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("unreadable reading") from None
    if not math.isfinite(number):
        raise ValueError("non-finite reading")
    return number
"""

_REFERENCE_DCF = (
    "import math\n\n"
    + _GUARD
    + """

def solve(inputs):
    rev = _reading(inputs["revenue_m"])
    ebit = _reading(inputs["ebit_margin"])
    tax = _reading(inputs["cash_tax_rate"])
    capex = _reading(inputs["capex_pct_revenue"])
    nwc = _reading(inputs["nwc_pct_revenue_build"])
    wacc = _reading(inputs["wacc"])
    growth = _reading(inputs["terminal_growth"])
    years = int(_reading(inputs["explicit_years"]))
    if rev <= 0 or years < 1 or wacc <= growth:
        raise ValueError("no bounded strip")
    run = rev
    pv = 0.0
    last = 0.0
    for year in range(1, years + 1):
        nxt = run * (1.0 + growth)
        fcff = nxt * ebit * (1.0 - tax) - capex * nxt - nwc * (nxt - run)
        pv += fcff / (1.0 + wacc) ** year
        last = fcff
        run = nxt
    tv = last * (1.0 + growth) / (wacc - growth)
    return {
        "enterprise_value_m": round(pv + tv / (1.0 + wacc) ** years, 2),
        "terminal_value_m": round(tv, 2),
    }
"""
)

_REFERENCE_VAR = (
    "import math\n\n"
    + _GUARD
    + """

def solve(inputs):
    book = _reading(inputs["book_value_m"])
    mu = _reading(inputs["mu_daily"])
    sigma = _reading(inputs["sigma_daily"])
    horizon = int(_reading(inputs["horizon_days"]))
    z95 = _reading(inputs["z95"])
    z99 = _reading(inputs["z99"])
    if book <= 0 or sigma <= 0 or horizon < 1:
        raise ValueError("degenerate book")
    scale = math.sqrt(horizon)
    return {
        "var95_h_m": round(book * (z95 * sigma - mu) * scale, 3),
        "var99_h_m": round(book * (z99 * sigma - mu) * scale, 3),
    }
"""
)

_REFERENCE_MULTIPLES = (
    "import math\n\n"
    + _GUARD
    + """

def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def solve(inputs):
    ebitda = _reading(inputs["ebitda_m"])
    net_debt = _reading(inputs["net_debt_m"])
    shares = _reading(inputs["shares_m"])
    rows = inputs["peers"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty peer set")
    evs = [_reading(row["ev_ebitda"]) for row in rows]
    if ebitda <= 0 or shares <= 0:
        raise ValueError("degenerate base")
    implied_ev = round(round(_median(evs), 2) * ebitda, 1)
    equity = round(implied_ev - net_debt, 1)
    if equity <= 0:
        raise ValueError("net debt swallows the enterprise")
    return {
        "implied_equity_m": equity,
        "implied_value_per_share": round(equity / shares, 2),
    }
"""
)

_REFERENCE_ATTRIBUTION = (
    "import math\n\n"
    + _GUARD
    + """

def solve(inputs):
    rows = inputs["sector_rows"]
    if not isinstance(rows, dict) or not rows:
        raise ValueError("empty attribution")
    cells = []
    for row in rows.values():
        cell = tuple(_reading(row[k]) for k in ("wb", "wp", "rb", "rp"))
        cells.append(cell)
    rb = sum(wb * r for wb, _, r, _ in cells)
    rp = sum(wp * p for _, wp, _, p in cells)
    if rb == 0.0 or rp == rb:
        raise ValueError("no active to attribute")
    k = math.log((1.0 + rp) / (1.0 + rb)) / ((rp - rb) / rb)
    alloc = sum((wp - wb) * (r - rb) * k for wb, wp, r, _ in cells)
    select = sum(wp * (p - r) * k for _, wp, r, p in cells)
    inter = sum((wp - wb) * (p - r) * k for wb, wp, r, p in cells)
    return {
        "active_bps": round((rp - rb) * 1e4, 1),
        "allocation_total_bps": round(alloc * 1e4, 1),
        "selection_total_bps": round(select * 1e4, 1),
        "interaction_total_bps": round(inter * 1e4, 1),
    }
"""
)

_REFERENCE_TCA = (
    "import math\n\n"
    + _GUARD
    + """

def solve(inputs):
    shares = _reading(inputs["shares"])
    adv = _reading(inputs["adv_shares"])
    arrival = _reading(inputs["arrival_price"])
    spread = _reading(inputs["quoted_spread_bps"])
    sigma = _reading(inputs["sigma_daily"])
    side = inputs["side"]
    if shares <= 0 or adv <= 0 or arrival <= 0 or sigma <= 0:
        raise ValueError("degenerate order")
    if side == "buy":
        signed = 1.0
    elif side == "sell":
        signed = -1.0
    else:
        raise ValueError("unknown side")
    cost = spread / 2.0 + sigma * math.sqrt(shares / adv) * 1e4
    return {
        "cost_bps_vs_arrival": round(cost, 2),
        "avg_fill_price": round(arrival * (1.0 + signed * cost / 1e4), 4),
        "dollar_cost": round(shares * arrival * cost / 1e4, 2),
    }
"""
)

#: The table of instruments. ``pins`` name the pack figures the suite asserts
#: (the ``public`` ones are shown to the student, the rest held back);
#: ``edges`` are the mis-set dials the spec demands the instrument reject --
#: (label, mutation applied to a copy of the inputs) -- each demanded as a
#: ``ValueError``. The wording columns compose the row's ``spec`` field, so
#: every row of a work type states its contract identically.
IMPL_SPECS: dict[str, dict] = {
    "valuation.equity.dcf": {
        "reference": _REFERENCE_DCF,
        "reads": (
            "revenue_m",
            "ebit_margin",
            "cash_tax_rate",
            "capex_pct_revenue",
            "nwc_pct_revenue_build",
            "wacc",
            "terminal_growth",
            "explicit_years",
        ),
        "ask": "Implement the FCFF enterprise-value instrument.",
        "measure": (
            "PV of the explicit FCFF strip plus the discounted terminal value; "
            "working capital is a charge on the revenue build, not on the level."
        ),
        "pins": ("enterprise_value_m", "terminal_value_m"),
        "public": ("enterprise_value_m",),
        "edges": (
            ("wacc at the terminal growth", {"wacc": 0.02, "terminal_growth": 0.02}),
            ("a missing reading", {"revenue_m": None}),
        ),
    },
    "risk.market.var_es": {
        "reference": _REFERENCE_VAR,
        "reads": (
            "book_value_m",
            "mu_daily",
            "sigma_daily",
            "horizon_days",
            "z95",
            "z99",
        ),
        "ask": "Implement the parametric VaR instrument.",
        "measure": (
            "Normal 95% and 99% quantiles of the book over the horizon, "
            "scaled by the square root of time."
        ),
        "pins": ("var95_h_m", "var99_h_m"),
        "public": ("var95_h_m",),
        "edges": (
            ("a zero-volatility book", {"sigma_daily": 0.0}),
            ("a non-finite mean", {"mu_daily": "NaN"}),
        ),
    },
    "valuation.equity.multiples": {
        "reference": _REFERENCE_MULTIPLES,
        "reads": ("ebitda_m", "net_debt_m", "shares_m", "peers"),
        "ask": "Implement the comparable-multiple valuation instrument.",
        "measure": (
            "The peer median EV/EBITDA applied to the subject, bridged to the "
            "equity through net debt and divided by the share count."
        ),
        "pins": ("implied_value_per_share", "implied_equity_m"),
        "public": ("implied_value_per_share",),
        "edges": (
            ("an empty peer set", {"peers": []}),
            ("a zero share count", {"shares_m": 0.0}),
        ),
    },
    "portfolio.attribution.brinson_carino": {
        "reference": _REFERENCE_ATTRIBUTION,
        "reads": ("sector_rows",),
        "ask": "Implement the Carino attribution instrument.",
        "measure": (
            "Allocation, selection and interaction in basis points, scaled by "
            "the Carino bridge so the effects sum to the active return."
        ),
        "pins": (
            "active_bps",
            "allocation_total_bps",
            "selection_total_bps",
            "interaction_total_bps",
        ),
        "public": ("active_bps",),
        "edges": (
            (
                "no active to attribute",
                {
                    "sector_rows": {
                        "solo": {"wb": 1.0, "wp": 1.0, "rb": 0.01, "rp": 0.01}
                    }
                },
            ),
            ("an empty sector table", {"sector_rows": {}}),
        ),
    },
    "execution.tca.arrival": {
        "reference": _REFERENCE_TCA,
        "reads": (
            "shares",
            "adv_shares",
            "arrival_price",
            "quoted_spread_bps",
            "sigma_daily",
            "side",
        ),
        "ask": "Implement the square-root implementation-shortfall instrument.",
        "measure": (
            "Half the quoted spread plus square-root impact against the "
            "arrival price, with the average fill and the dollar cost."
        ),
        "pins": ("cost_bps_vs_arrival", "avg_fill_price", "dollar_cost"),
        "public": ("cost_bps_vs_arrival",),
        "edges": (
            ("an order with no market", {"adv_shares": 0}),
            ("an unknown side", {"side": "sideways"}),
        ),
    },
}

__all__ = ["IMPL_SPECS"]
