"""valuation.equity.multiples -- relative-valuation fact computer.

The read this type must teach is that a multiple is only as good as its peer
set: the pack fixes the peer table, the medians, and the EV-to-equity bridge,
and forbids the completion from treating a single multiple as evidence. Median,
never mean: one levered comp in the mean is the classic wrong number.
"""

from __future__ import annotations

import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_numbers, make_ticker, pick_as_of

WORK_TYPE = "valuation.equity.multiples"
FAMILIES = ("specialty_retail", "enterprise_software", "consumer_platforms")

_BANDS = {
    "specialty_retail": {
        "ev_ebitda": (5.5, 9.5, 2),
        "p_e": (9.0, 16.0, 1),
        "ebitda_m": (80.0, 900.0, 1),
        "margin": (0.06, 0.13, 4),
        "names": ("Harbor & Main Retail", "Crestline Outfitters", "Vantage Home Group"),
    },
    "enterprise_software": {
        "ev_ebitda": (14.0, 26.0, 2),
        "p_e": (22.0, 48.0, 1),
        "ebitda_m": (60.0, 800.0, 1),
        "margin": (0.14, 0.32, 4),
        "names": (
            "Latticesoft Systems",
            "Nimbus Enterprise Cloud",
            "Ledgerline Software",
        ),
    },
    "consumer_platforms": {
        # Holdout family: marketplace economics that punish the retail-multiple
        # reflex -- network effects make the dispersion, not the sector label.
        "ev_ebitda": (10.0, 22.0, 2),
        "p_e": (17.0, 36.0, 1),
        "ebitda_m": (40.0, 500.0, 1),
        "margin": (0.09, 0.22, 4),
        "names": ("Bazaarly Marketplace", "Trellis Commerce", "Portico Network Co"),
    },
}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _draw_peers(rng: random.Random, bands: dict) -> list[dict]:
    n = rng.randint(3, 5)
    peers = []
    for i in range(n):
        peers.append(
            {
                "name": f"Peer {chr(65 + i)}",
                "ev_ebitda": round(
                    rng.uniform(*bands["ev_ebitda"][:2]), bands["ev_ebitda"][2]
                ),
                "p_e": round(rng.uniform(*bands["p_e"][:2]), bands["p_e"][2]),
            }
        )
    return peers


def _build(work_type: str, family: str, variant: int, rng: random.Random) -> FactPack:
    bands = _BANDS.get(family)
    if bands is None:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")

    ebitda_m = round(rng.uniform(*bands["ebitda_m"][:2]), 1)
    margin = round(rng.uniform(*bands["margin"][:2]), bands["margin"][2])
    revenue_m = round(ebitda_m / margin, 1)
    net_income_m = round(ebitda_m * rng.uniform(0.45, 0.85), 2)
    net_debt_m = round(ebitda_m * rng.uniform(-0.30, 0.90), 1)
    shares_m = round(rng.uniform(25.0, 250.0), 1)
    if net_income_m <= 0 or shares_m <= 0:
        raise PackError(f"{work_type}/{family}: degenerate share or income base")

    peers = _draw_peers(rng, bands)
    med_ev = round(_median([p["ev_ebitda"] for p in peers]), 2)
    med_pe = round(_median([p["p_e"] for p in peers]), 2)
    if med_ev <= 0:
        raise PackError(f"{work_type}/{family}: peer set has no positive EV/EBITDA")

    implied_ev_m = round(med_ev * ebitda_m, 1)
    implied_equity_m = round(implied_ev_m - net_debt_m, 1)
    implied_pps = round(implied_equity_m / shares_m, 2)
    if implied_equity_m <= 0:
        raise PackError(f"{work_type}/{family}: net debt swallows the enterprise value")

    name = rng.choice(list(bands["names"]))
    ticker = make_ticker(rng)
    peer_ev_list = ", ".join(f"{p['ev_ebitda']:.2f}" for p in peers)
    question = (
        f"Peer-market {name} ({ticker}): EBITDA ${ebitda_m:,.1f}M on a "
        f"{margin * 100:.1f}% margin, net debt ${net_debt_m:,.1f}M, "
        f"{shares_m:,.1f}M shares, net income ${net_income_m:,.2f}M. The peer "
        f"EV/EBITDA set is {peer_ev_list}. What equity value per share does the "
        f"peer median support, and what in the dispersion of that peer set "
        f"undermines the read?"
    )
    inputs = {
        "ebitda_m": ebitda_m,
        "revenue_m": revenue_m,
        "ebit_margin": margin,
        "net_income_m": net_income_m,
        "net_debt_m": net_debt_m,
        "shares_m": shares_m,
        "peers": peers,
    }
    computed = {
        "median_ev_ebitda": med_ev,
        "median_p_e": med_pe,
        "implied_ev_m": implied_ev_m,
        "implied_equity_m": implied_equity_m,
        "implied_value_per_share": implied_pps,
    }
    return FactPack(
        schema_version="v3.0",
        scenario_id=f"{work_type}.{family}",
        work_type=work_type,
        seed=pack_seed(work_type, family, variant),
        variant=variant,
        entities=[{"name": name, "ticker": ticker, "currency": "USD"}],
        inputs=inputs,
        computed=computed,
        formulas=[
            "Implied EV = median(EV/EBITDA) * EBITDA",
            "Implied equity = Implied EV - net debt",
            "Value per share = Implied equity / shares",
        ],
        allowed_numbers=assemble_numbers(
            inputs,
            computed,
            extra=(
                round(ebitda_m, 1),
                margin * 100,
                round(margin * 100, 1),
                net_debt_m,
                shares_m,
                net_income_m,
                round(net_debt_m),
                round(shares_m, 1),
                round(net_income_m, 2),
                float(med_ev),
                float(med_pe),
                100.0,
                *[float(p["ev_ebitda"]) for p in peers],
            ),
        ),
        forbidden_claims=[
            "a single multiple as sufficient evidence",
            "the mean is the robust central peer",
        ],
        must_mention=[
            "peer dispersion",
            "equity net debt",
            "median versus mean",
        ],
        register=rng.choice(("desk_chat", "ic_memo")),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
