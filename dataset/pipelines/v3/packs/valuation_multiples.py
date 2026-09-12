"""valuation.equity.multiples -- relative-valuation fact computer.

The read this type must teach is that a multiple is only as good as its peer
set: the pack fixes the peer table, the medians, and the EV-to-equity bridge,
and forbids the completion from treating a single multiple as evidence. Median,
never mean: one levered comp in the mean is the classic wrong number.
"""

from __future__ import annotations

import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_contract, make_ticker, pick_as_of
from .registers import pick_register

WORK_TYPE = "valuation.equity.multiples"
FAMILIES = ("specialty_retail", "enterprise_software", "consumer_platforms")

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
    "specialty_retail": {
        "ev_ebitda": (5.5, 9.5, 2),
        "p_e": (9.0, 16.0, 1),
        "ebitda_m": (80.0, 900.0, 1),
        "margin": (0.06, 0.13, 4),
        "names": _issuers(
            "Harbor & Main Retail",
            "Crestline Outfitters",
            "Vantage Home",
            "Bramblewood Supply",
            "Tidewater Outfitters",
            "Calderwood Stores",
        ),
    },
    "enterprise_software": {
        "ev_ebitda": (14.0, 26.0, 2),
        "p_e": (22.0, 48.0, 1),
        "ebitda_m": (60.0, 800.0, 1),
        "margin": (0.14, 0.32, 4),
        "names": _issuers(
            "Latticesoft Systems",
            "Nimbus Enterprise Cloud",
            "Ledgerline Software",
            "Quarrystone Data",
            "Fenwick Platform",
            "Arborline Analytics",
        ),
    },
    "consumer_platforms": {
        # Holdout family: marketplace economics that punish the retail-multiple
        # reflex -- network effects make the dispersion, not the sector label.
        "ev_ebitda": (10.0, 22.0, 2),
        "p_e": (17.0, 36.0, 1),
        "ebitda_m": (40.0, 500.0, 1),
        "margin": (0.09, 0.22, 4),
        "names": _issuers(
            "Bazaarly Marketplace",
            "Trellis Commerce",
            "Portico Network",
            "Wayfarer Exchange",
            "Junction Social",
            "Beacon Marketplace",
        ),
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

    # Indexed, not drawn -- see ``_issuers``.
    issuers = bands["names"]
    name = issuers[variant % len(issuers)]
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
        # The peer multiples are quantities in their own right (they live under
        # `inputs.peers.*`, so `assemble_contract` names each one); what is
        # declared here is only the *roundings the question prints*. The two
        # medians are computed figures and need no alias -- they are already
        # canonical.
        **assemble_contract(
            inputs,
            computed,
            aliases={
                "ebitda_m": [f"{ebitda_m:.1f}"],
                "ebit_margin": [f"{margin * 100:.1f}"],
                "net_debt_m": [f"{net_debt_m:.1f}"],
                "shares_m": [f"{shares_m:.1f}"],
                "net_income_m": [f"{net_income_m:.2f}"],
            },
            display={
                "net_debt_m": f"{net_debt_m:,.1f}",
                "shares_m": f"{shares_m:,.1f}",
                "net_income_m": f"{net_income_m:,.2f}",
            },
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
        # The peer set implies a value per share. What the market is paying
        # for it today is not in the pack, so "cheap or expensive" cannot be
        # answered here however precisely the implied figure is computed.
        abstention_question=(
            f"Is {name} trading cheap or expensive against this peer set today, "
            f"and what does the market price imply about the multiple?"
        ),
        abstention_missing="market price",
        register=pick_register(work_type, family, rng),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
