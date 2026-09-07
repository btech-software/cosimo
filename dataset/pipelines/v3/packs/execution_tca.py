"""execution.tca.arrival -- transaction-cost-analysis fact computer.

The square-root law is the reference model:

    impact_bps = sigma_daily * sqrt(participation) * 1e4
    cost_bps   = spread_bps / 2 + impact_bps          (paid vs the arrival mid)

with the sign of the shortfall set by the side. It is the *defensible* naive
model -- linear cost, permanent + temporary split, ADV as the volume anchor --
and the pack's forbidden_claims exist so the completion cannot ratify its two
known lies (that the printable mid is achievable, that cost is linear in
participation).

Guards: an order that is not executable inside the participation bands
(``participation > 10% of ADV``) raises PackError -- a scenario whose premise
is impossible is not an aggressive fill, it is a data error.
"""

from __future__ import annotations

import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_numbers, make_ticker, pick_as_of

WORK_TYPE = "execution.tca.arrival"
FAMILIES = ("large_cap_intraday", "mid_cap_swing", "etf_rebalance")

_BANDS = {
    "large_cap_intraday": {
        "adv": (5_000_000, 20_000_000),
        "shares": (50_000, 800_000),
        "price": (40.0, 480.0, 2),
        "spread_bps": (1.0, 4.0, 1),
        "sigma_daily": (0.011, 0.020, 4),
    },
    "mid_cap_swing": {
        "adv": (800_000, 6_000_000),
        "shares": (30_000, 350_000),
        "price": (18.0, 160.0, 2),
        "spread_bps": (4.0, 15.0, 1),
        "sigma_daily": (0.016, 0.030, 4),
    },
    # Holdout family: the index event whose prints look like the large names
    # but trade like the small ones -- participation is the binding constraint.
    "etf_rebalance": {
        "adv": (300_000, 3_000_000),
        "shares": (60_000, 900_000),
        "price": (10.0, 120.0, 2),
        "spread_bps": (6.0, 22.0, 1),
        "sigma_daily": (0.013, 0.028, 4),
    },
}

_MAX_PARTICIPATION = 0.10


def _build(work_type: str, family: str, variant: int, rng: random.Random) -> FactPack:
    bands = _BANDS.get(family)
    if bands is None:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")

    adv = rng.randint(*bands["adv"])
    shares = rng.randint(*bands["shares"])
    arrival = round(rng.uniform(*bands["price"][:2]), bands["price"][2])
    spread = round(rng.uniform(*bands["spread_bps"][:2]), bands["spread_bps"][2])
    sigma = round(rng.uniform(*bands["sigma_daily"][:2]), bands["sigma_daily"][2])
    side = rng.choice(("buy", "sell"))
    if adv <= 0 or shares <= 0 or arrival <= 0 or sigma <= 0:
        raise PackError(f"{work_type}/{family}: degenerate TCA inputs")

    participation = shares / adv
    if participation > _MAX_PARTICIPATION:
        raise PackError(
            f"{work_type}/{family}: {shares} shares vs ADV {adv} is "
            f"{participation:.1%} participation -- above the {_MAX_PARTICIPATION:.0%} "
            "single-schedule cap: the premise is impossible"
        )

    impact = sigma * (participation**0.5) * 1e4
    cost_bps = spread / 2.0 + impact
    signed = 1.0 if side == "buy" else -1.0
    avg_fill = round(arrival * (1.0 + signed * cost_bps / 1e4), 4)
    dollar_cost = shares * arrival * cost_bps / 1e4

    name = (
        f"{rng.choice(('North', 'Vale', 'Cobalt', 'Reed', 'Onyx'))} "
        f"{rng.choice(('Materials', 'Logistics', 'Industrials', 'Power', 'Mining'))}"
    )
    ticker = make_ticker(rng)
    question = (
        f"Work {shares:,} shares of {name} ({ticker}) {'buy' if side == 'buy' else 'sell'}; "
        f"ADV is {adv:,}, arrival price {arrival:,.2f}, quoted spread "
        f"{spread:.1f} bp, daily vol {sigma * 100:.2f}%. Under the square-root "
        f"model what implementation shortfall in bp of the arrival does this "
        f"cost, what average fill price does it imply, and at what participation "
        f"does the schedule itself become the risk?"
    )
    inputs = {
        "symbol": ticker,
        "side": side,
        "shares": float(shares),
        "adv_shares": float(adv),
        "arrival_price": arrival,
        "quoted_spread_bps": spread,
        "sigma_daily": sigma,
    }
    computed = {
        "participation": round(participation, 6),
        "impact_bps": round(impact, 2),
        "spread_cost_bps": round(spread / 2.0, 2),
        "cost_bps_vs_arrival": round(cost_bps, 2),
        "avg_fill_price": avg_fill,
        "dollar_cost": round(dollar_cost, 2),
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
            "impact_bps = sigma_daily * sqrt(shares / ADV) * 1e4",
            "cost_bps = spread_bps / 2 + impact_bps",
            "avg_fill = arrival * (1 +/- cost_bps / 1e4)",
        ],
        allowed_numbers=assemble_numbers(
            inputs,
            computed,
            extra=(
                float(shares),
                float(adv),
                arrival,
                spread,
                round(sigma * 100, 2),
                round(participation * 100, 2),
                10.0,
            ),
        ),
        forbidden_claims=[
            "the printable mid is achievable",
            "cost is linear in participation",
        ],
        must_mention=[
            "the square-root impact scaling",
            "participation near the ADV cap as the schedule constraint",
            "arrival versus decision benchmark choice",
        ],
        register="desk_chat",
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
