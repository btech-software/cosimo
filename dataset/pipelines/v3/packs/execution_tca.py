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
from .base import (
    FactPack,
    PackError,
    assemble_contract,
    fold_conditionals,
    make_ticker,
    pick_as_of,
)
from .registers import pick_register

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

#: The issuer name, as two cycled axes (see the note where it is composed).
_HOUSES = ("North", "Vale", "Cobalt", "Reed", "Onyx", "Marrow", "Thorne")
_SECTORS = (
    "Materials",
    "Logistics",
    "Industrials",
    "Power",
    "Mining",
    "Freight",
    "Chemicals",
)

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

    # Indexed on two axes with different strides, so the twenty-five
    # combinations are walked in order rather than sampled with collisions.
    # Two `rng.choice` calls used to sit here and five houses times five
    # sectors gave about sixteen distinct names over twenty variants; this
    # gives twenty, and it costs the pack no randomness it needed elsewhere.
    house = _HOUSES[variant % len(_HOUSES)]
    sector = _SECTORS[(variant // len(_HOUSES)) % len(_SECTORS)]
    name = f"{house} {sector}"
    ticker = make_ticker(rng)
    question = (
        f"Work {shares:,} shares of {name} ({ticker}) {'buy' if side == 'buy' else 'sell'}; "
        f"ADV is {adv:,}, arrival price {arrival:,.2f}, quoted spread "
        f"{spread:.1f} bp, daily vol {sigma * 100:.2f}%. Under the square-root "
        f"model what implementation shortfall in bp of the arrival does this "
        f"cost, what average fill price does it imply, and against the "
        f"{_MAX_PARTICIPATION:.0%} single-schedule cap, is the impact or the "
        f"schedule the cost being managed here?"
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
        # The question prints the vol as a percent and the participation the
        # same way; both are roundings of a canonical fraction, so they are
        # declared as aliases rather than smuggled into the allow-list as if
        # they were quantities of their own. ``display`` carries the two
        # figures the desk writes with separators -- an answer that prints
        # `430567.0` for a share count is not wrong, it is unpublishable, and
        # the repair turn can now name the spelling it wants.
        **assemble_contract(
            inputs,
            computed,
            # One entry, because the question prints exactly one token that is
            # not already a canonical value: the daily vol as a percent. The
            # declaration authorises it (a pack whose question quotes a token
            # its allow-list does not carry is a corrupt pack) and the drift
            # axis then ignores it, `sigma * 100` being an exact spelling
            # rather than a rounding.
            #
            # `participation` is deliberately absent. The question never prints
            # it, so it is not a question rounding -- and declaring its percent
            # form here failed every correct answer that wrote "4.86% of ADV",
            # which is how a desk says it.
            aliases={"sigma_daily": [f"{sigma * 100:.2f}"]},
            display={
                "shares": f"{shares:,}",
                "adv_shares": f"{adv:,}",
                "arrival_price": f"{arrival:,.2f}",
                # The figure §A.3 names by hand, and the one a live row got
                # wrong again on the first render after the amendment: the
                # desk says "4.86% of ADV", never "0.048555". Declared as a
                # *display* rather than an alias, because an alias would make
                # the percent spelling a question rounding and fail the
                # answers that use it -- which is exactly what happened the
                # last time this was tried.
                "participation": f"{participation * 100:.2f}%",
                # A fill is a price and a price is quoted to the cent. The
                # computer keeps four decimals because the exam lane compares
                # against it; the desk writes 295.77, and the first v3.2
                # capture wrote 295.7747 in three rows for want of this line.
                "avg_fill_price": f"{avg_fill:,.2f}",
            },
            # The schedule cap the question asks the answer to measure against.
            canonical_extra={"participation_cap_pct": _MAX_PARTICIPATION * 100},
        ),
        # The question asks "at what participation does the schedule itself
        # become the risk?", and for every pack this computer can build the
        # honest answer is "not at this one" -- participation above the cap is
        # a PackError. Until §A.2 that was a sentence the answer was free to
        # assert anyway, and a live desk note asserted it twice, in both
        # directions, on a 4.86% clip. The inequality is in the pack, so the
        # point is a test of it: the slogan is *forbidden* below the cap and
        # *required* at it.
        **fold_conditionals(
            inputs,
            computed,
            forbidden_claims=[
                "the printable mid is achievable",
                "cost is linear in participation",
            ],
            must_mention=[
                # Anchors, not sentences: content words a correct answer must
                # use, matched on stems in any order (verification/prose.py).
                "square-root impact scaling",
                "participation against ADV",
                "arrival versus decision benchmark",
            ],
            conditional_mentions=[
                {
                    "when": f"participation >= {_MAX_PARTICIPATION}",
                    "mention": "schedule risk at the cap",
                }
            ],
            conditional_forbids=[
                {
                    "when": f"participation < {_MAX_PARTICIPATION}",
                    "claim": "the schedule itself becomes the risk",
                }
            ],
        ),
        # The order was worked against arrival, and nobody recorded when the
        # decision was taken or what the mid was then. A shortfall measured
        # against a decision price is a number this computer cannot produce.
        abstention_question=(
            f"What implementation shortfall did this {ticker} order run against "
            f"the *decision* price, and how much of it is timing delay rather "
            f"than market impact?"
        ),
        abstention_missing="decision price",
        register=pick_register(work_type, family, rng),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
