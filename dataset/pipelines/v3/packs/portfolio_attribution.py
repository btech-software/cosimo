"""portfolio.attribution.brinson_carino -- multiplicative sector attribution.

Carino, not plain Brinson: the arithmetic-consistency requirement is the point.
The additive pieces (allocation + selection + interaction) must *sum to the
active return*, and the Carino adjustment factor k is what makes them do so;
a completion that quietly drops k and prints additively consistent numbers is
using a formula the pack does not contain.

Guards: a book identical to its benchmark has nothing to attribute
(PackError -- the variant is skipped, not shipped with zero-filled effects),
and degenerate weight tables cannot be normalised.
"""

from __future__ import annotations

import math
import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_numbers, pick_as_of

WORK_TYPE = "portfolio.attribution.brinson_carino"
FAMILIES = ("global_equity_long", "us_small_cap", "em_multi_asset")

_SECTORS = {
    "global_equity_long": (
        "Financials",
        "Health Care",
        "Industrials",
        "Consumer Disc",
        "Tech",
    ),
    "us_small_cap": (
        "Health Care",
        "Capital Goods",
        "Materials",
        "Energy",
        "Comm Services",
    ),
    # Holdout family: currency + commodity exposure is where single-currency
    # Brinson arithmetic most often gets quietly mis-stated.
    "em_multi_asset": ("Local Equity", "Hard-Debt", "Commodities", "Local Rates"),
}
_BOOKS = {
    "global_equity_long": "Global Equity Long",
    "us_small_cap": "US Small Cap Core",
    "em_multi_asset": "EM Multi-Asset",
}


def _carino_factors(rp: float, rb: float) -> float:
    if abs(rp - rb) < 1e-12:
        raise PackError("portfolio equals the benchmark: nothing to attribute")
    if 1.0 + rp <= 0.0 or 1.0 + rb <= 0.0:
        raise PackError("logarithmic Carino factor is not defined for this period")
    return math.log((1.0 + rp) / (1.0 + rb)) / ((rp - rb) / rb)


def _build(work_type: str, family: str, variant: int, rng: random.Random) -> FactPack:
    sectors = _SECTORS.get(family)
    if sectors is None:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")
    n = rng.randint(3, len(sectors))
    chosen = rng.sample(list(sectors), n)

    def _weights() -> dict[str, float]:
        raw = [rng.uniform(0.05, 0.45) for _ in range(n)]
        total = sum(raw)
        if total <= 0:
            raise PackError(f"{work_type}/{family}: degenerate weight draw")
        return {s: w / total for s, w in zip(chosen, raw)}

    wb, wp = _weights(), _weights()
    rows = {}
    for s in chosen:
        rb_i = rng.uniform(-0.04, 0.06)
        rp_i = rb_i + rng.uniform(-0.025, 0.025)
        rows[s] = {"wb": wb[s], "wp": wp[s], "rb": round(rb_i, 5), "rp": round(rp_i, 5)}
    rb_total = sum(rows[s]["wb"] * rows[s]["rb"] for s in chosen)
    rp_total = sum(rows[s]["wp"] * rows[s]["rp"] for s in chosen)
    k = _carino_factors(rp_total, rb_total)

    active_bps = (rp_total - rb_total) * 1e4
    if abs(active_bps) < 2.0:
        raise PackError(
            f"{work_type}/{family}: active return {active_bps:.2f} bp is noise, "
            "the attribution has no story"
        )

    effects, alloc_sum, sel_sum, inter_sum = {}, 0.0, 0.0, 0.0
    for s in chosen:
        r = rows[s]
        alloc = (r["wp"] - r["wb"]) * (r["rb"] - rb_total) * k
        select = r["wp"] * (r["rp"] - r["rb"]) * k
        inter = (r["wp"] - r["wb"]) * (r["rp"] - r["rb"]) * k
        alloc_sum += alloc
        sel_sum += select
        inter_sum += inter
        effects[s] = {
            "allocation_bps": round(alloc * 1e4, 1),
            "selection_bps": round(select * 1e4, 1),
            "interaction_bps": round(inter * 1e4, 1),
        }

    inputs = {
        "sectors": chosen,
        "sector_rows": rows,
        "benchmark": "policy mix",
    }
    computed = {
        "portfolio_return": round(rp_total, 5),
        "benchmark_return": round(rb_total, 5),
        "active_bps": round(active_bps, 1),
        "carino_k": round(k, 6),
        "allocation_total_bps": round(alloc_sum * 1e4, 1),
        "selection_total_bps": round(sel_sum * 1e4, 1),
        "interaction_total_bps": round(inter_sum * 1e4, 1),
        **{
            f"effects_{s.replace(' ', '_').replace('-', '_')}": effects[s]
            for s in chosen
        },
    }
    book = _BOOKS[family]
    sector_list = ", ".join(chosen)
    direction = "ahead of" if active_bps >= 0 else "behind"
    question = (
        f"{book} returned {rp_total * 100:.2f}% against its benchmark's "
        f"{rb_total * 100:.2f}% over the period, {direction} the policy mix by "
        f"{abs(active_bps):.0f} bp. Decompose the active return by sector "
        f"({sector_list}) into Brinson-Carino allocation, selection and "
        f"interaction effects, show that the pieces sum to the active number via "
        f"the Carino factor, and name which single effect is worth acting on."
    )
    return FactPack(
        schema_version="v3.0",
        scenario_id=f"{work_type}.{family}",
        work_type=work_type,
        seed=pack_seed(work_type, family, variant),
        variant=variant,
        entities=[
            {"name": book, "book_id": book.lower().replace(" ", "_"), "currency": "USD"}
        ],
        inputs=inputs,
        computed=computed,
        formulas=[
            "k = ln((1+Rp)/(1+Rb)) / ((Rp - Rb)/Rb)",
            "alloc_i = (wp_i - wb_i)(rb_i - Rb) k",
            "sel_i = wp_i (rp_i - rb_i) k",
            "inter_i = (wp_i - wb_i)(rp_i - rb_i) k",
        ],
        allowed_numbers=assemble_numbers(
            inputs,
            computed,
            extra=(
                round(rp_total * 100, 2),
                round(rb_total * 100, 2),
                abs(round(active_bps)),
                float(abs(round(active_bps))),
                100.0,
            ),
        ),
        forbidden_claims=[
            "attribution explains the next period",
            "interaction effects can be dropped",
        ],
        must_mention=[
            "the Carino factor as the consistency bridge",
            "allocation vs selection split of the active return",
            "single-period attribution is not a skill signal",
        ],
        register="ic_memo",
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
