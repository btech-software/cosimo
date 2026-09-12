"""portfolio.attribution.brinson_carino -- multiplicative sector attribution.

Carino, not plain Brinson: the arithmetic-consistency requirement is the point.
The additive pieces (allocation + selection + interaction) must *sum to the
active return*, and :data:`carino_k` is what carries them from an arithmetic
decomposition onto the linked scale the book is actually measured on.

Two bugs lived here until the v3.2 amendment, and they compounded:

1. the factor was ``ln((1+Rp)/(1+Rb)) / ((Rp - Rb)/Rb)`` -- that is
   ``Rb * k_carino``, so on a benchmark that returned -0.42% every effect came
   out about four thousandths of its size and five sector pieces summed to
   1.6 bp against a 372.6 bp active;
2. selection was taken at the *portfolio* weight while a separate interaction
   term was also reported, which double-counts ``(wp - wb)(rp - rb)`` and
   cannot add up whatever k is.

Both are fixed here, and the additivity is now an *assert* rather than a
claim in a docstring: a pack whose pieces do not reconcile raises
:class:`PackError` instead of shipping a scenario whose question ("show that
the pieces sum to the active number") the facts cannot answer. That mattered
more than it sounds -- a teacher told not to invent numbers and to engage
"Carino factor" wrote a theory about negative k flipping signs, because
inventing a mechanism was the only way left to save the pack.

``active_bps`` is therefore the *linked* active return,
``ln((1+Rp)/(1+Rb))``, which is the quantity the Carino-scaled effects add to.
The arithmetic difference a reader gets by subtracting the two printed returns
is beside it as ``arithmetic_active_bps``: it is what the question's own
percentages imply, so the pack must authorise it rather than make a correct
subtraction look invented.

Guards: a book identical to its benchmark has nothing to attribute
(PackError -- the variant is skipped, not shipped with zero-filled effects),
and degenerate weight tables cannot be normalised.
"""

from __future__ import annotations

import math
import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_contract, pick_as_of
from .registers import pick_register

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
#: The book each family's scenario is attributed for, as a *pool* rather than a
#: name. It used to be one string per family, so twenty variants of
#: ``us_small_cap`` were twenty attributions of "US Small Cap Core" -- the same
#: book, the same three sectors, different arithmetic. That is v1's stem repaint
#: at a smaller constant: the numbers vary and the scenario does not, and a
#: student reading twenty of them learns the mandate, not the method.
#:
#: Composed from parts so the pool is wide without being a wall of literals,
#: and *indexed by variant* rather than drawn (see ``_book_for``).
_BOOK_HOUSES = {
    "global_equity_long": (
        "Global",
        "International",
        "World",
        "Developed Markets",
        "Cross-Border",
        "Pan-Regional",
    ),
    "us_small_cap": (
        "US Small Cap",
        "Domestic Small Cap",
        "Russell Complement",
        "Micro & Small Cap",
        "North American Small Cap",
        "Small Cap Value",
    ),
    "em_multi_asset": (
        "EM",
        "Frontier & EM",
        "Emerging Markets",
        "EM Local",
        "Asia ex-Japan",
        "LatAm & EEMEA",
    ),
}

#: The sleeve names a house is run under. Cross-multiplied with the houses
#: above, so six houses and five sleeves give thirty distinct books per family
#: -- more than the twenty variants any family currently plans, which is what
#: makes ``_book_for`` collision-free rather than merely less collision-prone.
_BOOK_SLEEVES = ("Core", "Composite", "Sleeve", "Mandate", "Book")

_BOOKS = {
    family: tuple(f"{house} {sleeve}" for house in houses for sleeve in _BOOK_SLEEVES)
    for family, houses in _BOOK_HOUSES.items()
}


def _book_for(family: str, variant: int) -> str:
    """The book this variant attributes, cycled over the family's pool.

    Indexed, not drawn, and that is deliberate twice over. It consumes no
    ``rng`` draw -- the old code was a bare dict lookup, so every figure in
    this pack stays byte-identical to what it computed before -- and cycling
    beats sampling at the job in hand: twenty draws from a thirty-name pool
    collide about seven times by the birthday argument, while twenty indices
    into it collide never.
    """
    pool = _BOOKS.get(family)
    if not pool:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")
    return pool[variant % len(pool)]


#: How far the printed pieces may sit from the active return before the pack is
#: a data error rather than a scenario, in basis points. Generous next to the
#: float noise it is actually policing (1e-12 bp) and tight enough that any
#: return of the old arithmetic -- 1.6 bp against 372.6 -- fails on the seed
#: that produced it.
ADDITIVITY_TOLERANCE_BPS = 0.15


def _carino_k(rp: float, rb: float) -> float:
    """The textbook period factor ``ln((1+Rp)/(1+Rb)) / (Rp - Rb)``.

    Near one for any ordinary period; it is the log-to-arithmetic conversion
    that makes single-period effects link across periods without a residual.
    The old divisor ``(rp - rb)/rb`` made it ``Rb`` times this, which is not a
    smoothing factor at all -- it is the benchmark return wearing one.
    """
    if abs(rp - rb) < 1e-12:
        raise PackError("portfolio equals the benchmark: nothing to attribute")
    if 1.0 + rp <= 0.0 or 1.0 + rb <= 0.0:
        raise PackError("logarithmic Carino factor is not defined for this period")
    return math.log((1.0 + rp) / (1.0 + rb)) / (rp - rb)


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
    k = _carino_k(rp_total, rb_total)

    # The linked active return -- what the k-scaled pieces add to -- and the
    # plain difference of the two printed returns beside it. Both are the
    # pack's, and the answer may quote either; only the first is "active".
    active_bps = (rp_total - rb_total) * k * 1e4
    arithmetic_active_bps = (rp_total - rb_total) * 1e4
    if abs(active_bps) < 2.0:
        raise PackError(
            f"{work_type}/{family}: active return {active_bps:.2f} bp is noise, "
            "the attribution has no story"
        )

    effects, alloc_sum, sel_sum, inter_sum = {}, 0.0, 0.0, 0.0
    for s in chosen:
        r = rows[s]
        # Selection at the *benchmark* weight, with interaction carrying
        # ``(wp - wb)(rp - rb)`` on its own. Selection at wp is the two-term
        # decomposition's selection -- it already contains the interaction --
        # and reporting both is how the pieces stopped adding up.
        alloc = (r["wp"] - r["wb"]) * (r["rb"] - rb_total) * k
        select = r["wb"] * (r["rp"] - r["rb"]) * k
        inter = (r["wp"] - r["wb"]) * (r["rp"] - r["rb"]) * k
        alloc_sum += alloc
        sel_sum += select
        inter_sum += inter
        effects[s] = {
            "allocation_bps": round(alloc * 1e4, 1),
            "selection_bps": round(select * 1e4, 1),
            "interaction_bps": round(inter * 1e4, 1),
        }

    residual_bps = (alloc_sum + sel_sum + inter_sum) * 1e4 - active_bps
    printed_residual_bps = round(
        round(alloc_sum * 1e4, 1)
        + round(sel_sum * 1e4, 1)
        + round(inter_sum * 1e4, 1)
        - round(active_bps, 1),
        1,
    )
    if abs(residual_bps) > ADDITIVITY_TOLERANCE_BPS:
        raise PackError(
            f"{work_type}/{family}: allocation + selection + interaction miss "
            f"the active return by {residual_bps:.2f} bp -- the pack cannot ask "
            "for a reconciliation it does not contain"
        )

    inputs = {
        "sectors": chosen,
        "sector_rows": rows,
        "benchmark": "policy mix",
    }
    computed = {
        "portfolio_return": round(rp_total, 5),
        "benchmark_return": round(rb_total, 5),
        "active_bps": round(active_bps, 1),
        "arithmetic_active_bps": round(arithmetic_active_bps, 1),
        "carino_k": round(k, 6),
        "allocation_total_bps": round(alloc_sum * 1e4, 1),
        "selection_total_bps": round(sel_sum * 1e4, 1),
        "interaction_total_bps": round(inter_sum * 1e4, 1),
        # The residual a reader gets by adding the three printed totals and
        # subtracting the printed active -- a tenth of a basis point of
        # rounding, never the arithmetic (the assert above holds that at
        # ~1e-12 bp). Published because the answer is asked to show a
        # reconciliation, and a quantity the pack does not name is one the
        # answer has to invent a word for.
        "reconciling_residual_bps": printed_residual_bps,
        **{
            f"effects_{s.replace(' ', '_').replace('-', '_')}": effects[s]
            for s in chosen
        },
    }
    book = _book_for(family, variant)
    sector_list = ", ".join(chosen)
    direction = "ahead of" if active_bps >= 0 else "behind"
    question = (
        f"{book} returned {rp_total * 100:.2f}% against its benchmark's "
        f"{rb_total * 100:.2f}% over the period, {direction} the policy mix by "
        f"{abs(active_bps):.0f} bp once the period is linked. Decompose the "
        f"active return by sector ({sector_list}) into Brinson-Carino "
        f"allocation, selection and interaction effects, show that the pieces "
        f"sum to the active number via the Carino factor, and say which effect, "
        f"if any, is worth acting on."
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
            "k = ln((1+Rp)/(1+Rb)) / (Rp - Rb)",
            "alloc_i = (wp_i - wb_i)(rb_i - Rb) k",
            "sel_i = wb_i (rp_i - rb_i) k",
            "inter_i = (wp_i - wb_i)(rp_i - rb_i) k",
            "sum_i(alloc_i + sel_i + inter_i) = active = ln((1+Rp)/(1+Rb))",
        ],
        # The amendment's worked example: the question says "behind the policy
        # mix by 373 bp" because that is how a memo opens, while the arithmetic
        # is 372.6. Both used to sit in ``allowed_numbers`` as peers, so an
        # answer could quote either and the gate had no opinion. Now 373 is
        # declared for what it is -- the question's rounding of ``active_bps``
        # -- and an answer that reaches for it is reported as rounding drift.
        **assemble_contract(
            inputs,
            computed,
            # Every token the question prints that is not already canonical.
            # All three are authorised by being declared; only the first can
            # ever be *drift*, and it is the amendment's own example -- the
            # question opens "behind the policy mix by 373 bp" where the
            # arithmetic says 372.6. The two returns are printed to two
            # decimals, and "the book returned 3.31%" of a 3.309% figure is how
            # a memo is written, so `rounding_drift` refuses to argue with it.
            aliases={
                "active_bps": [f"{abs(round(active_bps))}"],
                "portfolio_return": [f"{rp_total * 100:.2f}"],
                "benchmark_return": [f"{rb_total * 100:.2f}"],
            },
            display={"active_bps": f"{active_bps:,.1f}"},
        ),
        forbidden_claims=[
            "attribution explains the next period",
            "interaction effects can be dropped",
        ],
        # One anchor, and the amendment's §A.1 reasoning for the two it drops:
        # "Carino factor" and "skill signal" were slogans the answer had to
        # enact, so a pack that did not reconcile still had to be written up as
        # though it did. The concept worth requiring is the one the question
        # actually turns on -- which kind of decision the active return came
        # from -- and "no single effect is large enough to act on" is a valid
        # answer to it.
        must_mention=["allocation versus selection"],
        # Attribution is a decomposition of a period that has happened. How
        # much of it repeats is a claim about the next period, which this pack
        # forbids and cannot support.
        abstention_question=(
            f"How much of {book}'s active return should we expect to repeat "
            f"next period, and what does that imply for the manager's fee?"
        ),
        abstention_missing="next period",
        register=pick_register(work_type, family, rng),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
