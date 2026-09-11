"""risk.market.var_es -- parametric VaR / expected-shortfall fact computer.

Parametric-on-purpose, because the teaching content is the *assumption list*:
z-score arithmetic is where a Gaussian narrative goes wrong on a real book
(gap risk, fat tails, correlation breakdown), and the pack's must_mention makes
the completion say so. The arithmetic itself is closed-form and recomputable:

    VaR_a(h) = V * (z_a * sigma - mu) * sqrt(h)
    ES_a(h)  = V * (sigma * phi(z_a) / (1 - a) - mu) * sqrt(h)

with phi the standard normal density. No Monte Carlo -- a sampled estimator
would make the verified number depend on the sampler, which is the trap this
corpus spent two rounds avoiding.
"""

from __future__ import annotations

import math
import random

from ..seed import pack_seed, rng_for
from .base import FactPack, PackError, assemble_contract, pick_as_of
from .registers import pick_register

WORK_TYPE = "risk.market.var_es"
FAMILIES = ("rates_book", "equity_longonly", "credit_focused")

_Z95, _Z99 = 1.645, 2.326
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

#: The sleeve words a book is run under, cross-multiplied with the desks below.
#: Two names per family was the old pool, so ten variants of `rates_book` were
#: ten VaR papers on the same two books -- the scenario repainted, which is the
#: v1 pathology at a smaller constant. Thirty names per family is wide enough
#: that a twenty-variant family rarely repeats, and the draw count is unchanged
#: (one `rng.choice`, whatever the pool holds), so every figure these packs
#: compute is byte-identical to what it computed before.
_BOOK_SLEEVES = ("Book", "Sleeve", "Portfolio", "Mandate", "Desk")


def _books(*desks: str) -> tuple[str, ...]:
    return tuple(f"{desk} {sleeve}" for desk in desks for sleeve in _BOOK_SLEEVES)


_BANDS = {
    "rates_book": {
        "value_m": (200.0, 4000.0, 1),
        "sigma_daily": (0.0020, 0.0110, 5),
        "mu_daily": (-0.0006, 0.0006, 5),
        "books": _books(
            "Sovereign Curve",
            "Duration Overlay",
            "Govvie Relative Value",
            "Rates Carry",
            "Inflation-Linked",
            "Swap Spread",
        ),
    },
    "equity_longonly": {
        "value_m": (80.0, 1500.0, 1),
        "sigma_daily": (0.0070, 0.0240, 5),
        "mu_daily": (-0.0010, 0.0012, 5),
        "books": _books(
            "Global Long Extension",
            "Small/Mid Alpha",
            "Quality Compounder",
            "Dividend Growth",
            "Concentrated Conviction",
            "Low Volatility",
        ),
    },
    "credit_focused": {
        # Holdout family: wide, gappy distributions -- the Gaussian z is the
        # weakest it will ever be, and the audit register calls that out.
        "value_m": (60.0, 900.0, 1),
        "sigma_daily": (0.0100, 0.0330, 5),
        "mu_daily": (-0.0015, 0.0010, 5),
        "books": _books(
            "IG Credit",
            "BB/B Transition",
            "Crossover Credit",
            "Short Duration Credit",
            "Subordinated Financials",
            "Fallen Angel",
        ),
    },
}


def _normal_pdf(z: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def _build(work_type: str, family: str, variant: int, rng: random.Random) -> FactPack:
    bands = _BANDS.get(family)
    if bands is None:
        raise PackError(f"{WORK_TYPE}: unknown scenario family {family!r}")

    value_m = round(rng.uniform(*bands["value_m"][:2]), bands["value_m"][2])
    sigma = round(rng.uniform(*bands["sigma_daily"][:2]), bands["sigma_daily"][2])
    mu = round(rng.uniform(*bands["mu_daily"][:2]), bands["mu_daily"][2])
    horizon = rng.choice((1, 5, 10))
    if sigma <= 0 or value_m <= 0:
        raise PackError(
            f"{work_type}/{family}: degenerate book (V={value_m}, s={sigma})"
        )
    scale = math.sqrt(horizon)
    var95_1 = value_m * (_Z95 * sigma - mu)
    var99_1 = value_m * (_Z99 * sigma - mu)
    if var95_1 <= 0:
        raise PackError(
            f"{work_type}/{family}: the drift swamps the 95% quantile "
            f"(mu={mu}, sigma={sigma}); nothing to price"
        )

    # Indexed, not drawn -- see ``_books``. Every book in the family's pool is
    # reached before any is repeated.
    books = bands["books"]
    book = books[variant % len(books)]
    computed = {
        "var95_1d_m": round(var95_1, 3),
        "var99_1d_m": round(var99_1, 3),
        "var95_h_m": round(var95_1 * scale, 3),
        "var99_h_m": round(var99_1 * scale, 3),
        "es95_1d_m": round(value_m * (sigma * _normal_pdf(_Z95) / 0.05 - mu), 3),
        "horizon_scale_sqrt": round(scale, 4),
    }
    inputs = {
        "book_value_m": value_m,
        "mu_daily": mu,
        "sigma_daily": sigma,
        "horizon_days": horizon,
        "z95": _Z95,
        "z99": _Z99,
    }
    question = (
        f"Book '{book}' marks at ${value_m:,.1f}M with a daily mean of "
        f"{mu * 100:.2f}% and a daily volatility of {sigma * 100:.2f}%. Under "
        f"a normal, iid-per-day assumption: give the 1-day and the {horizon}-day "
        f"95% and 99% VaRs and the 95% expected shortfall, and say what the "
        f"normality assumption is hiding for this book."
    )
    return FactPack(
        schema_version="v3.0",
        scenario_id=f"{work_type}.{family}",
        work_type=work_type,
        seed=pack_seed(work_type, family, variant),
        variant=variant,
        entities=[
            {
                "name": book,
                "book_id": book.lower().replace("/", "-").replace(" ", "_"),
                "currency": "USD",
            }
        ],
        inputs=inputs,
        computed=computed,
        formulas=[
            "VaR_a(h) = V (z_a sigma - mu) sqrt(h)",
            "ES_a(h) = V (sigma phi(z_a) / (1 - a) - mu) sqrt(h)",
        ],
        # The question prints the book value to one decimal and both daily
        # moments as percents; each is a rounding of a canonical figure, so
        # each is an alias of it rather than a peer in the allow-list. The
        # confidence levels are the *contract's* numbers, not the book's --
        # 95 and 99 name which VaR is being asked for -- so they stay extras.
        **assemble_contract(
            inputs,
            computed,
            aliases={
                "book_value_m": [f"{value_m:.1f}"],
                "mu_daily": [f"{mu * 100:.2f}"],
                "sigma_daily": [f"{sigma * 100:.2f}"],
            },
            display={"book_value_m": f"{value_m:,.1f}"},
            # The question's own quantities: which two VaRs it asks for, and
            # the one-day horizon it always prints beside the h-day one. Named
            # rather than dropped into the allow-list, because an answer that
            # says "the 95% VaR" is stating a quantity, not inventing one.
            canonical_extra={
                "confidence_95_pct": 95.0,
                "confidence_99_pct": 99.0,
                "horizon_1d": 1.0,
            },
        ),
        forbidden_claims=[
            "the book will not lose more than VaR",
            "95% VaR captures the tail",
            "VaR is a loss forecast",
        ],
        must_mention=[
            "normality assumption",
            "square-root horizon scaling",
            "expected shortfall tail",
        ],
        register=pick_register(work_type, family, rng),
        as_of=pick_as_of(rng),
        question=question,
    )


def compute(family: str, variant: int) -> FactPack:
    seed = pack_seed(WORK_TYPE, family, variant)
    return _build(WORK_TYPE, family, variant, rng_for(seed))


__all__ = ["WORK_TYPE", "FAMILIES", "compute"]
