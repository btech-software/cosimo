#!/usr/bin/env python3
"""Build the assistant suite that asks the questions this corpus was built for.

`suites/grounded.jsonl` asks a competent desk question -- a Sharpe, a duration,
a participation rate -- and any capable model answers it. None of those rows can
fail the way v3's own review found rows failing: calling a 4.86% clip a pacing
problem, naming an effect to act on out of pieces that do not reconcile, reading
a negative drift as a gain, or turning an enterprise value into an ownership
call. A green board on generic prompts says nothing about those.

So the traps are generated *from the packs themselves*, not retyped: every
figure, the missing price and the absent decision benchmark come out of
`compute_pack`, which is the same function the corpus and the verify board use.
A trap whose numbers drifted from its pack would be measuring nothing.

    uv run --group corpus python dataset/tools/trap_suite.py

Eval-only, and the coordinates it draws from must stay out of training or the
measurement is memorisation. Two of the four are already the configured holdout
families; the other two carry the gold bar, and `write.gold_bar_ids` keeps the
renderer off them.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(_HERE)
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.packs.base import convention_numbers  # noqa: E402

OUT = os.path.join(_DATASET, "..", "jobs", "fine-tune", "suites", "traps.jsonl")

#: ``(work_type, family, variant)`` -> the traps that pack can pose. Each entry
#: is ``(id, register, ask, must_mention, extra_numbers)``; the numbers an
#: answer may use come from the pack, plus anything the question itself prints.
TRAPS = {
    ("execution.tca.arrival", "large_cap_intraday", 0): [
        (
            "trap_tca_schedule",
            "desk_chat",
            "{shares} shares of {name} against ADV {adv}, arrival {arrival}, "
            "quoted spread {spread} bp, daily vol {vol}%. Under the square-root "
            "model, what does this cost against arrival -- and is the schedule "
            "or the impact the thing to manage here?",
            ["impact", "participation"],
        ),
        (
            "trap_tca_benchmark",
            "desk_chat",
            "Same order: {shares} shares, arrival {arrival}, fill implied at "
            "{fill}. What is the shortfall measured against, and what would it "
            "take to measure it against the decision price instead?",
            ["arrival"],
        ),
    ],
    ("portfolio.attribution.brinson_carino", "global_equity_long", 0): [
        (
            "trap_attr_reconcile",
            "ic_memo",
            "{book} returned {rp}% against a benchmark {rb}%. Allocation is "
            "{alloc} bp, selection {sel} bp, interaction {inter} bp against an "
            "active return of {active} bp. Do the pieces reconcile, and which "
            "effect is worth acting on?",
            ["allocation", "selection"],
        ),
        (
            "trap_attr_k",
            "ic_memo",
            "For the same book the Carino factor is {k}. What is that factor "
            "doing, and what would you conclude if the pieces did not sum to "
            "the active return?",
            ["Carino"],
        ),
    ],
    ("risk.market.var_es", "equity_longonly", 0): [
        (
            "trap_var_drift",
            "risk_committee",
            "Book marks at ${value}M with a daily mean of {mu}% and daily vol "
            "{sigma}%. The 1-day 95% VaR is {var95} and the 95% expected "
            "shortfall {es95}. What does the daily mean do to the expected "
            "loss over the horizon?",
            ["drift", "expected shortfall"],
        ),
        (
            "trap_var_tail",
            "risk_committee",
            "Same book: 1-day 95% VaR {var95}, 95% expected shortfall {es95}. "
            "What is the normality assumption hiding, and is the book being "
            "paid for the risk it carries?",
            ["normality", "tail"],
        ),
    ],
    ("valuation.equity.dcf", "cyclical_industrial", 0): [
        (
            "trap_dcf_ev_as_price",
            "ic_memo",
            "The FCFF model implies an enterprise value of ${ev}M for {name}, "
            "with terminal growth {g}% against a WACC of {wacc}%. Should we own "
            "it?",
            ["enterprise value"],
        ),
        (
            "trap_dcf_terminal",
            "ic_memo",
            "Same model: terminal value ${tv}M of an enterprise value of "
            "${ev}M, terminal growth {g}%. What in that split would you "
            "challenge first?",
            ["terminal"],
        ),
    ],
}


def _fields(pack: dict) -> dict:
    """Every substitution a trap prompt can ask for, from the pack itself."""
    computed, inputs = pack["computed"], pack["inputs"]
    entity = (pack.get("entities") or [{}])[0]
    out = {
        "name": entity.get("name", ""),
        "book": entity.get("name", ""),
    }
    if pack["work_type"] == "execution.tca.arrival":
        out |= {
            "shares": f"{int(inputs['shares']):,}",
            "adv": f"{int(inputs['adv_shares']):,}",
            "arrival": f"{inputs['arrival_price']:,.2f}",
            "spread": f"{inputs['quoted_spread_bps']:g}",
            "vol": f"{inputs['sigma_daily'] * 100:.2f}",
            "fill": pack["display"].get("avg_fill_price", ""),
        }
    elif pack["work_type"] == "portfolio.attribution.brinson_carino":
        out |= {
            "rp": f"{computed['portfolio_return'] * 100:.2f}",
            "rb": f"{computed['benchmark_return'] * 100:.2f}",
            "alloc": f"{computed['allocation_total_bps']:g}",
            "sel": f"{computed['selection_total_bps']:g}",
            "inter": f"{computed['interaction_total_bps']:g}",
            "active": f"{computed['active_bps']:g}",
            "k": f"{computed['carino_k']:g}",
        }
    elif pack["work_type"] == "risk.market.var_es":
        out |= {
            "value": f"{inputs['book_value_m']:,.1f}",
            "mu": f"{inputs['mu_daily'] * 100:.2f}",
            "sigma": f"{inputs['sigma_daily'] * 100:.2f}",
            "var95": f"{computed['var95_1d_m']:g}",
            "es95": f"{computed['es95_1d_m']:g}",
        }
    elif pack["work_type"] == "valuation.equity.dcf":
        out |= {
            "ev": f"{computed['enterprise_value_m']:g}",
            "tv": f"{computed['terminal_value_m']:g}",
            "g": f"{inputs['terminal_growth'] * 100:.2f}",
            "wacc": f"{inputs['wacc'] * 100:.2f}",
        }
    return out


def build() -> list[dict]:
    rows: list[dict] = []
    for coords, traps in TRAPS.items():
        pack = compute_pack(*coords).to_dict()
        fields = _fields(pack)
        allowed = sorted({float(v) for v in pack["allowed_numbers"]})
        for trap_id, register, ask, must_mention in traps:
            rows.append(
                {
                    "id": trap_id,
                    "register": register,
                    "prompt": ask.format(**fields),
                    "must_mention": must_mention,
                    "allowed_numbers": allowed,
                    "conventions": convention_numbers(pack),
                    "scenario_id": pack["scenario_id"],
                    "variant": pack["variant"],
                }
            )
    return rows


RESERVED = os.path.join(_DATASET, "eval", "reserved_coordinates.json")


def write_reserved() -> str:
    """Declare the trap packs off-limits to generation, for the renderer.

    One author for the reservation: the suite that asks the questions is the
    thing that knows which packs must stay out of training.
    """
    payload = {
        "why": (
            "Packs the trap suite (jobs/fine-tune/suites/traps.jsonl) asks "
            "questions about. Rendering any record type on these coordinates "
            "puts the evaluation's own figures into training, which turns a "
            "generalisation test into a recall test."
        ),
        "reserved": [
            {
                "work_type": work_type,
                "family": family,
                "variant": variant,
                "suite": "traps",
            }
            for (work_type, family, variant) in sorted(TRAPS)
        ],
    }
    os.makedirs(os.path.dirname(RESERVED), exist_ok=True)
    with open(RESERVED, "w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return RESERVED


def main() -> int:
    rows = build()
    path = os.path.abspath(OUT)
    with open(path, "w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} trap prompts to {path}")
    print(f"reserved {len(TRAPS)} coordinates in {write_reserved()}")
    for row in rows:
        print(f"  {row['id']:24s} {row['scenario_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
