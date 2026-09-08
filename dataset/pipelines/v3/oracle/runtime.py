"""The tool oracle: registry tools, answered from the fact pack and nothing else.

Spec §5.4's cell -- "oracle executes with optional injected fault" -- and the
rule that makes the whole agentic corpus verifiable is enforced here, in the
way the executors are written: *every number in a result is a projection of a
number the pack already owns*. The oracle never performs new arithmetic. A
computer derived the figures once, at pack time, from seeded inputs; the
oracle's job is to decide **which** of them a tool call reveals, not to
produce fresh ones. That single discipline is what lets ``verify_v3`` replay a
trajectory byte for byte, and what makes axis 10's grounding check ("every
final number in the pack ∪ the tool results") a set-membership test instead
of a negotiation.

Consequences worth stating plainly:

* an answer the pack cannot support comes back *honest and empty* -- no rows,
  a note, zero invented figures. The teacher that quotes a number the oracle
  did not return has a problem the gate will find;
* an argument mismatch (``compute_metrics`` called with someone else's wacc)
  is an ``expected`` echo of the scenario's own inputs -- corrective data the
  teacher can retry against, still pack numbers only;
* the five tools answer differently per work type: ``tools_for`` is the
  availability table, and it is data, consulted by the render loop, the
  fixture harness and the board alike.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import isfinite

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # dataset; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ``validate_call`` is a deliberate re-export: the renderer and the board
# both call it through the oracle namespace, so the schema check keeps the
# single home §5.8 asked for and the import is a name of the package's
# surface, not of this file's body (hence F401, which reads one file at a
# time).
from cosimo.tools import SCHEMAS, validate_call  # noqa: E402 F401

#: Registry names, iteration order fixed -- the order the brief advertises
#: tools in, the order the fixture hashes bodies built on, so it is a name of
#: the corpus, not an accident of dict.
SCHEMA_NAMES = tuple(SCHEMAS)

_TOLERANCE = 1e-9


class OracleError(RuntimeError):
    """The call was structurally impossible -- unknown tool, wrong shape.

    Distinct from an *empty answer*: an empty result is the pack truthfully
    saying "I have no such rows" and is a legitimate observation; asking a
    tool that does not exist is a programming bug and must not masquerade as
    data.
    """


@dataclass(frozen=True)
class Call:
    """One validated tool invocation, ready to execute."""

    name: str
    arguments: dict


def _scalars(node: dict) -> dict:
    """The finite, non-boolean numbers of *node*, keyed as the pack keys them."""
    return {
        key: value
        for key, value in node.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    }


def _entity_identity(pack: dict) -> str:
    for entity in pack.get("entities") or ():
        for field in ("ticker", "book_id", "symbol", "name"):
            value = entity.get(field)
            if isinstance(value, str) and value:
                return value
    return ""


def _same(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= _TOLERANCE * max(
            1.0, abs(float(left)), abs(float(right))
        )
    return left == right


def _matches(args: dict, pack: dict, field: str, pack_field: str) -> bool:
    return _same(args.get(field), pack["inputs"].get(pack_field))


def _fundamentals(pack: dict, args: dict) -> dict:
    identity = _entity_identity(pack)
    requested = str(args.get("symbol") or "")
    if not identity or requested.casefold() != identity.casefold():
        return {
            "rows": {},
            "requested": requested,
            "note": "no rows for that identifier in this scenario",
        }
    available = {
        **_scalars(pack.get("inputs") or {}),
        **_scalars(pack.get("computed") or {}),
    }
    wanted = [str(m) for m in (args.get("metrics") or [])]
    picked = {key: available[key] for key in wanted if key in available}
    missing = [key for key in wanted if key not in available]
    result = {"rows": picked, "as_of": pack.get("as_of")}
    if missing:
        result["missing"] = missing
    return result


def _metrics(pack: dict, args: dict) -> dict:
    """The DCF block, echoed when the caller argues from the scenario's inputs.

    The oracle does not re-run the valuation -- the computer already did. It
    checks the four arguments against the pack's own inputs (within float
    tolerance) and either hands back the derived block or echoes what the
    scenario actually says, so a mistaken call is *correctable data*, still
    every number a pack number.
    """
    inputs = pack.get("inputs") or {}
    expected = {
        "fcff": (pack.get("computed") or {}).get("fcff_year1_m"),
        "wacc": inputs.get("wacc"),
        "terminal_growth": inputs.get("terminal_growth"),
        "proj_years": inputs.get("explicit_years"),
    }
    agree = all(_same(args.get(key), value) for key, value in expected.items())
    if not agree:
        return {
            "status": "arguments do not match this scenario",
            "expected": {k: v for k, v in expected.items() if v is not None},
            "note": "the pack's own inputs are authoritative; retry against them",
        }
    computed = pack.get("computed") or {}
    rows = _scalars(computed)
    return {"method": "two-stage dcf", "rows": rows, "as_of": pack.get("as_of")}


def _series(pack: dict, args: dict) -> dict:
    identity = _entity_identity(pack)
    requested = str(args.get("portfolio_id") or "")
    if identity and requested and requested.casefold() != identity.casefold():
        return {
            "series": [],
            "requested": requested,
            "note": "no book by that name on this desk",
        }
    inputs = _scalars(pack.get("inputs") or {})
    computed = _scalars(pack.get("computed") or {})
    if {"mu_daily", "sigma_daily"} <= inputs.keys():
        series = {
            key: inputs[key]
            for key in ("mu_daily", "sigma_daily", "horizon_days", "z95", "z99")
            if key in inputs
        }
        series.update(
            {
                key: computed[key]
                for key in ("var95_1d_m", "es95_1d_m")
                if key in computed
            }
        )
        return {"series": series, "as_of": pack.get("as_of")}
    if {"portfolio_return", "benchmark_return"} <= computed.keys():
        return {
            "series": {
                key: computed[key]
                for key in ("portfolio_return", "benchmark_return", "active_bps")
                if key in computed
            },
            "as_of": pack.get("as_of"),
        }
    return {"series": [], "note": "no return series on file for this scenario"}


def _positions(pack: dict, args: dict) -> dict:
    rows = []
    sector_rows = (pack.get("inputs") or {}).get("sector_rows")
    if isinstance(sector_rows, dict):
        for sector in (pack.get("inputs") or {}).get("sectors") or sorted(sector_rows):
            row = {"sector": sector}
            row.update(sector_rows.get(sector) or {})
            rows.append(row)
    else:
        inputs = pack.get("inputs") or {}
        if "arrival_price" in inputs:  # the tca book is one order deep
            rows.append(
                {
                    key: inputs[key]
                    for key in ("symbol", "side", "shares", "arrival_price")
                    if key in inputs
                }
            )
        elif "book_value_m" in inputs:
            rows.append(
                {
                    "book_id": _entity_identity(pack),
                    "value_m": inputs["book_value_m"],
                }
            )
    identity = _entity_identity(pack)
    requested = str(args.get("book_id") or "")
    if (
        rows
        and identity
        and requested
        and requested.casefold()
        not in {
            identity.casefold(),
            str(rows[0].get("book_id", "")).casefold(),
            "all",
        }
    ):
        return {"rows": [], "requested": requested, "note": "no book by that name"}
    result = {"rows": rows, "as_of": pack.get("as_of")}
    if args.get("asset_class") and not rows:
        result["note"] = "no positions on file for this scenario"
    return result


def _cost(pack: dict, args: dict) -> dict:
    inputs = pack.get("inputs") or {}
    if "arrival_price" not in inputs:
        return {"rows": {}, "note": "no execution book on this desk"}
    expected = {
        "symbol": inputs.get("symbol"),
        "side": inputs.get("side"),
        "shares": inputs.get("shares"),
        "arrival_price": inputs.get("arrival_price"),
    }
    agree = all(_same(args.get(key), value) for key, value in expected.items())
    if not agree:
        return {
            "status": "arguments do not match this order",
            "expected": {k: v for k, v in expected.items() if v is not None},
            "note": "the pack's own order is authoritative; retry against it",
        }
    rows = _scalars(pack.get("computed") or {})
    return {"rows": rows, "as_of": pack.get("as_of")}


_EXECUTANTS = {
    "get_fundamentals": _fundamentals,
    "compute_metrics": _metrics,
    "get_returns_series": _series,
    "get_positions": _positions,
    "get_transaction_cost": _cost,
}

#: Which of the five tools the pack of this work type can honestly answer
#: with. Not a whitelist for the *teacher* -- calling another and getting an
#: honest empty back is legal and educational -- but the map the scripted
#: fixture walks, and the map a reader uses to predict what a clean
#: trajectory should have contained.
TOOL_AVAILABILITY = {
    "valuation.equity.dcf": ("get_fundamentals", "compute_metrics"),
    "valuation.equity.multiples": ("get_fundamentals", "get_fundamentals"),
    "risk.market.var_es": ("get_returns_series", "get_positions"),
    "portfolio.attribution.brinson_carino": ("get_positions", "get_fundamentals"),
    "execution.tca.arrival": ("get_fundamentals", "get_transaction_cost"),
}


def tools_for(work_type: str) -> tuple[str, ...]:
    """The answerable tools of *work_type*, in scripted-call order."""
    try:
        return TOOL_AVAILABILITY[work_type]
    except KeyError:
        raise OracleError(f"no oracle map for work type {work_type!r}") from None


def execute(call: Call, pack: dict) -> dict:
    """Answer *call* from *pack* alone. Deterministic; never fabricates.

    The registry is the gatekeeper: a name outside it is an :class:`OracleError`,
    never an empty dict -- silently emptying the unknown-tool case is how a
    corpus starts teaching models to shrug at outages.
    """
    if call.name not in SCHEMAS:
        raise OracleError(f"tool {call.name!r} is not in the registry")
    executant = _EXECUTANTS.get(call.name)
    if executant is None:
        raise OracleError(
            f"tool {call.name!r} is registered but has no oracle executant; "
            "the corpus would ship conversations the server cannot answer"
        )
    if not isinstance(call.arguments, dict):
        raise OracleError(f"arguments for {call.name!r} must be an object")
    return executant(pack, call.arguments)
