#!/usr/bin/env python3
"""Generate synthetic tool-calling SFT rows into data/processed/.

The Cosimo corpus contains no tool-call examples and the application's own tool
surface (`cosimo/mcp/tools.py`) is still a placeholder, so there is nothing real
to train against. These rows therefore teach the *format*, not a fixed tool set:
the conditional rule "given <|tool|> schemas in the system turn, emit a
<tool_call> naming one of them, then answer from its result".

Three properties are deliberate:

* **Schema variety over schema identity.** Every tool family carries several name
  variants and the distractor schemas are resampled per example, so the model
  learns to read the schema list rather than to memorise names it will never see
  again once the app ships real tools.
* **A no-call fraction** (`tools.no_call_rate`). Without examples where none of
  the offered tools fit, the model calls a tool for every question it is ever
  asked, including conceptual ones.
* **No FINAL ANSWER contract.** These rows render with `exam=False`, so the
  persona is present but the exam grading protocol is not. Mixing the two would
  teach the model to end a tool-mediated answer with `FINAL ANSWER:`, which is
  an exam-grading artefact and not something a ReAct loop should emit.

Output is written as separate files rather than appended to `sft_train.jsonl`,
so re-running this script never mutates what `01_prepare_data.py` produced.
`configs/sft.yaml` lists both files and `04_train_sft.py` concatenates them.

Example:
    ./scripts/02_prepare_tool_data.py --force
    ./scripts/02_prepare_tool_data.py --force --set tools.train_records=200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import chat, runlog  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402
from cosimo_ft import tools as tools_mod  # noqa: E402

LOGGER = logging.getLogger("prepare_tool_data")

TRAIN_FILE = "tool_train.jsonl"
VAL_FILE = "tool_val.jsonl"
OUTPUT_FILES = (TRAIN_FILE, VAL_FILE)

TICKERS = [
    "AAPL",
    "MSFT",
    "JPM",
    "GS",
    "BLK",
    "XOM",
    "NVDA",
    "BRK.B",
    "TSLA",
    "UNH",
    "V",
    "PG",
    "KO",
    "CVX",
    "META",
    "AMZN",
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF"]
HORIZONS = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}"


# ---------------------------------------------------------------------------
# Tool families
#
# `build(rng)` returns (arguments, question, result, answer). The schema's
# parameter names and the argument keys must agree -- that agreement is what the
# model is being taught to reproduce.
# ---------------------------------------------------------------------------


def _quote(rng: random.Random) -> tuple[dict, str, dict, str]:
    ticker = rng.choice(TICKERS)
    price = round(rng.uniform(25, 850), 2)
    change = round(rng.uniform(-4.5, 4.5), 2)
    args = {"symbol": ticker}
    question = rng.choice(
        [
            f"What is {ticker} trading at right now?",
            f"Pull the latest quote for {ticker}.",
            f"Where is {ticker} marked today?",
        ]
    )
    result = {"symbol": ticker, "price": price, "change_pct": change}
    direction = "up" if change >= 0 else "down"
    answer = (
        f"{ticker} is at {_fmt(price)}, {direction} {_fmt(abs(change))}% on the "
        f"session."
    )
    return args, question, result, answer


def _fundamentals(rng: random.Random) -> tuple[dict, str, dict, str]:
    ticker = rng.choice(TICKERS)
    pe = round(rng.uniform(8, 42), 1)
    roe = round(rng.uniform(4, 38), 1)
    margin = round(rng.uniform(3, 34), 1)
    args = {"symbol": ticker, "metrics": ["pe_ratio", "roe", "net_margin"]}
    question = rng.choice(
        [
            f"How is {ticker} valued on earnings, and what does its return "
            f"profile look like?",
            f"Give me {ticker}'s P/E, ROE and net margin.",
            f"Is {ticker} expensive relative to what it earns on equity?",
        ]
    )
    result = {"symbol": ticker, "pe_ratio": pe, "roe": roe, "net_margin": margin}
    answer = (
        f"{ticker} trades at {_fmt(pe, 1)}x earnings against a {_fmt(roe, 1)}% ROE "
        f"and a {_fmt(margin, 1)}% net margin. The multiple is only defensible if "
        f"that return on equity is durable rather than a cyclical peak."
    )
    return args, question, result, answer


def _yield_curve(rng: random.Random) -> tuple[dict, str, dict, str]:
    currency = rng.choice(CURRENCIES)
    two = round(rng.uniform(1.5, 5.5), 2)
    ten = round(two + rng.uniform(-1.2, 1.8), 2)
    args = {"currency": currency, "tenors": ["2Y", "10Y"]}
    question = rng.choice(
        [
            f"What does the {currency} curve look like at the 2s10s?",
            f"Pull the {currency} 2-year and 10-year yields.",
            f"Is the {currency} curve inverted?",
        ]
    )
    result = {"currency": currency, "2Y": two, "10Y": ten}
    spread = round((ten - two) * 100)
    shape = "inverted" if spread < 0 else "upward-sloping"
    answer = (
        f"{currency} 2s10s is {spread}bp ({_fmt(two)}% vs {_fmt(ten)}%), i.e. {shape}."
    )
    return args, question, result, answer


def _risk_metrics(rng: random.Random) -> tuple[dict, str, dict, str]:
    ticker = rng.choice(TICKERS)
    horizon = rng.choice(HORIZONS)
    vol = round(rng.uniform(9, 48), 1)
    beta = round(rng.uniform(0.3, 2.1), 2)
    var95 = round(rng.uniform(1.5, 9.5), 2)
    args = {"symbol": ticker, "window": horizon}
    question = rng.choice(
        [
            f"What is {ticker}'s realised vol and beta over {horizon}?",
            f"Give me the {horizon} risk profile for {ticker}.",
            f"How much downside is in {ticker} at the 95% level over {horizon}?",
        ]
    )
    result = {
        "symbol": ticker,
        "window": horizon,
        "realised_vol": vol,
        "beta": beta,
        "var_95": var95,
    }
    answer = (
        f"Over {horizon}, {ticker} ran {_fmt(vol, 1)}% realised vol with a beta of "
        f"{_fmt(beta)}; 95% VaR is {_fmt(var95)}%. Note VaR says nothing about the "
        f"tail beyond it."
    )
    return args, question, result, answer


def _fx_rate(rng: random.Random) -> tuple[dict, str, dict, str]:
    base, quote = rng.sample(CURRENCIES, 2)
    rate = round(rng.uniform(0.6, 165), 4)
    args = {"base": base, "quote": quote}
    question = rng.choice(
        [
            f"What is {base}/{quote} right now?",
            f"Pull the {base}{quote} spot rate.",
            f"How many {quote} to the {base}?",
        ]
    )
    result = {"pair": f"{base}/{quote}", "rate": rate}
    answer = f"{base}/{quote} is {_fmt(rate, 4)}."
    return args, question, result, answer


def _filing_search(rng: random.Random) -> tuple[dict, str, dict, str]:
    ticker = rng.choice(TICKERS)
    form = rng.choice(["10-K", "10-Q", "8-K"])
    year = rng.choice([2023, 2024, 2025])
    args = {"symbol": ticker, "form_type": form, "year": year}
    question = rng.choice(
        [
            f"Find {ticker}'s {year} {form}.",
            f"What did {ticker} disclose in its {form} for {year}?",
            f"Pull up the {form} {ticker} filed in {year}.",
        ]
    )
    result = {
        "symbol": ticker,
        "form_type": form,
        "year": year,
        "url": f"https://sec.gov/filings/{ticker.lower()}-{year}-{form.lower()}",
        "summary": "Segment revenue disclosed; no change to going-concern language.",
    }
    answer = (
        f"{ticker}'s {year} {form} is filed. Segment revenue is broken out and "
        f"there is no change to the going-concern language."
    )
    return args, question, result, answer


def _portfolio_position(rng: random.Random) -> tuple[dict, str, dict, str]:
    ticker = rng.choice(TICKERS)
    portfolio = rng.choice(["core-equity", "macro-overlay", "credit-opps"])
    weight = round(rng.uniform(0.2, 8.5), 2)
    pnl = round(rng.uniform(-1.8, 3.4), 2)
    args = {"portfolio_id": portfolio, "symbol": ticker}
    question = rng.choice(
        [
            f"How big is our {ticker} position in {portfolio}?",
            f"What is {portfolio} holding in {ticker}, and how is it doing?",
            f"Show me the {ticker} line in {portfolio}.",
        ]
    )
    result = {
        "portfolio_id": portfolio,
        "symbol": ticker,
        "weight_pct": weight,
        "contribution_bps": pnl * 100,
    }
    answer = (
        f"{portfolio} holds {_fmt(weight)}% in {ticker}, contributing "
        f"{_fmt(pnl * 100, 0)}bp."
    )
    return args, question, result, answer


def _option_pricer(rng: random.Random) -> tuple[dict, str, dict, str]:
    spot = round(rng.uniform(50, 400), 2)
    strike = round(spot * rng.uniform(0.85, 1.15), 2)
    vol = round(rng.uniform(0.12, 0.65), 3)
    tenor = round(rng.uniform(0.08, 2.0), 2)
    rate = round(rng.uniform(0.01, 0.055), 4)
    price = round(max(spot - strike, 0.5) * rng.uniform(0.8, 1.6), 2)
    delta = round(rng.uniform(0.2, 0.85), 3)
    args = {
        "spot": spot,
        "strike": strike,
        "volatility": vol,
        "time_to_expiry": tenor,
        "rate": rate,
        "option_type": "call",
    }
    question = rng.choice(
        [
            f"Price a {_fmt(strike)}-strike call, spot {_fmt(spot)}, "
            f"{_fmt(vol * 100, 1)}% vol, {_fmt(tenor)}y to expiry.",
            f"What is a {_fmt(tenor)}-year call struck at {_fmt(strike)} worth "
            f"with spot at {_fmt(spot)}?",
        ]
    )
    result = {"price": price, "delta": delta}
    answer = (
        f"That call marks at {_fmt(price)} with a delta of {_fmt(delta, 3)}. "
        f"Black-Scholes assumes constant vol, so treat this as a quote against a "
        f"flat surface, not a hedgeable price."
    )
    return args, question, result, answer


TOOL_FAMILIES: list[dict[str, Any]] = [
    {
        "key": "quote",
        "names": ["get_quote", "fetch_quote", "market_quote", "lookup_price"],
        "description": "Return the latest traded price and session change for a listed security.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."}
            },
            "required": ["symbol"],
        },
        "build": _quote,
    },
    {
        "key": "fundamentals",
        "names": ["get_fundamentals", "fetch_fundamentals", "company_metrics"],
        "description": "Return fundamental valuation and profitability metrics for a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metric names to return.",
                },
            },
            "required": ["symbol", "metrics"],
        },
        "build": _fundamentals,
    },
    {
        "key": "curve",
        "names": ["get_yield_curve", "sovereign_curve", "fetch_rates"],
        "description": "Return sovereign yields at the requested tenors for a currency.",
        "parameters": {
            "type": "object",
            "properties": {
                "currency": {"type": "string", "description": "ISO currency code."},
                "tenors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tenors such as 2Y or 10Y.",
                },
            },
            "required": ["currency", "tenors"],
        },
        "build": _yield_curve,
    },
    {
        "key": "risk",
        "names": ["get_risk_metrics", "risk_profile", "compute_risk"],
        "description": "Return realised volatility, beta and value-at-risk over a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "window": {
                    "type": "string",
                    "description": "Lookback window such as 1M, 1Y or 5Y.",
                },
            },
            "required": ["symbol", "window"],
        },
        "build": _risk_metrics,
    },
    {
        "key": "fx",
        "names": ["get_fx_rate", "fx_spot", "currency_rate"],
        "description": "Return the spot exchange rate for a currency pair.",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "Base currency code."},
                "quote": {"type": "string", "description": "Quote currency code."},
            },
            "required": ["base", "quote"],
        },
        "build": _fx_rate,
    },
    {
        "key": "filings",
        "names": ["search_filings", "find_filing", "regulatory_search"],
        "description": "Locate a company's regulatory filing by form type and year.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "form_type": {
                    "type": "string",
                    "description": "Filing form, e.g. 10-K.",
                },
                "year": {"type": "integer", "description": "Filing year."},
            },
            "required": ["symbol", "form_type", "year"],
        },
        "build": _filing_search,
    },
    {
        "key": "position",
        "names": ["get_position", "portfolio_position", "holding_lookup"],
        "description": "Return the weight and performance contribution of a holding in a portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Portfolio identifier.",
                },
                "symbol": {"type": "string", "description": "Ticker symbol."},
            },
            "required": ["portfolio_id", "symbol"],
        },
        "build": _portfolio_position,
    },
    {
        "key": "option",
        "names": ["price_option", "black_scholes", "option_valuation"],
        "description": "Price a European option and return its price and delta.",
        "parameters": {
            "type": "object",
            "properties": {
                "spot": {"type": "number", "description": "Spot price."},
                "strike": {"type": "number", "description": "Strike price."},
                "volatility": {
                    "type": "number",
                    "description": "Annualised volatility as a decimal.",
                },
                "time_to_expiry": {
                    "type": "number",
                    "description": "Years to expiry.",
                },
                "rate": {"type": "number", "description": "Risk-free rate."},
                "option_type": {
                    "type": "string",
                    "description": "call or put.",
                },
            },
            "required": [
                "spot",
                "strike",
                "volatility",
                "time_to_expiry",
                "rate",
                "option_type",
            ],
        },
        "build": _option_pricer,
    },
]

# Conceptual questions no data tool can answer. These carry the no-call lesson:
# the right move is to answer from knowledge, not to invent a call.
NO_CALL_QUESTIONS = [
    (
        "Why does convexity make a long-duration bond outperform a duration-matched "
        "short-duration barbell when rates move sharply in either direction?",
        "Convexity is the second derivative of price with respect to yield. Duration "
        "matching only equalises the first derivative, so for large moves the more "
        "convex position gains more on a rally and loses less on a sell-off. That "
        "asymmetry is not free: the market prices it, so the convex leg typically "
        "carries a lower yield. You are paying for the optionality in carry.",
    ),
    (
        "What is the practical difference between historical VaR and parametric VaR?",
        "Parametric VaR assumes a distribution -- usually Gaussian -- and reads the "
        "quantile off it. Historical VaR resamples actual returns and takes the "
        "empirical quantile. The parametric version is stable and wrong in the tails; "
        "the historical version is noisy but respects the fat tails that actually "
        "killed people in 2008. Neither tells you anything about losses beyond the "
        "quantile, which is why expected shortfall exists.",
    ),
    (
        "Explain why Sharpe ratio is a poor comparator across strategies with "
        "different return distributions.",
        "Sharpe divides excess return by standard deviation, which treats upside and "
        "downside dispersion identically and assumes the second moment describes the "
        "risk. A short-volatility strategy earns a small premium most months and "
        "loses catastrophically in rare ones; its realised Sharpe looks excellent "
        "right up until it does not. Comparing it to a long-convexity strategy on "
        "Sharpe alone is close to meaningless.",
    ),
    (
        "In game-theoretic terms, why do closing auctions attract disproportionate "
        "volume?",
        "The closing auction is a single-price mechanism, so it removes the "
        "continuous-market game of being picked off by faster participants. For an "
        "informed trader the auction is a pooling equilibrium: your order is "
        "indistinguishable from index rebalancing flow, so you pay less in adverse "
        "selection. The concentration of volume is the equilibrium outcome of every "
        "participant reasoning that way.",
    ),
    (
        "What does it mean when a factor model's alpha is statistically significant "
        "but economically small?",
        "It means the model reliably detects a return the factors do not explain, but "
        "the magnitude is inside transaction costs. Statistical significance scales "
        "with sample length; economic significance does not. A 12bp annual alpha "
        "measured over thirty years can be highly significant and completely "
        "unharvestable.",
    ),
]


def percentiles(values: list[int], budget: int) -> dict:
    """Nearest-rank p50/p95/p99/max plus how many rows exceed the budget."""
    if not values:
        return {"n": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "over_budget": 0}
    ordered = sorted(values)

    def at(quantile: float) -> int:
        rank = max(1, math.ceil(quantile * len(ordered)))
        return ordered[rank - 1]

    return {
        "n": len(ordered),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
        "over_budget": sum(1 for value in ordered if value > budget),
    }


def sample_schemas(
    rng: random.Random, target: dict, count: int
) -> tuple[list[dict], str]:
    """Build a shuffled schema list containing ``target`` plus distractors.

    Returns the schema list and the name the target was given in it. The name is
    resampled per example so the model reads the schema list rather than
    memorising a fixed vocabulary of tool names.
    """
    others = [f for f in TOOL_FAMILIES if f["key"] != target["key"]]
    chosen = rng.sample(others, min(count - 1, len(others)))
    target_name = rng.choice(target["names"])

    schemas = [
        tools_mod.tool_schema(target_name, target["description"], target["parameters"])
    ]
    for family in chosen:
        schemas.append(
            tools_mod.tool_schema(
                rng.choice(family["names"]),
                family["description"],
                family["parameters"],
            )
        )
    rng.shuffle(schemas)
    return schemas, target_name


def build_call_example(
    rng: random.Random, schema_count: int
) -> tuple[list[dict], list[dict], str, str]:
    """A question answered via one tool round-trip."""
    family = rng.choice(TOOL_FAMILIES)
    schemas, tool_name = sample_schemas(rng, family, schema_count)
    arguments, question, result, answer = family["build"](rng)

    messages = [
        {"role": "user", "content": question},
        tools_mod.assistant_tool_call_message(
            [{"name": tool_name, "arguments": arguments}]
        ),
        tools_mod.tool_result_message(tool_name, json.dumps(result)),
        {"role": "assistant", "content": answer},
    ]
    return messages, schemas, question, answer


def build_no_call_example(
    rng: random.Random, schema_count: int
) -> tuple[list[dict], list[dict], str, str]:
    """A question none of the offered tools can answer."""
    family = rng.choice(TOOL_FAMILIES)
    schemas, _ = sample_schemas(rng, family, schema_count)
    question, answer = rng.choice(NO_CALL_QUESTIONS)
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return messages, schemas, question, answer


def build_rows(
    cfg: dict, tokenizer: Any, count: int, seed: int, id_prefix: str
) -> list[dict]:
    """Generate ``count`` rendered tool rows in the SFT JSONL shape."""
    rng = random.Random(seed)
    low, high = config_mod.get(cfg, "tools.schemas_per_example", [2, 5])
    no_call_rate = float(config_mod.get(cfg, "tools.no_call_rate", 0.2) or 0.0)
    variation_rate = float(config_mod.get(cfg, "prompt.variation_rate", 0.0) or 0.0)

    rows = []
    for index in range(count):
        record_id = f"{id_prefix}-{index:06d}"
        schema_count = rng.randint(int(low), int(high))
        no_call = rng.random() < no_call_rate
        builder = build_no_call_example if no_call else build_call_example
        messages, schemas, question, answer = builder(rng, schema_count)

        # exam=False: the FINAL ANSWER protocol is an exam-grading contract and
        # must not leak into tool-mediated answers. The short-identity variation
        # is applied on the same deterministic id hash the exam rows use.
        system = chat.compose_system(
            cfg,
            short=chat.id_fraction(record_id) < variation_rate,
            exam=False,
        )
        conversation = [{"role": "system", "content": system}, *messages]
        rendered = chat.render_tool_example(tokenizer, conversation, schemas)

        rows.append(
            {
                "id": record_id,
                "program": "tools",
                "topic": "tool_use",
                "subtopic": "no_call" if no_call else "single_call",
                "difficulty": "n/a",
                "question_type": "tool_call",
                "generator": "tool_synth",
                "stem_family": "tool_synth",
                "question": question,
                "answer": answer,
                "distractors": [],
                "reasoning_trace": "",
                **rendered,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write rows as JSONL, replacing any previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if not rows:
        path.touch()
        return path
    return runlog.append_jsonl(path, rows)


def check_outputs(out_dir: Path, force: bool) -> None:
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not force:
        raise SystemExit(
            f"{out_dir} already contains {', '.join(existing)}; "
            "pass --force to overwrite"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="tokenizer to render with (default: model.base_id)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing tool files"
    )
    config_mod.add_config_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    cfg = config_mod.load_config(extra=args.config, overrides=args.set)

    out_dir = config_mod.harness_path(
        config_mod.get(cfg, "paths.processed_dir", "data/processed")
    )
    check_outputs(out_dir, args.force)

    enabled = bool(config_mod.get(cfg, "tools.enabled", True))
    seed = int(config_mod.get(cfg, "seed", 3407))
    max_seq_length = int(config_mod.get(cfg, "model.max_seq_length", 2048))

    if not enabled:
        # Still write the files: configs/sft.yaml lists them unconditionally, and
        # an empty file is skipped with a log line by 04_train_sft.py, whereas a
        # missing one is a hard error.
        LOGGER.warning("tools.enabled is false; writing empty tool files")
        write_jsonl(out_dir / TRAIN_FILE, [])
        write_jsonl(out_dir / VAL_FILE, [])
        print(f"tools disabled; wrote empty {TRAIN_FILE} and {VAL_FILE}")
        return

    from transformers import AutoTokenizer

    tokenizer_id = args.tokenizer_id or config_mod.get(cfg, "model.base_id")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, revision=config_mod.get(cfg, "model.revision")
    )
    if not chat.apply_chat_template_override(tokenizer, cfg):
        raise SystemExit(
            "chat.template_path is not set, so these rows would be rendered with "
            "the vendor template, which drops tool schemas entirely."
        )

    train_count = int(config_mod.get(cfg, "tools.train_records", 5000))
    val_count = int(config_mod.get(cfg, "tools.val_records", 100))

    # Distinct seeds so no validation conversation is a training conversation.
    train_rows = build_rows(cfg, tokenizer, train_count, seed, "tool-train")
    val_rows = build_rows(cfg, tokenizer, val_count, seed + 1, "tool-val")

    write_jsonl(out_dir / TRAIN_FILE, train_rows)
    write_jsonl(out_dir / VAL_FILE, val_rows)

    lengths = [
        len(tokenizer.encode(row["text"], add_special_tokens=False))
        for row in train_rows
    ]
    stats = percentiles(lengths, max_seq_length)
    no_call = sum(1 for row in train_rows if row["subtopic"] == "no_call")

    print(
        f"\nwrote {len(train_rows)} train / {len(val_rows)} val tool rows to {out_dir}"
        f"\n  no-call rows: {no_call} ({no_call / max(len(train_rows), 1):.1%})"
        f"\n  token lengths: p50={stats['p50']} p95={stats['p95']} "
        f"p99={stats['p99']} max={stats['max']}"
        f"\n  over model.max_seq_length={max_seq_length}: {stats['over_budget']}"
    )
    if stats["over_budget"]:
        print(
            "\nWARNING: some tool rows exceed the sequence budget and will be "
            "truncated from the right, cutting the final assistant turn. Raise "
            "model.max_seq_length or lower tools.schemas_per_example."
        )


if __name__ == "__main__":
    main()
