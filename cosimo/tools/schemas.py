"""The canonical Cosimo tool surface, in OpenAI function-schema form.

v2 defined tool schemas as inline literals inside each agentic generator
(``dataset/pipelines/templates/v2_agentic.py``), so the corpus, the serving
app and the harness could -- and did -- disagree about what ``get_fundamentals``
takes. v3 makes this module the only definition; the agentic oracle
(``dataset/pipelines/v3/oracle/``) implements exactly these names, and
``verify_v3`` rejects any row whose ``tool_schemas`` are not a subset of
:data:`TOOLS`.

``get_fundamentals`` and ``compute_metrics`` keep the names and argument shapes
the v2 rows used, so previously published conversations remain parseable by the
same registry.

Each schema is the OpenAI envelope ``{"type": "function", "function": {...}}``
that ``cosimo.tools.wire.tool_schema`` emits -- the shape vLLM forwards to the
chat template.
"""

from __future__ import annotations

from .wire import tool_schema

GET_FUNDAMENTALS = tool_schema(
    "get_fundamentals",
    "Get fundamental metrics for a stock symbol.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "metrics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["symbol"],
    },
)

COMPUTE_METRICS = tool_schema(
    "compute_metrics",
    "Compute DCF valuation metrics.",
    {
        "type": "object",
        "properties": {
            "fcff": {"type": "number"},
            "wacc": {"type": "number"},
            "terminal_growth": {"type": "number"},
            "proj_years": {"type": "integer"},
        },
        "required": ["fcff", "wacc", "terminal_growth", "proj_years"],
    },
)

GET_RETURNS_SERIES = tool_schema(
    "get_returns_series",
    "Get the daily return history of a book as of a valuation date.",
    {
        "type": "object",
        "properties": {
            "portfolio_id": {"type": "string"},
            "as_of": {"type": "string"},
            "window_days": {"type": "integer"},
        },
        "required": ["portfolio_id"],
    },
)

GET_POSITIONS = tool_schema(
    "get_positions",
    "Get the sector holdings and weights of a book.",
    {
        "type": "object",
        "properties": {
            "book_id": {"type": "string"},
            "asset_class": {"type": "string"},
        },
        "required": ["book_id"],
    },
)

GET_TRANSACTION_COST = tool_schema(
    "get_transaction_cost",
    "Estimate execution cost against the arrival price for an order.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "side": {"type": "string"},
            "shares": {"type": "number"},
            "arrival_price": {"type": "number"},
        },
        "required": ["symbol", "side", "shares", "arrival_price"],
    },
)

TOOLS: dict[str, dict] = {
    schema["function"]["name"]: schema
    for schema in (
        GET_FUNDAMENTALS,
        COMPUTE_METRICS,
        GET_RETURNS_SERIES,
        GET_POSITIONS,
        GET_TRANSACTION_COST,
    )
}
