"""The registry contract the corpus is written against.

The snapshot below is deliberate: a schema edit that changes bytes must also
change ``REGISTRY_VERSION`` and re-surface here, because rows already on disk
carry ``tool_schemas`` and the served template renders them verbatim. Silent
schema drift is the exact failure (spec §12: "Tool schema drift") this file
makes impossible.
"""

import json

import pytest

from cosimo.tools.registry import (  # noqa: E402
    REGISTRY_VERSION,
    SCHEMAS,
    names,
    render_tools,
    validate_call,
)

EXPECTED_NAMES = [
    "get_fundamentals",
    "compute_metrics",
    "get_returns_series",
    "get_positions",
    "get_transaction_cost",
]

SNAPSHOT = """{"compute_metrics": {"function": {"description": "Compute DCF valuation metrics.", "name": "compute_metrics", "parameters": {"properties": {"fcff": {"type": "number"}, "proj_years": {"type": "integer"}, "terminal_growth": {"type": "number"}, "wacc": {"type": "number"}}, "required": ["fcff", "wacc", "terminal_growth", "proj_years"], "type": "object"}}, "type": "function"}, "get_fundamentals": {"function": {"description": "Get fundamental metrics for a stock symbol.", "name": "get_fundamentals", "parameters": {"properties": {"metrics": {"items": {"type": "string"}, "type": "array"}, "symbol": {"type": "string"}}, "required": ["symbol"], "type": "object"}}, "type": "function"}, "get_positions": {"function": {"description": "Get the sector holdings and weights of a book.", "name": "get_positions", "parameters": {"properties": {"asset_class": {"type": "string"}, "book_id": {"type": "string"}}, "required": ["book_id"], "type": "object"}}, "type": "function"}, "get_returns_series": {"function": {"description": "Get the daily return history of a book as of a valuation date.", "name": "get_returns_series", "parameters": {"properties": {"as_of": {"type": "string"}, "portfolio_id": {"type": "string"}, "window_days": {"type": "integer"}}, "required": ["portfolio_id"], "type": "object"}}, "type": "function"}, "get_transaction_cost": {"function": {"description": "Estimate execution cost against the arrival price for an order.", "name": "get_transaction_cost", "parameters": {"properties": {"arrival_price": {"type": "number"}, "shares": {"type": "number"}, "side": {"type": "string"}, "symbol": {"type": "string"}}, "required": ["symbol", "side", "shares", "arrival_price"], "type": "object"}}, "type": "function"}}"""


def test_registry_version_and_names_are_pinned():
    assert REGISTRY_VERSION == "v3.0"
    assert names() == EXPECTED_NAMES


def test_schemas_match_the_locked_snapshot():
    assert json.dumps(SCHEMAS, sort_keys=True) == SNAPSHOT


def test_schemas_are_openai_envelopes():
    for name, schema in SCHEMAS.items():
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_v2_names_survive_byte_compatibility():
    """The two names v2 agentic rows embed stay addressable by the registry."""
    assert "get_fundamentals" in SCHEMAS and "compute_metrics" in SCHEMAS


def test_render_tools_is_json_and_registry_ordered():
    rendered = json.loads(render_tools(["compute_metrics", "get_fundamentals"]))
    assert [s["function"]["name"] for s in rendered] == [
        "get_fundamentals",
        "compute_metrics",
    ], "selection must follow registry order, not caller order"
    assert json.loads(render_tools()) == [SCHEMAS[n] for n in EXPECTED_NAMES]


def test_render_tools_refuses_unknown_names():
    with pytest.raises(ValueError, match="not in the registry"):
        render_tools(["get_weather"])


def test_validate_call_accepts_a_good_call():
    assert (
        validate_call("get_fundamentals", {"symbol": "AAPL", "metrics": ["pe"]}) == []
    )
    assert validate_call("get_fundamentals", {"symbol": "AAPL"}) == []


@pytest.mark.parametrize(
    ("name", "arguments", "needle"),
    [
        ("no_such_tool", {}, "unknown tool"),
        ("get_fundamentals", {}, "missing required"),
        ("get_fundamentals", {"symbol": "A", "extra": 1}, "unknown argument"),
        ("get_fundamentals", "AAPL", "must be an object"),
        (
            "compute_metrics",
            {"fcff": 5, "wacc": 0.09, "terminal_growth": 0.02, "proj_years": 5.5},
            "must be integer",
        ),
        ("get_fundamentals", {"symbol": True}, "must be string"),
    ],
)
def test_validate_call_reports_every_way_a_call_can_be_wrong(name, arguments, needle):
    errors = validate_call(name, arguments)
    assert errors, "a bad call must produce at least one error"
    assert any(needle in e for e in errors), errors
