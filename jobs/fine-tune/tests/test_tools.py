"""Tool-calling wire format, rendered by the shipped chat template.

The load-bearing property here is that ``cosimo_ft.tools`` and
``configs/chat_template.jinja`` produce byte-identical strings for the same tool
call. Training targets are written with the former and serving prompts are
rendered with the latter; if they drift, the model learns a format vLLM's parser
cannot read back, and nothing else in the harness would notice.

The second property is that adding tool support did not change how an ordinary
[system, user] conversation renders -- otherwise every prepared exam row would
have to be regenerated.
"""

from __future__ import annotations

import json

import pytest

from cosimo_ft import chat, tools

from conftest import FakeTokenizer

QUESTION = "What is BLK trading at?"
ANSWER = "BLK is at 812.40, up 1.2% on the session."

WEATHER_SCHEMA = tools.tool_schema(
    "get_quote",
    "Return the latest traded price for a listed security.",
    {
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "Ticker."}},
        "required": ["symbol"],
    },
)
SECOND_SCHEMA = tools.tool_schema(
    "get_fx_rate",
    "Return the spot rate for a pair.",
    {
        "type": "object",
        "properties": {
            "base": {"type": "string"},
            "quote": {"type": "string"},
        },
        "required": ["base", "quote"],
    },
)
SCHEMAS = [WEATHER_SCHEMA, SECOND_SCHEMA]
CALL = {"name": "get_quote", "arguments": {"symbol": "BLK"}}


def conversation(system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": QUESTION},
        tools.assistant_tool_call_message([CALL]),
        tools.tool_result_message("get_quote", json.dumps({"price": 812.40})),
        {"role": "assistant", "content": ANSWER},
    ]


# --------------------------------------------------------------------------
# backward compatibility — the whole exam corpus depends on this
# --------------------------------------------------------------------------


def test_plain_conversation_renders_exactly_as_before(cfg, jinja_tokenizer):
    """A [system, user] render must be unchanged by the tool-calling support."""
    system = chat.compose_system(cfg)
    real = chat.render_example(jinja_tokenizer, QUESTION, ANSWER, system)
    fake = chat.render_example(FakeTokenizer(), QUESTION, ANSWER, system)
    assert real == fake


def test_no_tool_markers_leak_into_a_plain_render(cfg, jinja_tokenizer):
    rendered = chat.render_example(
        jinja_tokenizer, QUESTION, ANSWER, chat.compose_system(cfg)
    )
    for marker in (tools.TOOL_CALL_OPEN, tools.TOOL_SCHEMA_OPEN, "<tool_response>"):
        assert marker not in rendered["text"]


# --------------------------------------------------------------------------
# the format equality that the module exists to guarantee
# --------------------------------------------------------------------------


def test_rendered_tool_call_matches_the_template(cfg, jinja_tokenizer):
    """cosimo_ft.tools and the chat template must agree byte for byte."""
    messages = [
        {"role": "system", "content": chat.compose_system(cfg, exam=False)},
        {"role": "user", "content": QUESTION},
        tools.assistant_tool_call_message([CALL]),
    ]
    rendered = chat.render_conversation(jinja_tokenizer, messages, SCHEMAS)
    assert tools.render_tool_calls([CALL]) in rendered


def test_schema_list_is_rendered_between_the_schema_markers(cfg, jinja_tokenizer):
    messages = [
        {"role": "system", "content": chat.compose_system(cfg, exam=False)},
        {"role": "user", "content": QUESTION},
    ]
    rendered = chat.render_conversation(
        jinja_tokenizer, messages, SCHEMAS, add_generation_prompt=True
    )
    expected = (
        tools.TOOL_SCHEMA_OPEN
        + tools.render_tool_schemas(SCHEMAS)
        + tools.TOOL_SCHEMA_CLOSE
    )
    assert expected in rendered


def test_tool_schemas_are_not_html_escaped(cfg, jinja_tokenizer):
    """Jinja's own tojson escapes <, > and &; transformers' does not."""
    schema = tools.tool_schema(
        "compare", "Return true when a < b.", {"type": "object", "properties": {}}
    )
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        [schema],
        add_generation_prompt=True,
    )
    assert "a < b" in rendered
    assert "\\u003c" not in rendered


# --------------------------------------------------------------------------
# the bug this work fixes: top-level `tools`, not message['tools']
# --------------------------------------------------------------------------


def test_top_level_tools_reach_the_prompt(cfg, jinja_tokenizer):
    """The vendor template read message['tools'] and dropped the schemas."""
    messages = [
        {"role": "system", "content": chat.compose_system(cfg, exam=False)},
        {"role": "user", "content": QUESTION},
    ]
    with_tools = chat.render_conversation(
        jinja_tokenizer, messages, SCHEMAS, add_generation_prompt=True
    )
    without = chat.render_conversation(
        jinja_tokenizer, messages, None, add_generation_prompt=True
    )
    assert "get_quote" in with_tools
    assert "get_quote" not in without


def test_tools_survive_a_conversation_with_no_system_message(jinja_tokenizer):
    """LangGraph can invoke with a bare human message; schemas must not vanish."""
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [{"role": "user", "content": QUESTION}],
        SCHEMAS,
        add_generation_prompt=True,
    )
    assert tools.TOOL_SCHEMA_OPEN in rendered
    assert "get_quote" in rendered


def test_schemas_render_once_with_several_system_messages(jinja_tokenizer):
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": QUESTION},
        ],
        SCHEMAS,
        add_generation_prompt=True,
    )
    assert rendered.count(tools.TOOL_SCHEMA_OPEN) == 1


# --------------------------------------------------------------------------
# tool results, and the masking property that depends on them
# --------------------------------------------------------------------------


def test_tool_result_renders_as_a_user_turn(cfg, jinja_tokenizer):
    """train_on_responses_only splits on <|user|>; a tool result must be masked."""
    rendered = chat.render_conversation(
        jinja_tokenizer, conversation(chat.compose_system(cfg, exam=False)), SCHEMAS
    )
    assert "<|user|><tool_response>" in rendered
    # The schema marker must not be reused for results, or the two are ambiguous.
    assert "<|tool|><tool_response>" not in rendered


def test_every_assistant_turn_stays_in_the_supervised_span(cfg, jinja_tokenizer):
    rendered = chat.render_tool_example(
        jinja_tokenizer, conversation(chat.compose_system(cfg, exam=False)), SCHEMAS
    )
    assert tools.render_tool_calls([CALL]) in rendered["completion"]
    assert ANSWER in rendered["completion"]
    assert QUESTION in rendered["prompt"]
    assert QUESTION not in rendered["completion"]


def test_tool_example_preserves_the_prefix_invariant(cfg, jinja_tokenizer):
    rendered = chat.render_tool_example(
        jinja_tokenizer, conversation(chat.compose_system(cfg, exam=False)), SCHEMAS
    )
    assert rendered["text"] == rendered["prompt"] + rendered["completion"]
    assert rendered["prompt"].endswith("<|assistant|>")


def test_tool_example_without_an_assistant_turn_is_refused(jinja_tokenizer):
    with pytest.raises(ValueError, match="at least one assistant turn"):
        chat.render_tool_example(
            jinja_tokenizer, [{"role": "user", "content": QUESTION}], SCHEMAS
        )


# --------------------------------------------------------------------------
# the OpenAI wire shape, which is what actually arrives from vLLM
# --------------------------------------------------------------------------


def test_string_arguments_are_not_double_encoded(jinja_tokenizer):
    """OpenAI sends `arguments` as a JSON *string*; it must not be re-quoted."""
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_quote",
                    "arguments": '{"symbol": "BLK"}',
                },
            }
        ],
    }
    rendered = chat.render_conversation(
        jinja_tokenizer, [{"role": "user", "content": QUESTION}, message], None
    )
    assert '{"name": "get_quote", "arguments": {"symbol": "BLK"}}' in rendered


def test_flattened_langchain_tool_call_shape_renders(jinja_tokenizer):
    """LangChain's native AIMessage.tool_calls uses {"name", "args"}."""
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "get_quote", "args": {"symbol": "BLK"}}],
    }
    rendered = chat.render_conversation(
        jinja_tokenizer, [{"role": "user", "content": QUESTION}, message], None
    )
    assert '{"name": "get_quote", "arguments": {"symbol": "BLK"}}' in rendered


def test_assistant_content_and_tool_call_can_coexist(jinja_tokenizer):
    message = tools.assistant_tool_call_message([CALL], content="Let me check.")
    rendered = chat.render_conversation(
        jinja_tokenizer, [{"role": "user", "content": QUESTION}, message], None
    )
    assert "Let me check." in rendered
    assert tools.render_tool_calls([CALL]) in rendered


def test_parallel_tool_calls_render_in_order(jinja_tokenizer):
    calls = [
        {"name": "get_quote", "arguments": {"symbol": "BLK"}},
        {"name": "get_fx_rate", "arguments": {"base": "USD", "quote": "EUR"}},
    ]
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [
            {"role": "user", "content": QUESTION},
            tools.assistant_tool_call_message(calls),
        ],
        None,
    )
    assert rendered.index("get_quote") < rendered.index("get_fx_rate")
    assert rendered.count(tools.TOOL_CALL_OPEN) == 2


# --------------------------------------------------------------------------
# parse_tool_calls — the inverse, used to score generations
# --------------------------------------------------------------------------


def test_parse_round_trips_render():
    calls = [
        {"name": "get_quote", "arguments": {"symbol": "BLK"}},
        {"name": "get_fx_rate", "arguments": {"base": "USD", "quote": "EUR"}},
    ]
    assert tools.parse_tool_calls(tools.render_tool_calls(calls)) == calls


def test_parse_tolerates_surrounding_prose():
    text = f"Let me look that up.\n{tools.render_tool_calls([CALL])}\nOne moment."
    assert tools.parse_tool_calls(text) == [CALL]


def test_parse_skips_malformed_json():
    text = "<tool_call>{not json}</tool_call>" + tools.render_tool_calls([CALL])
    assert tools.parse_tool_calls(text) == [CALL]


def test_parse_returns_empty_for_plain_text():
    assert tools.parse_tool_calls("BLK is at 812.40, up 1.2%.") == []
