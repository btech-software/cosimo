"""The shipped ChatML template, and the properties training depends on it for.

The student moved from Phi-4 to Qwen3.8, which changed every turn marker
(``<|system|>``/``<|user|>``/``<|end|>`` to ``<|im_start|>role\\n``/``<|im_end|>``)
while deliberately keeping the tool-calling wire format. These are the four
properties that a template rewrite can break silently, and each has a
downstream failure that is expensive to find any other way:

* the **prefix invariant** -- ``render_example`` isolates the supervised
  completion by string-slicing the full render against the prompt render, and a
  template whose two renders diverge would raise or, worse, mask the wrong span;
* the **EOS shape** -- under ChatML the turn terminator *is* the EOS token, so a
  template that also appended ``eos_token`` would train a doubled terminator and
  ``to_pref_row``'s strip would remove only one of them;
* the **masking markers** -- ``chat.instruction_part``/``response_part`` must be
  exactly what the template emits at a turn boundary, or
  ``train_on_responses_only`` masks the wrong side;
* the **wire contract** -- ``cosimo/tools/wire.py`` is shared with the corpus
  generator, so a schema block or tool call rendered even one character
  differently teaches a format the runtime cannot parse back.
"""

from __future__ import annotations

import json

import pytest

from cosimo_ft import chat, tools
from cosimo_ft import config as config_mod

from harness_fixtures import EOS_TOKEN, FakeTokenizer

QUESTION = "What is the modified duration of the bond?"
COMPLETION = "About 6.4 years, before convexity."


@pytest.fixture(scope="module")
def cfg() -> dict:
    return config_mod.load_config()


SCHEMA = tools.tool_schema(
    "get_fundamentals",
    "Fundamentals for a listed security.",
    {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
)


# --------------------------------------------------------------------------
# the shipped template is ChatML and carries no vendor identity
# --------------------------------------------------------------------------


def test_the_template_emits_chatml_turn_markers(jinja_tokenizer):
    """Asserted against what the template RENDERS, not against its source.

    The source names the Phi markers in its header comment, explaining why they
    are gone; a substring check over the file would read those and fail.
    """
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": COMPLETION},
        ],
    )
    for marker in ("<|im_start|>system\n", "<|im_start|>user\n", "<|im_end|>"):
        assert marker in rendered
    # The Phi markers are real tokens in Phi's vocabulary and ordinary text in
    # Qwen's; leaving them on turn boundaries would spend tokens on every turn
    # and put the model off its own distribution.
    for marker in ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>"):
        assert marker not in rendered


def test_the_template_carries_no_vendor_identity(chat_template_text):
    """The reason the vendor template was overridden, and it outlives the swap."""
    assert "Microsoft" not in chat_template_text
    assert "Your name is" not in chat_template_text


# --------------------------------------------------------------------------
# the invariants render_example / to_pref_row depend on
# --------------------------------------------------------------------------


def test_prompt_ends_on_the_generation_marker(cfg, jinja_tokenizer):
    prompt = chat.render_prompt(jinja_tokenizer, QUESTION, "SYS")
    assert prompt == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        f"<|im_start|>user\n{QUESTION}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_a_completed_render_ends_on_exactly_one_eos(cfg, jinja_tokenizer):
    rendered = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, "SYS")
    assert rendered["text"].endswith(EOS_TOKEN)
    assert not rendered["text"].endswith(EOS_TOKEN * 2)
    # to_pref_row strips exactly this suffix so DPO (which appends EOS
    # unconditionally) and ORPO (which appends it only when absent) train on
    # identical text for identical pairs.
    assert rendered["text"][: -len(EOS_TOKEN)].endswith(COMPLETION)
    assert rendered["completion"] == COMPLETION + EOS_TOKEN


def test_prefix_invariant_holds_with_and_without_tools(cfg, jinja_tokenizer):
    plain = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, "SYS")
    assert plain["text"] == plain["prompt"] + plain["completion"]

    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": COMPLETION},
    ]
    tooled = chat.render_tool_example(jinja_tokenizer, messages, [SCHEMA])
    assert tooled["text"] == tooled["prompt"] + tooled["completion"]
    assert tooled["prompt"].endswith("<|im_start|>assistant\n")


def test_the_fake_tokenizer_still_models_the_shipped_template(cfg, jinja_tokenizer):
    """conftest's FakeTokenizer is a Python restatement of this file.

    Most of the suite renders through it, so a template change that leaves it
    behind makes those tests assert a string production never emits.
    """
    system = chat.compose_system(cfg)
    real = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, system)
    fake = chat.render_example(FakeTokenizer(), QUESTION, COMPLETION, system)
    assert real == fake


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def test_the_configured_masking_markers_are_what_the_template_emits(
    cfg, jinja_tokenizer
):
    """`train_on_responses_only` masks on these exact strings."""
    instruction = config_mod.get(cfg, "chat.instruction_part")
    response = config_mod.get(cfg, "chat.response_part")
    rendered = chat.render_example(jinja_tokenizer, QUESTION, COMPLETION, "SYS")
    assert instruction in rendered["prompt"]
    assert rendered["prompt"].endswith(response)
    # The question must sit inside the masked span and the answer outside it,
    # or the model is trained to generate its own questions.
    assert QUESTION in rendered["prompt"]
    assert QUESTION not in rendered["completion"]
    assert response not in rendered["completion"]


def test_tool_results_render_as_user_turns_so_they_stay_masked(cfg, jinja_tokenizer):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": QUESTION},
        tools.assistant_tool_call_message(
            [{"name": "get_fundamentals", "arguments": {"symbol": "BLK"}}]
        ),
        tools.tool_result_message("get_fundamentals", json.dumps({"duration": 6.4})),
        {"role": "assistant", "content": COMPLETION},
    ]
    rendered = chat.render_tool_example(jinja_tokenizer, messages, [SCHEMA])
    instruction = config_mod.get(cfg, "chat.instruction_part")
    assert f"{instruction}<tool_response>" in rendered["completion"]
    # Both assistant turns stay supervised; only the tool result is masked.
    assert tools.TOOL_CALL_OPEN in rendered["completion"]
    assert COMPLETION in rendered["completion"]


# --------------------------------------------------------------------------
# the shared wire contract
# --------------------------------------------------------------------------


def test_the_schema_block_matches_cosimo_tools_byte_for_byte(cfg, jinja_tokenizer):
    """`cosimo.tools.wire` is shared with dataset/pipelines/v3.

    The wire markers deliberately did NOT move with the template: they are
    ordinary text under Qwen, and keeping them is what lets a v3 agentic row, a
    supervised target and a served prompt stay the same string.
    """
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": QUESTION}],
        [SCHEMA],
    )
    expected = (
        tools.TOOL_SCHEMA_OPEN
        + tools.render_tool_schemas([SCHEMA])
        + tools.TOOL_SCHEMA_CLOSE
    )
    assert expected in rendered


def test_a_rendered_tool_call_matches_cosimo_tools_byte_for_byte(jinja_tokenizer):
    call = {"name": "get_fundamentals", "arguments": {"symbol": "BLK"}}
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [
            {"role": "user", "content": QUESTION},
            tools.assistant_tool_call_message([call]),
        ],
    )
    assert tools.render_tool_calls([call]) in rendered
    # And it parses back, which is what vLLM's hermes parser does at serving.
    assert tools.parse_tool_calls(rendered) == [call]


def test_tools_bound_without_a_system_turn_are_not_dropped(jinja_tokenizer):
    """The failure the Phi template was changed to fix, re-asserted for ChatML."""
    rendered = chat.render_conversation(
        jinja_tokenizer, [{"role": "user", "content": QUESTION}], [SCHEMA]
    )
    assert tools.TOOL_SCHEMA_OPEN in rendered
    assert "get_fundamentals" in rendered
    assert rendered.startswith("<|im_start|>system\n")


def test_schemas_are_bound_to_one_system_turn_only(jinja_tokenizer):
    rendered = chat.render_conversation(
        jinja_tokenizer,
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": QUESTION},
        ],
        [SCHEMA],
    )
    assert rendered.count(tools.TOOL_SCHEMA_OPEN) == 1
    assert rendered.count("<|im_start|>system") == 1
