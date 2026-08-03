"""The tool-calling wire format, in one place.

Everything that has to agree on how a tool call looks on the wire goes through
this module: ``configs/chat_template.jinja`` renders it at serving time,
``scripts/02_prepare_tool_data.py`` writes it as a supervised target at
training time, and ``tests/test_tools.py`` asserts the two produce byte-identical
strings. That equality is the whole point of the module -- a training target that
differs from the served rendering by even a space teaches a format the runtime
cannot parse back.

The format is Hermes-style ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``,
chosen because vLLM already ships ``--tool-call-parser hermes``; no custom parser
plugin is needed to turn generated text back into OpenAI ``tool_calls``.

The markers are ordinary text, not special tokens. Adding special tokens would
require resizing the embedding matrix, and ``lora.target_modules`` covers only
the projections, so the new rows would never receive a gradient.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Markers. TOOL_SCHEMA_* wrap the schema list inside the system turn and are real
# Phi-4 special tokens; TOOL_CALL_* / TOOL_RESPONSE_* are plain text.
TOOL_SCHEMA_OPEN = "<|tool|>"
TOOL_SCHEMA_CLOSE = "<|/tool|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


def _dumps(value: Any) -> str:
    """JSON exactly as the chat template's ``tojson`` filter emits it.

    transformers replaces Jinja's ``tojson`` with ``json.dumps(..., ensure_ascii=False)``
    and default separators. Matching it here is what keeps a rendered training
    target equal to a rendered serving prompt.
    """
    return json.dumps(value, ensure_ascii=False)


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    """One OpenAI-shaped function schema, the shape vLLM forwards to the template."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def render_tool_schemas(tools: list[dict]) -> str:
    """The schema list as it appears between the ``<|tool|>`` markers."""
    return _dumps(tools)


def render_tool_calls(calls: list[dict]) -> str:
    """Serialise ``[{"name", "arguments"}]`` as the supervised completion text.

    Must stay byte-identical to the chat template's assistant-turn rendering;
    ``tests/test_tools.py::test_rendered_tool_call_matches_the_template`` pins it.
    """
    parts = []
    for call in calls:
        name = _dumps(str(call["name"]))
        arguments = call.get("arguments", {})
        # A string is already-serialised JSON (the OpenAI wire shape); a mapping
        # is the local shape and still needs dumping. The template branches the
        # same way on `fn_args is string`.
        payload = arguments if isinstance(arguments, str) else _dumps(arguments)
        parts.append(
            f'{TOOL_CALL_OPEN}{{"name": {name}, "arguments": {payload}}}'
            f"{TOOL_CALL_CLOSE}"
        )
    return "".join(parts)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract ``[{"name", "arguments"}]`` from generated text.

    Tolerates surrounding prose, which is what the model actually produces when
    it narrates before calling. Malformed JSON inside the markers is skipped
    rather than raised: callers counting well-formed calls want the count, not an
    exception.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            calls.append(
                {
                    "name": payload["name"],
                    "arguments": payload.get("arguments", {}),
                }
            )
    return calls


def assistant_tool_call_message(calls: list[dict], content: str = "") -> dict:
    """An assistant turn that calls tools, in the shape the template consumes."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                },
            }
            for call in calls
        ],
    }


def tool_result_message(name: str, content: str) -> dict:
    """A tool-result turn. Renders as a ``<|user|>`` turn (see the chat template)."""
    return {"role": "tool", "name": name, "content": content}


def render_tool_result(name: str, content: str) -> str:
    """The tool-result payload as it appears between the ``<tool_response>`` markers."""
    return _dumps({"name": name, "content": content})
