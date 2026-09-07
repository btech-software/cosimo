"""The tool-calling wire format, re-exported from the shared contract package.

v3 moved the single owner of this contract to ``cosimo.tools.wire`` so that
generation, training and serving import the same objects (spec §8.5,
``docs/prompts/rev_03/COSIMO_V3_ARCHITECTURE.md``). The names below are exactly
the ones this module defined before the move, so ``tests/test_tools.py`` keeps
pinning that ``cosimo_ft.tools`` and ``configs/chat_template.jinja`` render
byte-identically -- and the corpus can no longer describe a tool call in a
dialect the served chat template cannot parse back.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared contract package lives at the repository root, above the harness.
# Bootstrap it the way every other cross-tree import in this repo does.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cosimo.tools.wire import (  # noqa: E402
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_SCHEMA_CLOSE,
    TOOL_SCHEMA_OPEN,
    assistant_tool_call_message,
    parse_tool_calls,
    render_tool_calls,
    render_tool_result,
    render_tool_schemas,
    tool_result_message,
    tool_schema,
)

__all__ = [
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "TOOL_RESPONSE_CLOSE",
    "TOOL_RESPONSE_OPEN",
    "TOOL_SCHEMA_CLOSE",
    "TOOL_SCHEMA_OPEN",
    "assistant_tool_call_message",
    "parse_tool_calls",
    "render_tool_calls",
    "render_tool_result",
    "render_tool_schemas",
    "tool_result_message",
    "tool_schema",
]
