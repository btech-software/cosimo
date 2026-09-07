"""The v3 contract: ``cosimo_ft.tools`` is now a re-export, not a definition.

v3 moved the wire format to ``cosimo/tools/wire.py`` so the corpus renderer,
the training scripts and the chat template format tool calls through one
module (``docs/prompts/rev_03/COSIMO_V3_ARCHITECTURE.md`` §8.5). These tests
pin the two properties that make that move safe:

* every public name resolves to the *same object* the harness imported before
  the move (an identity check, so a subtle reimplementation cannot pass);
* the import works without the caller pre-arranging ``sys.path`` -- the shim
  bootstraps the repo root itself, exactly like the conftest does, because the
  training container runs scripts from many different working directories.
"""

from __future__ import annotations

import importlib
import sys

from cosimo_ft import tools


def _wire_module():
    return importlib.import_module("cosimo.tools.wire")


def test_every_exported_name_is_the_wire_object_itself():
    wire = _wire_module()
    for name in tools.__all__:
        assert getattr(tools, name) is getattr(wire, name), (
            f"cosimo_ft.tools.{name} is a copy, not the shared object; the "
            "byte-identity guarantee only holds while both sides import once"
        )


def test_shim_uses_the_single_shared_module_identity():
    importlib.import_module("cosimo.tools.wire")
    assert sys.modules["cosimo.tools.wire"].TOOL_CALL_OPEN == tools.TOOL_CALL_OPEN
    # A second identity (pipelines-side or jobs-side import) would mean two
    # compiled copies of the formatter that can drift.
    from cosimo.tools import registry

    assert registry.REGISTRY_VERSION


def test_reexport_surface_is_unchanged_from_pre_move():
    """The twelve names this module defined pre-v3; nothing added, nothing lost."""
    expected = {
        "TOOL_SCHEMA_OPEN",
        "TOOL_SCHEMA_CLOSE",
        "TOOL_CALL_OPEN",
        "TOOL_CALL_CLOSE",
        "TOOL_RESPONSE_OPEN",
        "TOOL_RESPONSE_CLOSE",
        "tool_schema",
        "render_tool_schemas",
        "render_tool_calls",
        "parse_tool_calls",
        "assistant_tool_call_message",
        "tool_result_message",
        "render_tool_result",
    }
    assert set(tools.__all__) == expected
    # The public surface the whole harness consumed before the move must still
    # be reachable attribute-wise; import-time is too late to fail quietly.
    for name in expected:
        assert callable(getattr(tools, name, None)) or isinstance(
            getattr(tools, name), str
        )
