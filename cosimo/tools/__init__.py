"""Shared tool contracts: the wire format and the canonical schema registry.

This package is the one exception to the corpus/harness separation ("joined by
published artifacts, not by imports"): tool schemas and the tool-calling wire
format must be the *same objects* in generation, training and serving, or they
drift -- which is how v2 shipped agentic rows describing tool calls in a shape
the served chat template could not parse back.

Nothing here imports the serving app (``agent_lab``, ``cosimo.main``); the
package is stdlib-only so the CPU corpus image can import it.
"""

from __future__ import annotations

from .registry import REGISTRY_VERSION, SCHEMAS, names, render_tools, validate_call
from .schemas import TOOLS

__all__ = [
    "REGISTRY_VERSION",
    "SCHEMAS",
    "TOOLS",
    "names",
    "render_tools",
    "validate_call",
]
