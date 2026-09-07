"""name → schema registry for the canonical tool surface, versioned.

Three consumers, one source of truth (spec §3 of the v3 architecture doc):

* ``dataset/pipelines/v3`` validates every proposed ``tool_call`` in an agentic
  render against :func:`validate_call` and has the oracle execute only names
  that resolve here;
* ``jobs/fine-tune/cosimo_ft`` renders the schemas into the supervised prompt
  through :func:`render_tools`, the same bytes the served chat template embeds;
* the serving app registers the same names.

``REGISTRY_VERSION`` is stamped into generated rows so a schema edit is an
explicit, inspectable event rather than silent drift between a training
corpus and a running server.
"""

from __future__ import annotations

from .schemas import TOOLS
from .wire import render_tool_schemas

REGISTRY_VERSION = "v3.0"

SCHEMAS: dict[str, dict] = TOOLS


def names() -> list[str]:
    """Tool names in registry order."""
    return list(SCHEMAS)


def render_tools(tool_names: list[str] | None = None) -> str:
    """The selected schemas as the JSON list the chat template embeds.

    ``tool_names`` defaults to the whole registry; selection keeps registry
    order so the rendering depends on the set, not on the order a caller
    happened to iterate it. An unknown name is an error, not a silent skip -- a
    silently dropped schema trains a prompt the server never builds.
    """
    if tool_names is None:
        selected = list(SCHEMAS.values())
    else:
        wanted = set(tool_names)
        unknown = sorted(wanted - set(SCHEMAS))
        if unknown:
            raise ValueError(f"tools not in the registry: {unknown}")
        selected = [SCHEMAS[n] for n in SCHEMAS if n in wanted]
    return render_tool_schemas(selected)


_TYPE_FORMS = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _matches(value: object, type_name: str) -> bool:
    forms = _TYPE_FORMS.get(type_name)
    if forms is None:
        return True  # a type the validator does not know is not a veto
    if isinstance(value, bool) and "boolean" not in type_name:
        return False  # bool is a subclass of int; it is never a number here
    return isinstance(value, forms)


def validate_call(name: str, arguments: object) -> list[str]:
    """Every way a proposed tool call can violate its registered schema.

    Returns the error strings; an empty list is a valid call. Errors are
    collected rather than raised because the agentic loop feeds them back to
    the teacher for one corrective turn before the row dies.
    """
    schema = SCHEMAS.get(name)
    if schema is None:
        return [f"unknown tool {name!r}"]
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return [
            f"arguments for {name!r} must be an object, got {type(arguments).__name__}"
        ]

    parameters = schema.get("function", {}).get("parameters", {}) or {}
    properties = parameters.get("properties", {}) or {}
    required = parameters.get("required", []) or []

    errors: list[str] = []
    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument {key!r} for {name}")
    for key, value in arguments.items():
        if key not in properties:
            errors.append(f"unknown argument {key!r} for {name}")
            continue
        declared = (properties[key] or {}).get("type")
        if declared and not _matches(value, declared):
            errors.append(
                f"argument {key!r} for {name} must be {declared}, "
                f"got {type(value).__name__}"
            )
    return errors
