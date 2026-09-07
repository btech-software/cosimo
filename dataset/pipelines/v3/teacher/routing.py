"""Record type -> teacher model + think flag (spec §5.4, one table).

"One client, two model names": a reasoning lane for the rows whose quality is
correctness (exam, implementation, agentic, critique) and a prose lane for the
rows whose quality is *voice* (analysis, memo, grounded). The names are
environment variables, not literals, because the model you can afford for the
80k-call run changes while the corpus is still byte-stable -- pinning the
choice in the deployment's env, and stamping the answer's model into every
row's ``verification.teacher``, is what keeps a mid-run teacher swap auditable
instead of a silent style drift (spec §12, axis 13).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .. import config


class RoutingError(ValueError):
    """No lane defined for this record type -- refuse, do not default."""


#: spec §5.4 verbatim; ``abstention`` rides the reasoning lane with think
#: off: an abstention's quality is *restraint*, and a chain of thought that
#: talks itself into answering is the failure mode, not a feature.
_LANES: dict[str, tuple[str, bool]] = {
    "exam": ("reasoning", True),
    "implementation": ("reasoning", True),
    "agentic": ("reasoning", True),
    "critique": ("reasoning", True),
    "abstention": ("reasoning", False),
    "analysis": ("prose", True),
    "memo": ("prose", True),
    "grounded": ("prose", True),
}


@dataclass(frozen=True)
class Route:
    record_type: str
    lane: str
    model: str
    think: bool


def default_models() -> dict[str, str]:
    return {
        "reasoning": os.environ.get(config.TEACHER_REASONING_ENV)
        or config.TEACHER_MODEL_REASONING_DEFAULT,
        "prose": os.environ.get(config.TEACHER_PROSE_ENV)
        or config.TEACHER_MODEL_PROSE_DEFAULT,
    }


def route(record_type: str, *, rejected: bool = False) -> Route:
    """The lane for ``record_type``; env is read per call, never captured.

    ``rejected=True`` is the preference stage's rejected side: same model as
    its parent type (spec §5.4's "same as parent type" -- the pair must not be
    separable by style), think forced off (rejected samples are cheap to
    produce and a thinking trace is not what's being taught).
    """
    if record_type not in _LANES:
        raise RoutingError(
            f"no teacher lane for record type {record_type!r} (known: {sorted(_LANES)})"
        )
    lane, think = _LANES[record_type]
    model = default_models()[lane]
    return Route(
        record_type=record_type,
        lane=lane,
        model=model,
        think=False if rejected else think,
    )
