"""Record type -> teacher model + think flag + budget (spec §5.4, one table).

"One client, two model names": a reasoning lane for the rows whose quality is
correctness (exam, implementation, agentic, critique) and a prose lane for the
rows whose quality is *voice* (analysis, memo, grounded). The names are
environment variables, not literals, because the model you can afford for the
80k-call run changes while the corpus is still byte-stable -- pinning the
choice in the deployment's env, and stamping the answer's model into every
row's ``verification.teacher``, is what keeps a mid-run teacher swap auditable
instead of a silent style drift (spec §12, axis 13).

**Think is off by default now, and that is the amendment's change.** The
original table gave analysis, memo and grounded ``think=True`` on the prose
lane, which was three mistakes at once: it bought a chain of thought for rows
whose quality is voice rather than derivation, it made the answer text the
*tail* of a completion whose budget the reasoning had already eaten (the
16,384-token default and the 900-second timeout in ``dataset_build.sh`` were
both consequences of that, not causes), and it put think text in front of a
renderer that had to strip it back out. A lane that does not think needs no
16k budget and no 900s deadline; the two caps below are what replaces them.

The one lane where thinking is still plausibly worth its cost is ``memo`` --
the longest form in the corpus, the only one with real structure to plan --
and it is a *measurement*, not a preference: ``COSIMO_V3_MEMO_THINK=1`` turns
it on, and ``dataset/progress/v3_think_ablation.md`` is where the 20-row
bake-off that would justify making it the default gets recorded. The
amendment's bar is explicit: promote think-on only if the invented-number rate
drops by two points or more.
"""

from __future__ import annotations

from dataclasses import dataclass

import os

from .. import config


class RoutingError(ValueError):
    """No lane defined for this record type -- refuse, do not default."""


#: ``record_type -> (lane, think)``, amendment §B verbatim.
#:
#: ``abstention`` rides the *prose* lane now, not the reasoning one: an
#: abstention's quality is restraint, a chain of thought that talks itself into
#: answering is its failure mode, and a lane whose think flag is off has no
#: business paying reasoning-model prices for the privilege.
#:
#: ``agentic`` is the one entry this table cannot state alone -- it thinks on
#: the planner turn and not on the tool-call turns -- so it carries the
#: planner's flag here and :func:`agentic_turn_think` decides the rest.
_LANES: dict[str, tuple[str, bool]] = {
    "exam": ("reasoning", True),
    "critique": ("reasoning", True),
    "implementation": ("reasoning", True),
    "agentic": ("reasoning", True),
    "abstention": ("prose", False),
    "analysis": ("prose", False),
    "grounded": ("prose", False),
    "memo": ("prose", False),
}

#: The lanes whose think flag an operator may still move, and the accessor that
#: decides. Everything absent from this mapping is table law.
_THINK_OVERRIDES = {"memo": config.memo_think}


@dataclass(frozen=True)
class Route:
    record_type: str
    lane: str
    model: str
    think: bool
    #: The completion budget for one call on this lane. A property of the lane
    #: rather than of the client, because it is the *think flag* that decides
    #: how much of a budget is spent before the first answer token: 800 buys a
    #: 400-word ceiling with room to spare when nothing reasons first, and 2048
    #: is real headroom when something does.
    max_tokens: int


def default_models() -> dict[str, str]:
    return {
        "reasoning": os.environ.get(config.TEACHER_REASONING_ENV)
        or config.TEACHER_MODEL_REASONING_DEFAULT,
        "prose": os.environ.get(config.TEACHER_PROSE_ENV)
        or config.TEACHER_MODEL_PROSE_DEFAULT,
    }


def budget_for(think: bool) -> int:
    """The completion cap for a call with this think flag (amendment §B).

    The amendment's two numbers plus the deployment's reasoning overhead, which
    is zero unless an operator declares it. §B's caps assume the flag decides
    whether a chain of thought is produced at all; against a teacher that
    reasons unconditionally they cap the *reasoning* and the answer never
    arrives -- measured, on the reference box, as forty empty drafts and forty
    `finish_reason: length`. See ``config.THINK_OVERHEAD_ENV``.
    """
    base = config.MAX_TOKENS_THINK_ON if think else config.MAX_TOKENS_THINK_OFF
    return base + config.think_overhead()


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
    override = _THINK_OVERRIDES.get(record_type)
    if override is not None:
        think = override()
    if rejected:
        think = False
    return Route(
        record_type=record_type,
        lane=lane,
        model=default_models()[lane],
        think=think,
        max_tokens=budget_for(think),
    )


def agentic_turn_think(route_: Route, *, tool_results_seen: bool) -> bool:
    """Whether *this* agentic turn thinks (amendment §B's split entry).

    The planner turn -- the first ask, before any tool has answered -- is where
    a trajectory is decided, and it is worth a chain of thought. Every turn
    after a tool result is a *reaction* to bytes the oracle just handed over;
    reasoning there re-derives a plan that is already fixed, and the corpus
    pays for it on every hop of every conversation.
    """
    return route_.think and not tool_results_seen
