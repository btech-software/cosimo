"""The tool oracle: the server's answers, reconstructed from the fact pack.

Two halves, and the split is the spec's (§5.4, §5.8):

* :mod:`.runtime` -- the honest machine. Registry tools answered from pack
  values only; it never performs new arithmetic, because every figure was
  computed once, at pack time, from seeded inputs, and a second derivation
  somewhere else is exactly how a corpus starts teaching two answers to one
  question.
* :mod:`.faults` -- the dirty layer. Which jobs see a degraded response
  (exact by variant residue, replayable on any machine) and the five shapes
  that degradation takes: empty, stale, wrong-ticker, rate-limited, drifted.

Together they make the agentic corpus *verifiable* rather than merely
generated: a stored conversation is a claim about what the oracle said, and
the claim is audited by running the oracle again.
"""

from __future__ import annotations

from . import faults, runtime
from .faults import (
    FAULTS,
    FAULT_ACK,
    STALE_AS_OF,
    Schedule,
    canonical,
    inject,
    schedule_of,
)
from .runtime import Call, OracleError, tools_for, execute

__all__ = [
    "FAULTS",
    "FAULT_ACK",
    "STALE_AS_OF",
    "Call",
    "OracleError",
    "Schedule",
    "canonical",
    "execute",
    "faults",
    "inject",
    "runtime",
    "schedule_of",
    "tools_for",
]
