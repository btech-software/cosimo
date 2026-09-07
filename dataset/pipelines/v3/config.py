"""Control-plane constants for the v3 corpus: ids, caps, paths, env names.

Everything that *decides* what a row looks like or how much of it ships lives
here, so an audit can read the rules without reading the renderers. Values are
the ones fixed in ``docs/prompts/rev_03/COSIMO_V3_ARCHITECTURE.md`` (§4 ids,
§5 inventory caps, §6 verification shares, §9 env).

No wall clock is ever consulted: ``out_dir`` and the env accessors read process
environment at call time, which is configuration, not data -- the bytes written
into a shard depend only on (work_type, family, record_type, variant, seed).
"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # dataset/

SCHEMA_VERSION = "v3.0"
SHARD_SIZE = 500

# Id namespaces are disjoint by prefix, and verify_v3 enforces it (spec §4).
SUPERVISED_ID_PREFIX = "cosimov3"
PREFERENCE_ID_PREFIX = "cosimov3pref"

# Publish gate shares (verify_v3 axes 6-8) and the plan-time family cap (§5.1).
EXAM_SHARE_BAND = (0.12, 0.18)
LITURGY_CAP = 0.25
FAMILY_MAX_SHARE = 0.03

# The registers a renderer may target; a row whose pack register is not one of
# these was composed by something that was not the register contract.
VALID_REGISTERS = ("desk_chat", "ic_memo", "risk_committee", "auditor", "code_review")

TEACHER_BASE_URL_ENV = "TEACHER_BASE_URL"
TEACHER_API_KEY_ENV = "TEACHER_API_KEY"
TEACHER_REASONING_ENV = "TEACHER_REASONING"
TEACHER_PROSE_ENV = "TEACHER_PROSE"
TEACHER_TIMEOUT_ENV = "TEACHER_TIMEOUT_S"
DEFAULT_TEACHER_TIMEOUT_S = 120
OUT_ENV = "COSIMO_V3_OUT"
LIVE_ENV = "COSIMO_V3_LIVE"


def out_dir() -> str:
    """Absolute root of the v3 shard tree (spec §12: never CWD-relative)."""
    return os.path.abspath(
        os.environ.get(OUT_ENV) or os.path.join(BASE_DIR, "shards", "v3")
    )


def taxonomy_path() -> str:
    """Absolute path of the plan file the inventory expands."""
    return os.path.abspath(os.path.join(BASE_DIR, "taxonomy", "work_types.yaml"))


def live_enabled() -> bool:
    """True only when the operator opted into real teacher calls (spec §10 PR2)."""
    return os.environ.get(LIVE_ENV) == "1"


def supervised_id(record_type: str, seed: int) -> str:
    return f"{SUPERVISED_ID_PREFIX}_{record_type}_{seed:016x}"


def preference_id(seed: int) -> str:
    return f"{PREFERENCE_ID_PREFIX}_{seed:016x}"
