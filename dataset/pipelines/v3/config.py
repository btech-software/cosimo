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

# The exam slice (analysis spec §5.10). The liturgy is not left to the
# sampler's luck: it is a residue class of the variant -- 1 in
# EXAM_LITURGY_MODULUS, strictly under LITURGY_CAP -- so no pre-thinning
# slice can overshoot the cap, and the board still measures the shipped
# corpus rather than trusting the sampler. A measured overshoot after
# pack-guard thinning is itself a finding: the slice thinned unevenly.
EXAM_LITURGY_MODULUS = 5
EXAM_LITURGY_RESIDUE = 4
#: §5.10: a distractor that cannot be made distinct is dropped, never
#: inflated ("do not add 7.0"); an item that cannot reach MIN distinct
#: distractors is no item at all and is dead-lettered, not shipped thin.
EXAM_MIN_DISTRACTORS = 2
EXAM_MAX_DISTRACTORS = 3
#: Share-support floors for the corpus-level axes 6-8: below these counts a
#: share is a ratio of noise, and a gate that measured noise would either
#: shade red forever or be switched off. The axes therefore *report* on thin
#: samples and certify -- can shade -- only on samples with support; the
#: certification site for the full corpus is the publish gate.
SHARE_MIN_ROWS = 200
SHARE_MIN_PER_FAMILY = 30
SHARE_MIN_EXAM_ROWS = 50
#: The emitted-corpus reading of the §5.1 family cap (see the axis-7 note on
#: the board): no family's share may exceed the leanest family's by more
#: than this factor, which is the dominance the cap exists to prevent, read
#: at a scale where ten stems can actually be balanced.
FAMILY_BALANCE_TOLERANCE = 1.25

# The implementation slice (analysis spec §5.9). ``limitations`` is the one
# authored field of the record and it is fact-locked like any prose that
# touches numbers -- §5.6's "temperature 0 on anything allowed to touch
# numbers that are not already in the pack" -- so the ladder is nearly cold
# and short: a second attempt at 0.1 is a different sample, a third would be
# hoping. ``SANDBOX_TIMEOUT_S`` bounds what a hung instrument can cost the
# board that re-executes it; a reference is a pure function, so five seconds
# is minutes of slack, not a budget.
IMPL_ATTEMPTS = 2
IMPL_TEMPERATURES = (0.1, 0.0)
SANDBOX_TIMEOUT_S = 5.0
#: A limitation statement shorter than this is not a limitation; it is a
#: shrug in a field (§5.9: the field must state what the record omits).
MIN_LIMITATION_WORDS = 8

# The registers a renderer may target; a row whose pack register is not one of
# these was composed by something that was not the register contract.
VALID_REGISTERS = ("desk_chat", "ic_memo", "risk_committee", "auditor", "code_review")

# Prose repair ladder (analysis spec §6.3 item 8): attempt one at the routed
# temperature, then cool it -- a violation is usually the model padding, and
# cold models pad less. Three strikes and the row is dead-lettered, never
# silently shipped.
PROSE_ATTEMPTS = 3
PROSE_TEMPERATURES = (0.7, 0.3, 0.1)

# The agentic loop's budgets (analysis spec §5.8: "a real multi-step loop,
# 6-16 turns"). Measured over the *non-system* messages, so the band bounds
# what a training run pays in tokens per conversation, not the boilerplate.
# ``AGENTIC_MAX_TOOL_CALLS`` is the hard stop the loop enforces before asking
# again; one call that returns an error is worth re-issuing, ten is a model
# arguing with itself.
AGENTIC_MIN_MESSAGES = 6
AGENTIC_MAX_MESSAGES = 16
AGENTIC_MAX_TOOL_CALLS = 6

#: The mix, exact by residue of the variant (see ``oracle.faults.schedule_of``):
#: every fifth agentic job answers from the pack alone ("calling a tool is
#: waste when the fact pack is already in the user message" -- spec §5.8), and
#: every fifth *other* one walks into an injected fault. Two residues of ten,
#: so each slice is exactly 20% -- inside the spec's 15-20% band, and exact
#: counts beat a dice roll because the 20-row PR3 gate can assert them.
AGENTIC_STRIDE = 5
AGENTIC_NO_CALL_RESIDUE = 0
AGENTIC_FAULT_RESIDUE = 2

#: The agentic repair ladder sits colder than the prose one: a teacher that
#: is creative about *which tool to call* is a teacher that ships trajectories
#: the server cannot replay. Same three strikes as prose (a verdict about this
#: teacher on this brief), different temperatures because tool calling is
#: meaning, not voice.
AGENTIC_ATTEMPTS = 3
AGENTIC_TEMPERATURES = (0.3, 0.2, 0.1)

# The analysis spec's "tiny whitelist": quantities that are arithmetic
# furniture rather than pack facts -- the percent denominator, the two of a
# two-way bridge, the trading-day convention -- plus every number spelled in
# the pack's own as-of date. Token-level (the string as the model wrote it),
# shared by the render gate and the verify board so both read the same mercy.
NUMBER_WHITELIST = ("100", "2", "252")

TEACHER_BASE_URL_ENV = "TEACHER_BASE_URL"
TEACHER_API_KEY_ENV = "TEACHER_API_KEY"
TEACHER_REASONING_ENV = "TEACHER_REASONING"
TEACHER_PROSE_ENV = "TEACHER_PROSE"
TEACHER_TIMEOUT_ENV = "TEACHER_TIMEOUT_S"
TEACHER_FIXTURE_ENV = "COSIMO_V3_TEACHER_FIXTURE"
DEFAULT_TEACHER_TIMEOUT_S = 120
OUT_ENV = "COSIMO_V3_OUT"
LIVE_ENV = "COSIMO_V3_LIVE"

#: The two names of spec §5.4's "one client, two model names". Deployment
#: overrides via TEACHER_REASONING / TEACHER_PROSE; these are the defaults a
#: fresh box gets, and every row's verification.teacher block records which
#: one actually answered, so a mid-run swap is visible in the data.
TEACHER_MODEL_REASONING_DEFAULT = "deepseek-v4-flash"
TEACHER_MODEL_PROSE_DEFAULT = "qwen3.8-flash-next"


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


def default_teacher_fixture() -> str:
    """The committed replay file every offline teacher call falls back to.

    Lives with the test suite, not under an out-dir: a fixture is source. The
    harness that generated it is ``tests/v3/fixtures/make_teacher_echo.py``,
    and a drift test pins the file against that harness -- replay data must be
    reproducible or it is folklore (see §7 of the analysis spec on gold bars
    that nobody can regenerate).
    """
    return os.path.abspath(
        os.path.join(BASE_DIR, "tests", "v3", "fixtures", "teacher_echo.json")
    )


def supervised_id(record_type: str, seed: int) -> str:
    return f"{SUPERVISED_ID_PREFIX}_{record_type}_{seed:016x}"


def preference_id(seed: int) -> str:
    return f"{PREFERENCE_ID_PREFIX}_{seed:016x}"
