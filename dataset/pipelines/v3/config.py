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

# The preference slice (analysis spec §5.7, arch spec §5.6). Pair generation
# is a second pass over *shipped* SFT rows, so its ladders live beside the
# prose one it repairs against: the chosen side is a paraphrase (temperature
# 0.7 per §5.6) that must clear the whole prose gate, and one cold retry is
# the difference between a pair and a dead letter. ``PREF_MAX_SHINGLE_OVERLAP``
# is the acceptance criterion 5 ("chosen not-equal SFT target, n-gram overlap
# below a set threshold") with the threshold actually set: 0.6 on the Jaccard
# index of 8-word shingles is far enough from 1.0 that v1's near-copy rejected
# sides -- chosen with one number changed -- could never pass for a pair.
PREF_ATTEMPTS = 2
PREF_TEMPERATURES = (0.7, 0.3)
PREF_SHINGLE_WORDS = 8
PREF_MAX_SHINGLE_OVERLAP = 0.6
#: The rejected side is not run through the prose gate (its crime is the
#: point); fluency is a word-count band, deliberately generous, deliberately
#: narrow: a shrug is not a training signal and a thousand words is a novel.
PREF_MIN_REJECTED_WORDS = 20
PREF_MAX_REJECTED_WORDS = 400
#: §5.7 verbatim: the per-type probability a shipped row is paired.
PREF_PROBABILITIES = {
    "analysis": 0.25,
    "memo": 0.25,
    "critique": 0.25,
    "grounded": 0.25,
    "abstention": 0.40,
}

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
# furniture rather than pack facts -- the multiplicative identity, the percent
# denominator, the two of a two-way bridge, the trading-day convention -- plus
# every number spelled in the pack's own as-of date. Token-level (the string as
# the model wrote it), shared by the render gate and the verify board so both
# read the same mercy.
#
# "1" earns its place the way "10000" did, by a measurement: a 170-word desk
# answer that got every figure from the pack, covered every contract point and
# named its own hidden assumption was dead-lettered for inventing the number 1
# -- from `296.61 * (1 - 28.16 / 10000)`, the identity that turns a basis-point
# cost into a fill price. A gate that rejects the best answer it has yet seen,
# over notation that asserts nothing about the world, is measuring the wrong
# thing. The cost is real and accepted: a teacher may now write "1 basis point"
# uncaught. That is a weaker claim than the "252" already forgiven here, and
# the alternative is selecting against arithmetic the desk actually writes.
NUMBER_WHITELIST = ("1", "100", "2", "252", "10000", "10,000")
#: 10000 joined the list after the first successful live render. The packs'
#: own formulas convert to basis points with `1e4`, but the whitelist did not
#: carry it and the tokenizer cannot read scientific notation -- `1e4` scans as
#: the two tokens `1` and `4`, so even the honest spelling was a violation.
#: The teacher, boxed in, wrote the conversion as `1.27% x sqrt(0.010197) x
#: 10.0 x 10.0 x 10.0 x 10.0` -- legal, because 10.0 happened to be a pack
#: value, and unreadable. A gate that makes a correct answer ugly is measuring
#: the wrong thing. Both spellings are listed because matching is on the token
#: as written, commas and all.

TEACHER_BASE_URL_ENV = "TEACHER_BASE_URL"
TEACHER_API_KEY_ENV = "TEACHER_API_KEY"
TEACHER_REASONING_ENV = "TEACHER_REASONING"
TEACHER_PROSE_ENV = "TEACHER_PROSE"
TEACHER_TIMEOUT_ENV = "TEACHER_TIMEOUT_S"
#: Completion budget per call. Overridable because it is a property of the
#: *teacher*, not of the corpus: a reasoning model bills its chain of thought
#: against this budget before it emits a single answer token, and the first
#: live run died entirely on that -- deepseek-v4-flash spent 4,851 completion
#: tokens to produce a 109-word answer, so at the old 1024 it returned
#: `content: null` with `finish_reason: length` on every attempt, three
#: attempts a row, forever.
TEACHER_MAX_TOKENS_ENV = "COSIMO_V3_MAX_TOKENS"
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

# -- Publish gate (spec §6 axes 12-13) and the gold bar (§5.11) --------------
#: The human gold bar the train corpus is fenced against for near-duplication.
#: It is *not* generator output -- a charterholder/desk reviewed set -- and the
#: v3 fence is the same Jaccard shingle instrument the preference lane and
#: ``suite_overlap`` trust. A train row that reads this close to a gold item is
#: a leak of the eval set into training, not diversity, so the bar and the
#: threshold are both control-plane, not caller-chosen.
GOLDBAR_FILENAME = "gold_bar_v3.jsonl"
GOLDBAR_NEAR_DUP_THRESHOLD = 0.6
#: The bar is a source artefact, so the path is a control-plane value: the
#: default is the commited human file, but CI and the tests point this at a
#: synthetic one to exercise the fence -- which is exactly why the gate checks
#: *presence* rather than trusting the path to be the real bar.
GOLDBAR_ENV = "COSIMO_V3_GOLDBAR"

#: Teacher provenance is pinned per row (``verification.teacher.model``); a
#: corpus that mixes models is a mistake until an operator says it is on
#: purpose. This env lists, comma-separated, the models whose mixing is
#: declared intentional; absent or empty means exactly one teacher may appear,
#: and the board need not be asked to allow a second style by silence.
TEACHER_ALLOWLIST_ENV = "COSIMO_V3_TEACHER_ALLOWLIST"

#: The publish card and the Hub id it carries. Absolute paths only (spec §7):
#: a CWD-relative card is the v2 bug where CI "publishes" into a temp dir and
#: passes by publishing nothing.
PUBLISH_DIR_ENV = "COSIMO_V3_PUBLISH_DIR"
HUB_REPO_ID_ENV = "COSIMO_V3_HUB_REPO"
DATASET_CARD_FILENAME = "dataset_card_v3.md"


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


def gold_bar_v3_path() -> str:
    """Absolute path of the human v3 gold bar the publish gate fences against.

    Source, not out-dir artefact: like the replay fixtures the bar must be a
    commited, reviewable file, and it lives beside the frozen v2
    ``gold_bar.jsonl`` as its *disjoint* successor -- spec §5.11 is emphatic
    that it is not generator output, which is why the gate reads a path the
    operator points at rather than synthesising one.
    """
    return os.path.abspath(
        os.environ.get(GOLDBAR_ENV)
        or os.path.join(BASE_DIR, "goldbar", GOLDBAR_FILENAME)
    )


def publish_dir() -> str:
    """Where ``publish`` writes the v3 dataset card (spec §7: absolute, never CWD)."""
    return os.path.abspath(
        os.environ.get(PUBLISH_DIR_ENV) or os.path.join(BASE_DIR, "publish")
    )


def teacher_allowlist() -> frozenset[str]:
    """The teacher models whose mixing the operator has declared intentional.

    Read per call like every other control-plane value -- a corpus is
    certified against the environment of the box that certified it -- and a
    comma-separated list is the whole syntax. An absent env is the empty set,
    the strictest reading: one teacher throughout, or the mix is not allowed.
    """
    raw = os.environ.get(TEACHER_ALLOWLIST_ENV, "")
    return frozenset(token.strip() for token in raw.split(",") if token.strip())


def hub_repo_id() -> str | None:
    """The Hub repo id the dataset card records, or None when it is not set."""
    return os.environ.get(HUB_REPO_ID_ENV) or None


def supervised_id(record_type: str, seed: int) -> str:
    return f"{SUPERVISED_ID_PREFIX}_{record_type}_{seed:016x}"


def preference_id(seed: int) -> str:
    return f"{PREFERENCE_ID_PREFIX}_{seed:016x}"
