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
#: The anti-dominance headroom the family cap grants over an even split. A flat
#: 0.03 was advertised here and it was arithmetic that only ever worked on a
#: plan with fifteen-plus train families: with five work types and ten train
#: families a 3% cap cannot be met by *each* family and also sum to 100%, so the
#: number was a promise the expansion could not keep and the manifest printed a
#: "max realised share" three times the cap on every run. The cap's real job is
#: to stop one easy family owning half the pool, so it is stated as what it is
#: -- an even split plus a quarter -- and derived from the plan that is actually
#: loaded (:func:`family_max_share`). Add work types and it tightens by itself.
FAMILY_MAX_SHARE_HEADROOM = 1.25

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
#: How many times one row may be handed more room before its budget failure is
#: called a verdict. Separate from PROSE_ATTEMPTS because they measure
#: different things: an attempt is a try at the *contract*, a truncation retry
#: is a try at the *budget*, and conflating them spends a row's contract
#: allowance discovering a number the process could have learned once.
#: Four doublings take 800 to 12,800, which covers the measured spread.
PROSE_TRUNCATION_RETRIES = 4

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

# The deepest a *written* figure may go in prose, regardless of how the pack
# spells it. An absolute ceiling rather than a relative one, because the packs
# themselves publish raw division results: 1,522 of the figures on disk carry
# 9-12 decimal places, and a rule of "no deeper than the pack printed it" would
# wave through an answer quoting a portfolio weight as 0.472041725693. Six is
# the deepest any figure legitimately needs -- a participation rate (0.017381,
# 1.74% of ADV) -- and every figure past that is a float that escaped rounding,
# not a measurement.
#
# This is a presentation rule, not a correctness one: the invented-number gate
# still decides whether the VALUE is the pack's. This decides whether the desk
# would have written it that way. A reader who is handed twelve decimals is
# being told the book is known to a picometre.
PROSE_MAX_DECIMALS = 6

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
#: The amendment widens the band from the three hand-picked small integers to
#: 0-10 whole. A single digit in a desk answer is an ordinal, a count of
#: sectors, a horizon in days or an arithmetic identity far more often than it
#: is a quantity about the world, and each one that was *not* listed cost a
#: correct answer its row (see the "1" note above, which was the third such
#: measurement). Everything above ten still has to come from the pack.
NUMBER_WHITELIST = (
    *(str(n) for n in range(11)),
    "100",
    "252",
    "10000",
    "10,000",
)
#: 10000 joined the list after the first successful live render. The packs'
#: own formulas convert to basis points with `1e4`, but the whitelist did not
#: carry it and the tokenizer cannot read scientific notation -- `1e4` scans as
#: the two tokens `1` and `4`, so even the honest spelling was a violation.
#: The teacher, boxed in, wrote the conversion as `1.27% x sqrt(0.010197) x
#: 10.0 x 10.0 x 10.0 x 10.0` -- legal, because 10.0 happened to be a pack
#: value, and unreadable. A gate that makes a correct answer ugly is measuring
#: the wrong thing. Both spellings are listed because matching is on the token
#: as written, commas and all.

# -- Two surfaces (amendment §A) --------------------------------------------
#: The teacher transcript is *debug*, not corpus. Off by default: a shard tree
#: that carries the factory's own system turn is a shard tree whose next reader
#: trains on it, which is precisely how ``01_prepare_data`` ended up teaching
#: the labelling protocol. Set to ``1`` and every render also drops a
#: ``teacher_logs/<kind>/<id>.json`` beside the shard for a post-mortem to read.
KEEP_TEACHER_MESSAGES_ENV = "COSIMO_V3_KEEP_TEACHER_MESSAGES"
#: The fingerprint of a factory brief, in one string. It appears verbatim in
#: :data:`teacher.prompts.TEACHER_SYSTEM` and ``AGENTIC_SYSTEM``, it is the
#: thing a student row may never carry, and both the render-time gate and the
#: harness's prepare gate match on *this* constant rather than on two copies of
#: a substring that could drift apart.
TEACHER_FINGERPRINT = "Cosimo v3 teacher"

# -- Think policy (amendment §B) --------------------------------------------
#: ``memo`` is the one lane whose think flag is an operator decision rather
#: than a table entry: an IC memo is the longest form in the corpus and the
#: only one where a chain of thought plausibly buys structure. It stays off
#: until a bake-off says otherwise (see ``dataset/progress/v3_think_ablation.md``).
MEMO_THINK_ENV = "COSIMO_V3_MEMO_THINK"
#: Completion budgets per lane, replacing the flat 16384. The old number was
#: sized for a reasoning teacher that spends its whole budget thinking before
#: it writes; with think *off* there is no chain of thought to run out of, and
#: a 16k ceiling only buys a teacher enough rope to ramble past its word band.
#: Think-on lanes keep real headroom, but 2048 rather than 16384: no row type
#: has yet demonstrated it needs more, and the 900s timeout that the old
#: budget forced (see ``dataset_build.sh``) hid every slow brief instead of
#: reporting it.
MAX_TOKENS_THINK_OFF = 800
MAX_TOKENS_THINK_ON = 2048

#: How far the budget may climb when a teacher is observed to truncate, as a
#: multiple of the lane's cap. The caps above are the *opening* offer, not a
#: ceiling: §B sized them for a teacher whose think flag decides whether a
#: chain of thought exists at all, and against one that reasons regardless they
#: are an order of magnitude short.
#:
#: Sized against a teacher that was thinking when it should not have been
#: (nine live rows wanting 4,084-13,167 tokens, a 3.2x spread) -- the headroom
#: a *genuinely* reasoning lane needs. With the think flag reaching the model
#: this ceiling is never approached: think-off rows land near 450 tokens.
#: It stays because a lane that truncates must be able to recover, and the
#: alternative is the run that spent forty-eight minutes producing nothing.
MAX_TOKENS_TRUNCATION_CEILING = 24
#: What a budget is multiplied by when a call truncates before writing a word.
#: Doubling, because the quantity being searched for varies by 3x between rows
#: and a linear probe would spend the run discovering it.
TRUNCATION_GROWTH = 2.0

#: Tokens this teacher spends *before* it writes a word, added to both caps.
#: Zero by default, because the amendment's two numbers are what an *answer*
#: costs and that is a property of the corpus.
#:
#: It exists for a stack whose reasoning genuinely cannot be switched off.
#:
#: It was *introduced* for the wrong reason, and the correction is worth
#: keeping: the first live bake-off returned `think_present: true` on all
#: twenty think-off calls, which was read as "this model reasons
#: unconditionally". It did not. `build_body` was sending only DeepSeek's
#: cloud dialect (`thinking: {"type": ...}`), which a vLLM/SparkInfer serve
#: ignores, and sending *nothing* for think=False -- so the server's own
#: `thinking: true` default stood and think-off was never off. With
#: `chat_template_kwargs` also sent, the same endpoint answers in 391-469
#: completion tokens and the 800-token cap fits with room to spare.
#:
#: So: before raising this, run `probe_teacher.py`. If it reports the flag
#: unhonoured, suspect the request dialect first and the model second.
#:
#: Raising this is a statement about the deployment, not about the corpus, and
#: it is deliberately separate from ``COSIMO_V3_MAX_TOKENS`` (which replaces a
#: budget outright and would flatten the two lanes into one number, losing the
#: distinction the ablation exists to measure). Set it to roughly what the
#: endpoint's reasoning costs; the answer budget on top stays the amendment's.
THINK_OVERHEAD_ENV = "COSIMO_V3_THINK_OVERHEAD"

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
#: Completion tokens per second the slowest acceptable teacher emits. The HTTP
#: deadline is derived from this and the call's budget, so a timeout means the
#: endpoint is slower than declared -- not that some brief is mysteriously long.
#:
#: Measured on the reference box: 5,080 completion tokens in 144s is ~35 tok/s,
#: and three of four probe calls timed out at the old deadline. 20 is that with
#: room for a queue.
#:
#: §B's "delete the 900s timeout, 120s is the ceiling, a live slice that
#: exceeds it is a brief bug" is right *for a lane that is not thinking* -- a
#: think-off row now finishes in 26-89s well inside the floor. The deadline
#: still follows the budget because a think-on lane legitimately wants minutes,
#: and a ceiling that cannot cover the budget it is waiting on turns a
#: recoverable truncation into a lost run.
TEACHER_TOKENS_PER_SECOND = 20.0
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

#: The same instrument turned *inward*: how close two rows of the corpus may
#: read to each other before the pair stops being two scenarios.
#:
#: The gold-bar fence above only ever looked outward -- train against the human
#: set -- so a corpus could repaint one scenario twenty times and the board
#: stayed green on every axis. That is v1's "71 stems x 1,000 paints" with a
#: smaller constant, and it is invisible to the numeric gates by construction:
#: every copy is *numerically* impeccable, because every copy came off the same
#: fact computer.
#:
#: Looser than the gold-bar threshold on purpose. Two rows of one family share
#: a question shape, a must_mention set and a register, so honest variants sit
#: higher against each other than a train row ever sits against a gold item.
#: 0.75 is the point past which two answers are the same telling with the
#: numbers swapped -- which is the thing worth failing, and the only thing this
#: can see.
CORPUS_NEAR_DUP_THRESHOLD = 0.75
#: Below this many rows in a cell a near-duplicate *rate* is a ratio of noise,
#: so the axis reports what it measured and certifies nothing -- the same
#: discipline the share axes keep.
NEAR_DUP_MIN_ROWS = 8
#: How many offending pairs the board names before it stops listing. A corpus
#: that repainted one family produces hundreds; the operator needs the first
#: few and the count, not the cross-product.
NEAR_DUP_MAX_REPORTED = 12
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


def family_max_share(n_train_families: int) -> float:
    """The per-family ceiling for a plan with *n_train_families* train families.

    An even split (``1 / n``) plus :data:`FAMILY_MAX_SHARE_HEADROOM`, which is
    what the cap was always *for*: not a promise that every family is under
    three per cent -- ten families cannot all be -- but a bound on how far the
    fattest family may run ahead of an even share before the expansion truncates
    it. Degenerate plans (no train families) get 1.0, i.e. no cap: there is no
    dominance to prevent when there is nothing to dominate.
    """
    if n_train_families <= 0:
        return 1.0
    return min(1.0, FAMILY_MAX_SHARE_HEADROOM / n_train_families)


def think_overhead() -> int:
    """The deployment's reasoning overhead in tokens; 0 unless declared."""
    raw = os.environ.get(THINK_OVERHEAD_ENV)
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def keep_teacher_messages() -> bool:
    """True when the operator asked for the debug transcript (amendment §A)."""
    return os.environ.get(KEEP_TEACHER_MESSAGES_ENV) == "1"


def memo_think() -> bool:
    """True when the operator turned the memo lane's chain of thought back on."""
    return os.environ.get(MEMO_THINK_ENV) == "1"


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
