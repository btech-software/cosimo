"""The publish gate, as a function (arch spec §6; analysis spec §6.3).

PR2 ships axes 1-5 -- everything that can be checked against a recomputed
fact pack and the prose contract, with no model in the room:

* **1 schema** -- the fields that make a prose row a prose row, the message
  roles in order, the self-consistency of ``answer`` with the assistant turn
  (verify one text, ship another is a corpus incident, not a mismatch), and
  no row holding both live and dead-lettered citizenship;
* **2 recompute** -- ``compute_pack(work_type, family, variant)`` must still
  produce the pack this row was written against, and the row's question,
  register, seed stamp and id must agree with it: the row carries its
  coordinates, the computers are pure, so *the pack is re-derived, never
  trusted from disk* -- which is the whole v3 bet, made into a check;
* **3 invented numbers** -- every number in the answer traces to the
  recomputed ``allowed_numbers`` plus the tiny structural whitelist;
* **4 must_mention / forbidden_claims** -- full coverage, zero forbidden
  hits, graded against the *recomputed* contract so a tampered stored field
  grades against truth;
* **5 ``FINAL ANSWER:`` is exam-only** -- its presence in a prose row is the
  v1 exam-skeleton leak, made unforgivable here.

PR3 adds the two axes an agentic row can fail with no model in the room:

* **9 tool schemas + roles** -- the spec's sentence verbatim ("tool schemas
  \u2286 ``cosimo.tools.registry`` and conversation roles valid"): every
  schema the row advertises resolves in the registry, every call validates
  against it, the transcript reads as a conversation a server could have
  served (the gate that judges it is :mod:`verification.agentic`, the one
  policy the render loop also faces -- one contract, three call sites);
* **10 tool-result replay** -- every stored tool block is re-executed,
  fault and all, against the *recomputed* pack: a trajectory is a claim
  about what the oracle said, and the claim is audited by saying it again
  and comparing bytes. A hand-edited result, a fabricated block, or a
  schedule field lying about which call ran dirty all fail here.

PR4 opens with the three corpus-measurement axes, the first that are
properties of the *slice* rather than of a row:

* **6 exam-liturgy share** -- ``ASSUMPTIONS:``/``Step N.`` counted from the
  shipped exam answers (marker detection on the text, not trust in the
  render field), must stay at or under ``config.LITURGY_CAP`` of the exam
  slice;
* **7 family share** -- no scenario family may dominate the field; the
  literal §5.1 cap (3% of the pool) is a *planning* contract the inventory
  enforces before generation, and at ten train families an emitted corpus
  cannot satisfy it literally -- ten stems must average a tenth each -- so
  the board measures the emitted, meaningful, satisfiable reading: every
  family carries the same shape within ``config.FAMILY_BALANCE_TOLERANCE``
  of the leanest;
* **8 exam share** -- the exam slice inside ``config.EXAM_SHARE_BAND``,
  v1's collapse-prevention band made enforceable.

Axes 6-8 *report* below ``config.SHARE_MIN_*`` support -- a share of a
twenty-row sample is noise, and a gate that shades red on noise gets
switched off -- and certify (can shade red) only on samples with support;
the certification site for the whole corpus is the publish gate, and the
tests certify the arithmetic on synthetic full-support corpora.

Axes 13-14 (gold-bar near-dup, teacher pinning) join with the publish
stage that produces their inputs, later in PR4. One numbering note for the
auditor reading against §6: the
spec's list jumps from 9 (tool schemas) to hidden tests without a number of
its own for the §5.8 trajectory audit, so the board ships 9 and 10 as the
two agentic axes above and the spec's 10-13 arrive renumbered 11-14 -- the
list's order preserved, its content untouched, the agentic audit given a
number of its own. ``--quick`` will skip the expensive dedup/gold-bar axes
then, exactly as spec'd; schema, replay and the share measurements are cheap
and always run, and the flag is accepted today so DAG command lines never
change shape.

The report is a plain dict, not a printed table: Airflow needs exit codes,
CI needs numbers, and the human-readable board is one ``print`` away in the
CLI -- formatting is the CLI's job, not this one's.
"""

from __future__ import annotations

import itertools
import os

from . import config, write
from . import row as rowlib
from .packs import PackError, compute_pack
from .render.prose import row_id
from .oracle.runtime import SCHEMA_NAMES as ADVERTISED_NAMES
from .seed import pack_seed
from .teacher.prompts import BRIEF_KINDS
from .verification.agentic import GROUNDING_TAG, REPLAY_TAG, trajectory_violations
from .verification.exam import (
    EXAM_KIND,
    LITURGY_MARKERS,
    TAG_FINAL,
    TAG_FORBIDDEN,
    TAG_INVENTED,
    TAG_TAMPER,
    exam_gate_violations,
)
from .verification.prose import (
    FINAL_ANSWER_TAG,
    canonical_numbers,
    forbidden_hits,
    integer_format_offenders,
    missing_mentions,
    rounding_drift,
    whitelist_for,
)
from .verification.register import (
    REGISTER_MIN_SEPARATION,
    profile_distance,
    register_profile,
    register_violations,
)
from .verification.invented_numbers import invented_numbers
from .verification.implementation import (
    IMPL_KIND,
    TAG_SANDBOX,
    TAG_FORBIDDEN as IMPL_TAG_FORBIDDEN,
    TAG_TAMPER as IMPL_TAG_TAMPER,
    impl_gate_violations,
    run_sandboxed,
)
from .verification.preference import (
    PREFER_KIND,
    TAG_COPY,
    TAG_PITFALL,
    TAG_SHAPE as PREF_TAG_SHAPE,
    TAG_TAMPER as PREF_TAG_TAMPER,
    pair_gate_violations,
    pair_id,
    shingle_overlap,
)

#: The axes ``--quick`` withholds: the sandboxed hidden suite and the two
#: near-duplicate sweeps (spec §6's "skips 10-12", renumbered for the board's
#: own 9/10). They are expensive and none is a *cheap* integrity gate -- schema,
#: replay and the share measurements always run -- so a CI smoke may stand on a
#: run that reports them skipped rather than having paid for them. The publish
#: gate, by contrast, refuses on a report where any of them was skipped.
EXPENSIVE_AXES = frozenset(
    {
        "hidden tests",
        "preference disjointness",
        "gold-bar near-dup",
        "corpus near-dup",
        "register separation",
    }
)

#: The axes implemented so far, in spec order, named as the board prints them.
AXES = (
    (1, "schema"),
    (2, "pack recompute"),
    (3, "invented numbers"),
    (4, "must_mention / forbidden_claims"),
    (5, "FINAL ANSWER is exam-only"),
    (6, "exam liturgy share"),
    (7, "family share"),
    (8, "exam share"),
    (9, "tool schemas + roles"),
    (10, "tool-result replay"),
    (11, "hidden tests"),
    (12, "preference disjointness"),
    (13, "gold-bar near-dup"),
    (14, "teacher pinned"),
    # The two axes the numeric gates are blind to by construction. Every row a
    # fact computer produces is impeccable *about its own pack*, so a corpus
    # can repaint one scenario twenty times, or write four registers in one
    # voice, and axes 1-14 stay green on every one of them. Both are properties
    # of the *slice*, so both are measured after the rows are read.
    (15, "corpus near-dup"),
    (16, "register separation"),
)

#: The agentic record type, named once: the board branches on it, the
#: ``kinds=`` default below carries it, and the axis-9/10 rows of the report
#: are populated only by rows that claim it.
AGENTIC_KIND = "agentic"

#: What every row must carry before a model's word is even read.
#: ``variant`` + ``scenario_id`` are not decoration: without them the row
#: cannot be re-derived from the computers, and axis 2 is the axis the whole
#: v3 design rests on.
#:
#: ``messages`` is gone from the shared list and that is the amendment's §A
#: change. A prose row's prompt *is* its question; the message list it used to
#: carry was the teacher's brief, and requiring it here is what made the brief
#: look like part of the contract. The three kinds whose prompt genuinely is a
#: transcript still declare it, in their own required lists below.
#:
#: ``family``, ``holdout``, ``fact_pack`` and ``verified`` are new and required:
#: they are what lets a reader downstream refuse a leak without re-deriving one
#: (``holdout``), score a row against its own contract without re-running the
#: generator (``fact_pack``), and read the corpus's claim about a row rather
#: than a proxy for it (``verified``).
ROW_REQUIRED = (
    "id",
    "record_type",
    "work_type",
    "scenario_id",
    "family",
    "holdout",
    "variant",
    "question",
    "answer",
    "register",
    "fact_pack",
    "verified",
    "verification",
)

#: What every agentic row must carry before any transcript is read: the
#: prose fields' coordinates plus the two the replay and the registry audit
#: cannot do without (the advertised schemas, and the calls the transcript
#: claims ran -- the second is redundant with the transcript and checked as
#: consistency, which is exactly why an ``answer`` can be tampered with here
#: and still be caught).
_AGENTIC_ROW_REQUIRED = ROW_REQUIRED + (
    "messages",
    "tool_names",
    "tool_schemas",
)


#: What every exam row must carry beyond the shared coordinates: the item's
#: own anatomy. ``options``/``answer_value`` are redundant with the answer by
#: construction -- which is the point: the gate re-derives the item from the
#: pack, and every stored field that disagrees with the recomposition is a
#: tamper finding, not a style choice.
_EXAM_ROW_REQUIRED = ROW_REQUIRED + (
    "messages",
    "question_text",
    "options",
    "answer_value",
    "answer_key",
    "unit",
)

#: What every implementation row must carry before its code is run at all:
#: the §5.9 record fields whole. Every one of them the board re-derives from
#: the pack -- their presence here is only so a half-pasted row is a *schema*
#: finding rather than a ``KeyError`` inside the sandbox call.
_IMPL_ROW_REQUIRED = ROW_REQUIRED + (
    "messages",
    "spec",
    "reference_code",
    "public_tests",
    "hidden_tests",
    "dirty_fixture",
    "limitations",
)


def _family_of(row: dict) -> str:
    return row["scenario_id"][len(row["work_type"]) + 1 :]


def _check_row(
    row: dict,
    dead_ids: frozenset[str],
    *,
    quick: bool = False,
    expect_holdout: bool = False,
) -> dict[str, list[str]]:
    """Per-axis failure messages for one row; empty lists mean clean."""
    failures: dict[str, list[str]] = {name: [] for _, name in AXES}
    rid = row.get("id", "?")
    kind = row.get("record_type")

    # -- axis 1: the shape that lets the rest of the board read the row ---
    missing = [field for field in ROW_REQUIRED if field not in row]
    if missing:
        failures["schema"].append(f"missing fields: {', '.join(missing)}")
    roles = [m.get("role") for m in row.get("messages") or [] if isinstance(m, dict)]
    answer = row.get("answer")
    if kind == AGENTIC_KIND:
        missing = [field for field in _AGENTIC_ROW_REQUIRED if field not in row]
        if missing:
            failures["schema"].append(f"missing fields: {', '.join(missing)}")
        if roles[:1] != ["user"] or (roles and roles[-1] != "assistant"):
            failures["schema"].append(
                f"an agentic transcript must open on the user's goal and close "
                f"on the desk's answer; roles read {roles!r}"
            )
        if "system" in roles:
            failures["schema"].append(
                "an agentic transcript carries a system turn -- that is the "
                "factory's brief, not the student's prompt (amendment §A)"
            )
        if isinstance(answer, str) and answer.strip() and roles[-1:] == ["assistant"]:
            if (row["messages"][-1] or {}).get("content") != answer:
                failures["schema"].append(
                    "answer and the final assistant turn have drifted apart "
                    "(verify one text, ship another)"
                )
        advertised = row.get("tool_schemas") or []
        unknown = sorted(
            {
                (schema.get("function") or {}).get("name")
                for schema in advertised
                if isinstance(schema, dict)
            }
            - set(ADVERTISED_NAMES)
        )
        if unknown:
            failures["schema"].append(
                "advertised schemas name unregistered tools: "
                + ", ".join(repr(n) for n in unknown)
            )
    elif kind == EXAM_KIND:
        missing = [field for field in _EXAM_ROW_REQUIRED if field not in row]
        if missing:
            failures["schema"].append(f"missing fields: {', '.join(missing)}")
        if roles != ["user", "assistant"]:
            failures["schema"].append(
                f"an exam row must read user/assistant -- the answer protocol is "
                f"the harness's to bind, not the corpus's; roles read {roles!r}"
            )
        # Guarded on `roles`, not on the key: `_EXAM_ROW_REQUIRED` catches a
        # *missing* `messages`, and an empty list would sail past it straight
        # into an IndexError. The board reports findings; it does not crash on
        # the malformed rows it exists to find.
        if roles and row.get("question_text") != (row["messages"][0] or {}).get(
            "content"
        ):
            failures["schema"].append(
                "question_text and the user turn have drifted apart; the item's "
                "options must be one string, not two"
            )
        if isinstance(answer, str) and roles[-1:] == ["assistant"]:
            if (row["messages"][-1] or {}).get("content") != answer:
                failures["schema"].append(
                    "answer and the assistant turn have drifted apart (verify one "
                    "text, ship another)"
                )
        if (row.get("verification") or {}).get("teacher") is not None:
            failures["schema"].append(
                "an exam row claims a teacher -- the exam slice is composed, not "
                "dictated; a row that was dictated is not an exam row"
            )
    elif kind == IMPL_KIND:
        missing = [field for field in _IMPL_ROW_REQUIRED if field not in row]
        if missing:
            failures["schema"].append(f"missing fields: {', '.join(missing)}")
        if roles != ["user", "assistant"]:
            failures["schema"].append(
                f"an implementation row must read user/assistant -- the spec is "
                f"the prompt, the factory's brief around it is not; roles read "
                f"{roles!r}"
            )
        if isinstance(answer, str) and roles[-1:] == ["assistant"]:
            if (row["messages"][-1] or {}).get("content") != answer:
                failures["schema"].append(
                    "answer and the assistant turn have drifted apart (verify one "
                    "text, ship another)"
                )
    else:
        # A prose row carries no transcript at all. Its prompt is its question
        # and its target is its answer; the brief that produced it -- system
        # turn, JSON contract, word budget -- is the teacher log's, and a prose
        # row that still has `messages` came out of a renderer that predates
        # the two-surface split (amendment §A).
        if "messages" in row:
            failures["schema"].append(
                "a prose row carries `messages` -- that is the teacher's brief, "
                "and the only trainable surface is question + answer"
            )
        if not isinstance(answer, str) or not answer.strip():
            failures["schema"].append("answer is empty")
        if kind not in BRIEF_KINDS:
            failures["schema"].append(f"record_type {kind!r} is not a prose kind")
    if not str(rid).startswith(f"{config.SUPERVISED_ID_PREFIX}_"):
        failures["schema"].append(
            f"id {rid!r} is outside the supervised namespace "
            f"{config.SUPERVISED_ID_PREFIX!r}"
        )
    if str(rid) in dead_ids:
        failures["schema"].append(
            "row is simultaneously live in sft/ and dead-lettered -- one of the "
            "two files is lying; delete the entry that is wrong and replay"
        )
    # The §A leak fingerprint, read at the board as well as at prepare. Two
    # gates on one rule is deliberate: the corpus must refuse to *ship* a row
    # that carries the factory's protocol, and the harness must refuse to
    # *train* on one, because the two subsystems join on the Hub and either can
    # be handed bytes the other never saw.
    if rowlib.carries_teacher_brief(row):
        failures["schema"].append(
            f"row carries the teacher fingerprint {config.TEACHER_FINGERPRINT!r} "
            "-- the factory's own brief reached a trainable surface (§A)"
        )
    if row.get("verified") is not True:
        failures["schema"].append(
            "row does not claim `verified: true`; a shard row is the corpus's "
            "assertion that its gates passed, not a place to leave the claim open"
        )
    if bool(row.get("holdout")) is not bool(expect_holdout):
        where = "eval/" if expect_holdout else "sft/"
        failures["schema"].append(
            f"row says holdout={row.get('holdout')!r} but was read out of "
            f"{where} -- a holdout family in a training shard is the one leak "
            "no downstream number can un-certify"
        )
    if failures["schema"]:
        # Axes 2-5 and 10 grade content, and content cannot be located
        # without a shape. Report the break once; do not stack derivatives.
        return failures

    # -- axis 2: the pack, re-derived, is the only authority ---------------
    try:
        pack = compute_pack(row["work_type"], _family_of(row), row["variant"])
    except PackError as exc:
        failures["pack recompute"].append(f"pack no longer computable: {exc}")
        return failures
    stamp = (row.get("verification") or {}).get("pack_seed")
    expected_stamp = (
        f"{pack_seed(row['work_type'], _family_of(row), row['variant']):016x}"
    )
    if stamp != expected_stamp:
        failures["pack recompute"].append(
            f"seed stamp {stamp!r} does not match the seed its coordinates "
            f"hash to ({expected_stamp!r})"
        )
    if row["question"] != pack.question:
        failures["pack recompute"].append("question drifted from the recomputed pack")
    if row["register"] != pack.register:
        failures["pack recompute"].append("register drifted from the recomputed pack")
    if str(rid) != row_id(kind, pack.to_dict()):
        failures["pack recompute"].append(
            "id does not hash from the coordinates the row carries -- a pasted "
            "or hand-written id, not a derived one"
        )

    if kind == AGENTIC_KIND:
        # Axes 3-5 are the prose contract; their agentic counterparts are
        # §5.8's grounding/acknowledgement disciplines, which the trajectory
        # gate below enforces whole -- graded against the *recomputed* pack,
        # never the stored one, exactly as axes 3-4 grade prose.
        render_field = (row.get("verification") or {}).get("render") or {}
        if render_field.get("kind") != AGENTIC_KIND:
            failures["tool schemas + roles"].append(
                "verification.render.kind is "
                f"{render_field.get('kind')!r}, not {AGENTIC_KIND!r} -- the row "
                "does not say who rendered it"
            )
            return failures
        violations = trajectory_violations(
            pack_dict := pack.to_dict(),
            [m for m in row["messages"] if isinstance(m, dict)],
            mode=render_field.get("mode"),
            fault=render_field.get("fault"),
        )
        # File each violation under the axis that owns it: the replay under
        # 10, the §5.8 subset check under 3 (the spec's own invented-number
        # axis, read with the pack ∪ results authority), the shared clauses
        # under 4 and 5 exactly where prose files them, and the rest -- the
        # shape, budget, mode and acknowledgement disciplines -- under 9,
        # which is the spec's sentence "conversation roles valid" widened to
        # the whole transcript. One gate, the same board an audit reads.
        for violation in violations:
            if violation.startswith(REPLAY_TAG):
                failures["tool-result replay"].append(violation)
            elif violation.startswith(GROUNDING_TAG):
                failures["invented numbers"].append(violation)
            elif violation.startswith("must_mention not covered:"):
                failures["must_mention / forbidden_claims"].append(violation)
            elif violation.startswith("forbidden claim asserted:"):
                failures["must_mention / forbidden_claims"].append(violation)
            elif "is an exam contract" in violation:
                failures["FINAL ANSWER is exam-only"].append(violation)
            else:
                failures["tool schemas + roles"].append(violation)
        return failures

    if kind == EXAM_KIND:
        # The exam row's content contract is one recomposition: the gate
        # rebuilds the item from the recomputed pack and every stored byte
        # that disagrees is filed where its kind is judged -- tampering and
        # shape under 1, the §5.6 number-subset reading under 3 (the student
        # may print only the pack's own figures; the examiner's distractors
        # live in the question), the forbidden wall under 4, the exam-only
        # closing tag under 5. ``must_mention`` is deliberately absent: it is
        # the *prose* coverage contract, and an exam answer is a worked
        # computation, not a memo with points to hit.
        render_field = (row.get("verification") or {}).get("render") or {}
        if render_field.get("kind") != EXAM_KIND:
            failures["schema"].append(
                "verification.render.kind is "
                f"{render_field.get('kind')!r}, not {EXAM_KIND!r} -- the row "
                "does not say who composed it"
            )
            return failures
        for violation in exam_gate_violations(pack.to_dict(), row):
            if violation.startswith(TAG_TAMPER):
                failures["schema"].append(violation)
            elif violation.startswith(TAG_INVENTED):
                failures["invented numbers"].append(violation)
            elif violation.startswith(TAG_FORBIDDEN):
                failures["must_mention / forbidden_claims"].append(violation)
            elif violation.startswith(TAG_FINAL):
                failures["FINAL ANSWER is exam-only"].append(violation)
            else:  # TAG_SHAPE, and anything a later gate adds: shape of an item
                failures["schema"].append(violation)
        return failures

    if kind == IMPL_KIND:
        # The implementation row's content contract, in the doctrine's order:
        # first the recomposition (every stored byte the pack entails, checked
        # by the gate), and only then the suite -- executed over bytes that
        # agreed, because the board runs no code the corpus did not author.
        # The suite's verdict is this axis's; a tamper is integrity (axis 1,
        # where the exam files its own) *and* a refusal to execute, filed
        # here so a red board cannot be read as "the tests passed silently";
        # the limitations' fact-lock reads with 3 and 4, where every other
        # lane's words are read.
        render_field = (row.get("verification") or {}).get("render") or {}
        if render_field.get("kind") != IMPL_KIND:
            failures["schema"].append(
                "verification.render.kind is "
                f"{render_field.get('kind')!r}, not {IMPL_KIND!r} -- the row "
                "does not say who composed it"
            )
            return failures
        violations = impl_gate_violations(pack.to_dict(), row)
        for violation in violations:
            if violation.startswith(IMPL_TAG_FORBIDDEN):
                failures["must_mention / forbidden_claims"].append(violation)
            else:  # shape, recomposition tamper, a gone-pack: integrity findings
                failures["schema"].append(violation)
        if not quick:
            if any(v.startswith(IMPL_TAG_TAMPER) for v in violations):
                failures["hidden tests"].append(
                    f"{TAG_SANDBOX}not executed: the stored bytes are not the "
                    "recomposition, and the board runs only what it authored"
                )
            else:
                passed, log = run_sandboxed(
                    row["reference_code"],
                    [*row["public_tests"], *row["hidden_tests"]],
                    pack.to_dict().get("inputs") or {},
                )
                if not passed:
                    failures["hidden tests"].append(f"{TAG_SANDBOX}{log}")
        return failures

    # -- axes 3-5: the words, against the re-derived contract --------------
    pack_dict = pack.to_dict()
    # Against ``canonical``, not the union, exactly as the render gate now
    # reads it: one policy, three call sites is the rule this whole module
    # exists to keep, and an auditor that graded against a looser authority
    # than the generator did would certify rows the generator would have
    # repaired (amendment §D).
    for token in invented_numbers(
        answer, canonical_numbers(pack_dict), whitelist_for(pack_dict)
    ):
        failures["invented numbers"].append(f"{token!r} is not in the fact pack")
    for drift in rounding_drift(pack_dict, answer):
        failures["invented numbers"].append(drift)
    for spelled in integer_format_offenders(pack_dict, answer):
        failures["invented numbers"].append(
            f"whole number written with a decimal tail: {spelled}"
        )
    for violation in register_violations(pack.register, answer, kind=kind):
        failures["schema"].append(violation)
    for point in missing_mentions(pack_dict, answer):
        failures["must_mention / forbidden_claims"].append(
            f"must_mention not covered: {point!r}"
        )
    for claim in forbidden_hits(pack_dict, answer):
        failures["must_mention / forbidden_claims"].append(
            f"forbidden claim asserted: {claim!r}"
        )
    if FINAL_ANSWER_TAG.casefold() in answer.casefold():
        failures["FINAL ANSWER is exam-only"].append(
            f"{FINAL_ANSWER_TAG!r} in a {kind} row"
        )
    return failures


def _measure_shares(axes_report: dict, rows: list[dict]) -> None:
    """Axes 6-8: properties of the slice, filled after the rows are read.

    Measurement, not trust: the liturgy count scans the shipped answer text
    for the compositor's markers instead of reading the ``render.liturgy``
    field the same file could lie about, and the family tallies count what
    ``sft/`` holds, not what the plan promised (the plan-time cap is the
    inventory's; this is the emitted corpus's). Below the configured support
    floors an axis *reports*: it counts what it saw, attaches the note that
    says it cannot certify, and stays silent -- a 3% cap measured on a
    twenty-row sample would shade red on noise, and a gate that shades red
    on noise is a gate somebody switches off. Certification at full support
    happens over the whole corpus, which is the publish gate's material.
    """
    total = len(rows)
    exam_rows = [r for r in rows if r.get("record_type") == EXAM_KIND]
    families: dict[tuple[str, str], int] = {}
    for row in rows:
        try:
            key = (row["work_type"], _family_of(row))
        except (KeyError, TypeError):
            continue  # a shapeless row is axis 1's finding, already filed
        families[key] = families.get(key, 0) + 1

    liturgy_share = exam_share = family_ratio = None
    if exam_rows:
        liturgy_share = sum(
            1
            for row in exam_rows
            if isinstance(row.get("answer"), str)
            and any(marker in row["answer"] for marker in LITURGY_MARKERS)
        ) / len(exam_rows)
    if total:
        exam_share = len(exam_rows) / total
    if families:
        leanest = min(families.values())
        family_ratio = max(families.values()) / leanest if leanest else float("inf")

    certified = total >= config.SHARE_MIN_ROWS
    axes_report["exam liturgy share"]["checked"] = len(exam_rows)
    if not exam_rows:
        axes_report["exam liturgy share"]["note"] = "no exam rows: nothing to weigh"
    elif not certified or len(exam_rows) < config.SHARE_MIN_EXAM_ROWS:
        axes_report["exam liturgy share"]["note"] = (
            f"{len(exam_rows)} exam rows of {total}: below the support "
            f"({config.SHARE_MIN_EXAM_ROWS}) the cap cannot be certified"
        )
    elif liturgy_share > config.LITURGY_CAP + 1e-9:
        axes_report["exam liturgy share"]["failures"].append(
            {
                "id": "<slice>",
                "problem": (
                    f"liturgy share {liturgy_share:.3f} exceeds the cap "
                    f"{config.LITURGY_CAP} -- the slice thinned unevenly or the "
                    "sampler drifted; count the Step N. markers"
                ),
            }
        )

    axes_report["family share"]["checked"] = total
    if not families:
        axes_report["family share"]["note"] = "no families measured"
    elif not certified or min(families.values()) < config.SHARE_MIN_PER_FAMILY:
        axes_report["family share"]["note"] = (
            f"{len(families)} families, leanest "
            f"{min(families.values()) if families else 0} rows: below the "
            f"support ({config.SHARE_MIN_PER_FAMILY}) dominance cannot be certified"
        )
    elif family_ratio > config.FAMILY_BALANCE_TOLERANCE + 1e-9:
        worst = max(families, key=families.get)
        axes_report["family share"]["failures"].append(
            {
                "id": f"{worst[0]}/{worst[1]}",
                "problem": (
                    f"family share ratio {family_ratio:.2f} exceeds the balance "
                    f"tolerance {config.FAMILY_BALANCE_TOLERANCE} -- one stem is "
                    "dominating the field (the v2 1,000x71 pathology, axis 7)"
                ),
            }
        )

    axes_report["exam share"]["checked"] = total
    if exam_share is None:
        axes_report["exam share"]["note"] = "empty corpus"
    elif not certified or len(exam_rows) < config.SHARE_MIN_EXAM_ROWS:
        axes_report["exam share"]["note"] = (
            f"{len(exam_rows)} exam rows of {total}: below the support "
            f"({config.SHARE_MIN_EXAM_ROWS}) the band cannot be certified"
        )
    elif (
        not config.EXAM_SHARE_BAND[0] <= exam_share <= config.EXAM_SHARE_BAND[1] + 1e-9
    ):
        axes_report["exam share"]["failures"].append(
            {
                "id": "<slice>",
                "problem": (
                    f"exam share {exam_share:.3f} outside the band "
                    f"{list(config.EXAM_SHARE_BAND)} -- the mix the plan did not "
                    "sign (v1's 12% collapse guard, axis 8)"
                ),
            }
        )


def _measure_teacher_pinning(axes_report: dict, rows: list[dict]) -> None:
    """Axis 14: every row pins the teacher that wrote it, and one style rules.

    Provenance is a fact the row carries, not a hope the operator keeps:
    ``verification.teacher.model`` names the model that answered, and a corpus
    that quietly carries two is the v2 "diversity" that was really a mid-run
    swap -- two house styles passed off as range. So a single pinned model is
    always intentional; more than one is red until the operator declares that
    exact set intentional through ``COSIMO_V3_TEACHER_ALLOWLIST``. A row with no
    The audited population is the rows that *claim* teacher authorship -- a
    ``verification.teacher`` dict. An exam row sets that field to ``None`` on
    purpose (its item is composed by the computer, not dictated by a model),
    and is therefore not this axis's to hold to a model pin; a prose, agentic
    or implementation row carries the dict and must name who wrote it. A row
    with no ``verification`` envelope at all is axis 1's finding: the board
    files a break once and does not stack a provenance finding on a schema one.
    """
    axis = axes_report["teacher pinned"]
    models: dict[str, list[str]] = {}
    unpinned: list[str] = []
    audited = 0
    for row in rows:
        verification = row.get("verification")
        if not isinstance(verification, dict):
            continue
        teacher = verification.get("teacher")
        if teacher is None:
            continue  # composed, not dictated: the row says it has no teacher
        if not isinstance(teacher, dict):
            continue
        audited += 1
        model = teacher.get("model")
        rid = str(row.get("id", "?"))
        if not (isinstance(model, str) and model.strip()):
            unpinned.append(rid)
        else:
            models.setdefault(model, []).append(rid)
    axis["checked"] = audited
    for rid in unpinned:
        axis["failures"].append(
            {
                "id": rid,
                "problem": (
                    "the row pins no teacher model in verification.teacher.model "
                    "-- provenance is unpinned and a silent swap could not be "
                    "seen (axis 14)"
                ),
            }
        )
    if len(models) > 1:
        observed = frozenset(models)
        if not observed <= config.teacher_allowlist():
            listed = ", ".join(
                f"{model} ({len(ids)} rows)" for model, ids in sorted(models.items())
            )
            axis["failures"].append(
                {
                    "id": "<slice>",
                    "problem": (
                        f"the corpus mixes teachers -- {listed}; a mid-run swap is "
                        "how you get two house styles and call it diversity. Run "
                        "one teacher, or declare the mix intentional via "
                        f"{config.TEACHER_ALLOWLIST_ENV}="
                        f"{','.join(sorted(observed))!r} (axis 14)"
                    ),
                }
            )


#: Record types whose answers are *meant* to repeat, so the repaint axis must
#: not read them. ``implementation`` ships the reviewed reference instrument
#: from ``verification.impl_references`` -- one function per work type, byte
#: identical across every variant, and that identity is the contract: the board
#: re-executes those exact bytes, and a lane that wrote a different instrument
#: per variant would be a lane nobody had reviewed. Their *diversity* lives in
#: the suites and the dirty fixture, which the hidden-test axis already proves.
_NEAR_DUP_EXEMPT = frozenset({IMPL_KIND})


def _measure_corpus_near_dup(axes_report: dict, rows: list[dict]) -> None:
    """Axis 15: no two rows of a cell may be the same telling, numbers aside.

    The gold-bar fence looks outward and this looks in. Nothing else on the
    board can: a repainted scenario passes the recompute axis (its pack is
    real), the invented-number axis (its figures are the pack's) and the share
    axes (it is one row like any other). The only thing wrong with it is that
    the corpus already contains it, and that is a statement about a *pair*.

    Compared within ``(work_type, record_type)`` rather than across the corpus.
    Two work types answering different questions should differ, and saying so
    costs a quadratic sweep over thousands of rows to learn nothing; the repaint
    this exists to catch lives between variants of one family and between
    families of one computer, and both sit inside the cell.
    """
    axis = axes_report["corpus near-dup"]
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            continue  # a shapeless row is axis 1's finding, already filed
        if row.get("record_type") in _NEAR_DUP_EXEMPT:
            continue
        cells.setdefault(
            (str(row.get("work_type")), str(row.get("record_type"))), []
        ).append(row)

    checked = pairs = flagged = 0
    thin: list[str] = []
    for (work_type, kind), cell in sorted(cells.items()):
        checked += len(cell)
        if len(cell) < config.NEAR_DUP_MIN_ROWS:
            thin.append(f"{work_type}/{kind} ({len(cell)})")
            continue
        for left, right in itertools.combinations(cell, 2):
            pairs += 1
            overlap = shingle_overlap(left["answer"], right["answer"])
            if overlap < config.CORPUS_NEAR_DUP_THRESHOLD:
                continue
            flagged += 1
            if len(axis["failures"]) < config.NEAR_DUP_MAX_REPORTED:
                axis["failures"].append(
                    {
                        "id": left.get("id", "?"),
                        "problem": (
                            f"reads {overlap:.2f} against {right.get('id', '?')} "
                            f"(threshold {config.CORPUS_NEAR_DUP_THRESHOLD}) -- two "
                            "rows of one cell telling the same thing is repaint, "
                            "not coverage"
                        ),
                    }
                )
    axis["checked"] = checked
    notes = []
    if pairs:
        notes.append(f"{flagged} of {pairs} in-cell pairs at or over threshold")
    if flagged > len(axis["failures"]):
        notes.append(f"only the first {len(axis['failures'])} named")
    if thin:
        notes.append(
            f"below the {config.NEAR_DUP_MIN_ROWS}-row support, not swept: "
            + ", ".join(thin[:4])
        )
    if notes and not axis["failures"]:
        axis["note"] = "; ".join(notes)


def _measure_register_separation(axes_report: dict, rows: list[dict]) -> None:
    """Axis 16: the registers must be *distinguishable*, not merely labelled.

    ``verification.register`` is a per-row gate and it can only ask whether a
    row breaks the shape its own register forbids. That catches a desk note
    wearing memo headings; it cannot catch every register writing one voice,
    because no single row is wrong -- the corpus is. So the distinctiveness
    question is asked here, where the slice is, and asked as a measurement
    rather than a rule: a row is never failed for it.

    The profile is deliberately crude -- how long the sentences run, how often
    headings appear, how often a call is made. Three numbers a reader can
    argue with beat a similarity score nobody can act on, and the failure this
    is watching for is gross: ``register_match`` collapsing while every other
    axis reads clean.
    """
    axis = axes_report["register separation"]
    by_register: dict[str, list[dict]] = {}
    for row in rows:
        answer = row.get("answer")
        register = str(row.get("register") or "")
        if register and isinstance(answer, str) and answer.strip():
            by_register.setdefault(register, []).append(row)
    axis["checked"] = sum(len(v) for v in by_register.values())

    profiles = {
        register: register_profile([r["answer"] for r in cell])
        for register, cell in by_register.items()
        if len(cell) >= config.NEAR_DUP_MIN_ROWS
    }
    if len(profiles) < 2:
        axis["note"] = (
            "fewer than two registers have the "
            f"{config.NEAR_DUP_MIN_ROWS}-row support a comparison needs; "
            "separation not measured"
        )
        return
    collapsed = []
    for left, right in itertools.combinations(sorted(profiles), 2):
        distance = profile_distance(profiles[left], profiles[right])
        if distance < REGISTER_MIN_SEPARATION:
            collapsed.append(f"{left} vs {right} ({distance:.2f})")
    summary = ", ".join(
        f"{name}={profiles[name]['sentence_len']:.0f}w/sentence "
        f"heads={profiles[name]['heading_rate']:.2f} "
        f"calls={profiles[name]['call_rate']:.2f}"
        for name in sorted(profiles)
    )
    if collapsed:
        # Reported, never red. A collapsed register is a finding about the
        # *briefs* -- the teacher was asked for four voices and gave one -- and
        # failing the board would block a publish on prose nobody has read yet.
        # The number belongs where a human decides, which is this note and the
        # `register_match` column of `09_assistant_eval`.
        axis["note"] = (
            "registers read alike, separation below "
            f"{REGISTER_MIN_SEPARATION}: " + "; ".join(collapsed) + f" | {summary}"
        )
    else:
        axis["note"] = f"separated | {summary}"


def _measure_gold_bar(axes_report: dict, rows: list[dict], gold_bar_path: str) -> None:
    """Axis 13: no training row may read like a gold-bar item (spec §5.11).

    The gold bar is the held-out human set -- eval material -- and a train row
    that near-duplicates one is a leak, not coverage. The fence is the same
    Jaccard shingle overlap the preference lane trusts, at the shared control
    threshold. It is a measurement needing both sides: with no bar at the
    configured path the axis reports what it could not do rather than shade a
    corpus red for the absence of a human artefact a CI box has never held --
    but ``publish`` treats that same absence as red, for a real publish has no
    excuse to ship without the fence standing.
    """
    axis = axes_report["gold-bar near-dup"]
    if not os.path.isfile(gold_bar_path):
        axis["checked"] = 0
        axis["note"] = (
            f"no gold bar at {gold_bar_path!r}: the near-duplicate fence could "
            "not run (publish refuses on this)"
        )
        return
    try:
        gold = write.read_jsonl(gold_bar_path)
    except ValueError as exc:  # a corrupt bar is a finding on the fence itself
        axis["checked"] = 0
        axis["failures"].append({"id": gold_bar_path, "problem": str(exc)})
        return
    gold_texts = [
        (
            str(item.get("id", "?")),
            str(item.get("answer") or item.get("content") or ""),
        )
        for item in gold
    ]
    gold_texts = [(gid, text) for gid, text in gold_texts if text.strip()]
    axis["checked"] = len(rows)
    for row in rows:
        text = str(row.get("answer") or "")
        if not text.strip():
            continue
        for gid, gold_text in gold_texts:
            overlap = shingle_overlap(text, gold_text)
            if overlap >= config.GOLDBAR_NEAR_DUP_THRESHOLD:
                axis["failures"].append(
                    {
                        "id": str(row.get("id", "?")),
                        "problem": (
                            f"train row reads {overlap:.2f} against gold-bar item "
                            f"{gid!r}, at or over the "
                            f"{config.GOLDBAR_NEAR_DUP_THRESHOLD} near-duplicate "
                            "fence -- a leak of the held-out set into training "
                            "(axis 13)"
                        ),
                    }
                )
                break


def _check_pair(
    pair: dict, dead_ids: frozenset[str], parent_loader
) -> dict[str, list[str]]:
    """Per-axis failure messages for one preference pair; empty lists mean clean.

    The pack is recomputed and the parent row fetched from the shard its
    kind names; the pair's stored bytes are then graded against that
    authority. Axes 3, 4 and 5 take the *chosen side's* clauses -- a chosen
    that claims to tell the pack is graded as any prose lane is -- and
    axis 12 owns the pair's own physics: id disjointness by prefix, the
    no-copy thresholds, and whether the rejected side actually committed
    the named crime. A pair that cannot locate its parent is an integrity
    finding on 12, not an orphan to be forgiven.
    """
    failures: dict[str, list[str]] = {name: [] for _, name in AXES}
    try:
        work_type = pair["work_type"]
        variant = pair["variant"]
        kind = pair["parent_kind"]
        family = pair["scenario_id"][len(work_type) + 1 :]
    except (KeyError, TypeError):
        failures["schema"].append(
            "the pair does not carry the coordinates the board reads it by"
        )
        return failures
    rid = pair.get("id", "?")
    if str(rid) in dead_ids:
        failures["schema"].append(
            "pair is simultaneously live in preference/ and dead-lettered -- one "
            "of the two files is lying; delete the entry that is wrong and replay"
        )
    try:
        pack = compute_pack(work_type, family, variant)
    except PackError as exc:
        failures["pack recompute"].append(f"pack no longer computable: {exc}")
        return failures
    stamp = (pair.get("verification") or {}).get("pack_seed")
    expected_stamp = f"{pack_seed(work_type, family, variant):016x}"
    if stamp != expected_stamp:
        failures["pack recompute"].append(
            f"seed stamp {stamp!r} does not match the seed its coordinates hash "
            f"to ({expected_stamp!r})"
        )
    if str(rid) != pair_id(work_type, family, kind, variant):
        failures["pack recompute"].append(
            "id does not hash from the coordinates the pair carries -- a pasted "
            "or hand-written id, not a derived one"
        )
    parent = parent_loader(kind, pair.get("source_sft_id"))
    for violation in pair_gate_violations(pack.to_dict(), parent, pair):
        if violation.startswith(PREF_TAG_SHAPE):
            failures["schema"].append(violation)
        elif violation.startswith((PREF_TAG_TAMPER, TAG_COPY, TAG_PITFALL)):
            failures["preference disjointness"].append(violation)
        elif "chosen side: invented numbers" in violation:
            failures["invented numbers"].append(violation)
        elif "chosen side:" in violation and (
            "must_mention" in violation or "forbidden" in violation
        ):
            failures["must_mention / forbidden_claims"].append(violation)
        elif "chosen side:" in violation and "exam contract" in violation:
            failures["FINAL ANSWER is exam-only"].append(violation)
        else:  # untagged residue is structure of a side; the shape axis reads it
            failures["schema"].append(violation)
    return failures


def verify_dir(
    out_dir: str,
    *,
    kinds: tuple[str, ...] = (
        *BRIEF_KINDS,
        AGENTIC_KIND,
        EXAM_KIND,
        IMPL_KIND,
        PREFER_KIND,
    ),
    quick: bool = False,
    gold_bar_path: str | None = None,
) -> dict:
    """Run every implemented axis over ``<out>/sft/`` and report.

    Exit-0 discipline lives in the CLI; this function only states facts:
    ``{"ok", "rows", "axes": {name: {"checked", "failures": [{id, problem}],
    "note"?}}}`` -- the optional ``note`` is a measurement that could not be
    certified for want of support, printed by the CLI and never red.
    """
    skipped = EXPENSIVE_AXES if quick else frozenset()
    bar_path = config.gold_bar_v3_path() if gold_bar_path is None else gold_bar_path
    axes_report: dict[str, dict] = {}
    for _, name in AXES:
        axes_report[name] = (
            {
                "checked": 0,
                "failures": [],
                "skipped": True,
                "note": "--quick: expensive axis not run",
            }
            if name in skipped
            else {"checked": 0, "failures": []}
        )
    ok = True
    rows_seen = 0
    all_rows: list[dict] = []
    for kind in kinds:
        if kind == PREFER_KIND:
            continue  # pairs live in their own shard and their own config
        dead_ids = write.existing_ids(write.path_for("dead_letter", kind, out_dir))
        # Both cohorts, each audited against what its own bucket claims. The
        # eval tree is held to exactly the same axes -- a gold-bar row nobody
        # verified is worth less than no eval slice at all -- but its rows are
        # kept out of ``all_rows``, because the share axes measure the *training*
        # distribution and an eval family is not competing for training rows.
        for bucket, expect_holdout in (("sft", False), ("eval", True)):
            path = write.path_for(bucket, kind, out_dir)
            try:
                rows = write.read_jsonl(path)
            except ValueError as exc:  # a corrupt shard is itself a finding
                axes_report["schema"]["failures"].append(
                    {"id": path, "problem": str(exc)}
                )
                ok = False
                continue
            for row in rows:
                rows_seen += 1
                if not expect_holdout:
                    all_rows.append(row)
                # One audit per row, all axes read from it: ``_check_row``
                # re-derives the pack, and doing that once per axis multiplied
                # the bill by the number of axes for nothing.
                result = _check_row(
                    row, dead_ids, quick=quick, expect_holdout=expect_holdout
                )
                for _, name in AXES:
                    if name in skipped:
                        continue
                    axes_report[name]["checked"] += 1
                    for problem in result[name]:
                        axes_report[name]["failures"].append(
                            {"id": row.get("id", "?"), "problem": problem}
                        )
    if PREFER_KIND in kinds and not quick:
        parent_cache: dict[str, dict[str, dict]] = {}

        def parent_loader(kind, source_id):
            """The parent SFT row by id, shard cached.

            A corrupt parent shard was already recorded (and ``ok`` already
            falsified) by the supervised pass over the very same file; here
            it reads empty, which the pair gate files as an orphan finding
            on axis 12 rather than losing twice.
            """
            if kind not in parent_cache:
                try:
                    rows = write.read_jsonl(write.path_for("sft", kind, out_dir))
                except ValueError:
                    parent_cache[kind] = {}
                else:
                    parent_cache[kind] = {r.get("id"): r for r in rows}
            return parent_cache[kind].get(source_id)

        dead_pref = write.existing_ids(
            write.path_for("dead_letter", PREFER_KIND, out_dir)
        )
        for pair in write.read_jsonl(write.path_for("preference", "pairs", out_dir)):
            rows_seen += 1
            # Deliberately not into ``all_rows``: the share axes weigh the
            # supervised corpus (spec §5.2 keeps the preference config
            # separate); pairing a tenth of the analyses must not move the
            # exam-share band it never touched.
            result = _check_pair(pair, dead_pref, parent_loader)
            for _, name in AXES:
                if name in skipped:
                    continue
                axes_report[name]["checked"] += 1
                for problem in result[name]:
                    axes_report[name]["failures"].append(
                        {"id": pair.get("id", "?"), "problem": problem}
                    )
    _measure_shares(axes_report, all_rows)
    _measure_teacher_pinning(axes_report, all_rows)
    if not quick:
        _measure_gold_bar(axes_report, all_rows, bar_path)
        _measure_corpus_near_dup(axes_report, all_rows)
        _measure_register_separation(axes_report, all_rows)
    ok = ok and not any(a["failures"] for a in axes_report.values())
    return {"ok": ok, "rows": rows_seen, "axes": axes_report}
