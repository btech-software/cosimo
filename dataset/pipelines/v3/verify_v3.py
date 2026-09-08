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

Axes 11-14 (hidden tests, preference disjointness, gold-bar near-dup,
teacher pinning) join with the stages that produce their inputs, later in
PR4. One numbering note for the auditor reading against §6: the
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

from . import config, write
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
    forbidden_hits,
    missing_mentions,
    whitelist_for,
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
)

#: The agentic record type, named once: the board branches on it, the
#: ``kinds=`` default below carries it, and the axis-9/10 rows of the report
#: are populated only by rows that claim it.
AGENTIC_KIND = "agentic"

#: What every agentic row must carry before any transcript is read: the
#: prose fields' coordinates plus the two the replay and the registry audit
#: cannot do without (the advertised schemas, and the calls the transcript
#: claims ran -- the second is redundant with the transcript and checked as
#: consistency, which is exactly why an ``answer`` can be tampered with here
#: and still be caught).
_AGENTIC_ROW_REQUIRED = (
    "id",
    "record_type",
    "work_type",
    "scenario_id",
    "variant",
    "question",
    "answer",
    "messages",
    "register",
    "verification",
    "tool_names",
    "tool_schemas",
)

#: What every prose row must carry before a model's word is even read.
#: ``variant`` + ``scenario_id`` are not decoration: without them the row
#: cannot be re-derived from the computers, and axis 2 is the axis the whole
#: v3 design rests on.
ROW_REQUIRED = (
    "id",
    "record_type",
    "work_type",
    "scenario_id",
    "variant",
    "question",
    "answer",
    "messages",
    "register",
    "verification",
)

#: What every exam row must carry beyond the shared coordinates: the item's
#: own anatomy. ``options``/``answer_value`` are redundant with the answer by
#: construction -- which is the point: the gate re-derives the item from the
#: pack, and every stored field that disagrees with the recomposition is a
#: tamper finding, not a style choice.
_EXAM_ROW_REQUIRED = ROW_REQUIRED + (
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
    "spec",
    "reference_code",
    "public_tests",
    "hidden_tests",
    "dirty_fixture",
    "limitations",
)


def _family_of(row: dict) -> str:
    return row["scenario_id"][len(row["work_type"]) + 1 :]


def _check_row(row: dict, dead_ids: frozenset[str]) -> dict[str, list[str]]:
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
        if roles[:2] != ["system", "user"] or (roles and roles[-1] != "assistant"):
            failures["schema"].append(
                f"an agentic transcript must open system/user and close on the "
                f"desk's answer; roles read {roles!r}"
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
        if roles != ["system", "user", "assistant"]:
            failures["schema"].append(
                f"an exam row must read system/user/assistant; roles read {roles!r}"
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
        if roles != ["system", "user", "assistant"]:
            failures["schema"].append(
                f"an implementation row must read system/user/assistant; roles "
                f"read {roles!r}"
            )
        if isinstance(answer, str) and roles[-1:] == ["assistant"]:
            if (row["messages"][-1] or {}).get("content") != answer:
                failures["schema"].append(
                    "answer and the assistant turn have drifted apart (verify one "
                    "text, ship another)"
                )
    else:
        if roles != ["system", "user", "assistant"]:
            failures["schema"].append(
                f"message roles are {roles!r}, not the prose triad system/user/assistant"
            )
        if not isinstance(answer, str) or not answer.strip():
            failures["schema"].append("answer is empty")
        elif roles == ["system", "user", "assistant"]:
            assistant = (row["messages"][2] or {}).get("content")
            if assistant != answer:
                failures["schema"].append(
                    "answer and the assistant turn have drifted apart (verify one "
                    "text, ship another)"
                )
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
    for token in invented_numbers(
        answer, pack.allowed_numbers, whitelist_for(pack_dict)
    ):
        failures["invented numbers"].append(f"{token!r} is not in the fact pack")
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


def verify_dir(
    out_dir: str,
    *,
    kinds: tuple[str, ...] = (*BRIEF_KINDS, AGENTIC_KIND, EXAM_KIND, IMPL_KIND),
) -> dict:
    """Run every implemented axis over ``<out>/sft/`` and report.

    Exit-0 discipline lives in the CLI; this function only states facts:
    ``{"ok", "rows", "axes": {name: {"checked", "failures": [{id, problem}],
    "note"?}}}`` -- the optional ``note`` is a measurement that could not be
    certified for want of support, printed by the CLI and never red.
    """
    axes_report = {name: {"checked": 0, "failures": []} for _, name in AXES}
    ok = True
    rows_seen = 0
    all_rows: list[dict] = []
    for kind in kinds:
        path = write.path_for("sft", kind, out_dir)
        try:
            rows = write.read_jsonl(path)
        except ValueError as exc:  # a corrupt shard is itself a finding
            axes_report["schema"]["failures"].append({"id": path, "problem": str(exc)})
            ok = False
            continue
        dead_ids = write.existing_ids(write.path_for("dead_letter", kind, out_dir))
        for row in rows:
            rows_seen += 1
            all_rows.append(row)
            # One audit per row, all axes read from it: ``_check_row``
            # re-derives the pack, and doing that once per axis multiplied the
            # bill by the number of axes for nothing.
            result = _check_row(row, dead_ids)
            for _, name in AXES:
                axes_report[name]["checked"] += 1
                for problem in result[name]:
                    axes_report[name]["failures"].append(
                        {"id": row.get("id", "?"), "problem": problem}
                    )
    _measure_shares(axes_report, all_rows)
    ok = ok and not any(a["failures"] for a in axes_report.values())
    return {"ok": ok, "rows": rows_seen, "axes": axes_report}
