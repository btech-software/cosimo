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

Axes 6-13 (shares, tool schemas, hidden tests, dedup, gold-bar near-dup,
teacher pinning) join with the stages that produce their inputs, PR3/PR4;
``--quick`` will skip 10-12 then, exactly as spec'd, and the flag is
accepted today so DAG command lines never change shape.

The report is a plain dict, not a printed table: Airflow needs exit codes,
CI needs numbers, and the human-readable board is one ``print`` away in the
CLI -- formatting is the CLI's job, not this one's.
"""

from __future__ import annotations

from . import config, write
from .packs import PackError, compute_pack
from .render.prose import row_id
from .seed import pack_seed
from .teacher.prompts import BRIEF_KINDS
from .verification.prose import (
    FINAL_ANSWER_TAG,
    forbidden_hits,
    missing_mentions,
    whitelist_for,
)
from .verification.invented_numbers import invented_numbers

#: The axes implemented in PR2, in spec order, named as the board prints them.
AXES = (
    (1, "schema"),
    (2, "pack recompute"),
    (3, "invented numbers"),
    (4, "must_mention / forbidden_claims"),
    (5, "FINAL ANSWER is exam-only"),
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
    if roles != ["system", "user", "assistant"]:
        failures["schema"].append(
            f"message roles are {roles!r}, not the prose triad system/user/assistant"
        )
    answer = row.get("answer")
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
        # Axes 2-5 grade content, and content cannot be located without a
        # shape. Report the break once; do not stack four derivative errors.
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


def verify_dir(out_dir: str, *, kinds: tuple[str, ...] = BRIEF_KINDS) -> dict:
    """Run every implemented axis over ``<out>/sft/`` and report.

    Exit-0 discipline lives in the CLI; this function only states facts:
    ``{"ok", "rows", "axes": {name: {"checked", "failures": [{id, problem}]}}}``.
    """
    axes_report = {name: {"checked": 0, "failures": []} for _, name in AXES}
    ok = True
    rows_seen = 0
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
            for _, name in AXES:
                axes_report[name]["checked"] += 1
                for problem in _check_row(row, dead_ids)[name]:
                    axes_report[name]["failures"].append(
                        {"id": row.get("id", "?"), "problem": problem}
                    )
    ok = ok and not any(a["failures"] for a in axes_report.values())
    return {"ok": ok, "rows": rows_seen, "axes": axes_report}
