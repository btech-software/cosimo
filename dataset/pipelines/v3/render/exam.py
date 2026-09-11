"""The exam renderer: composed locally, no teacher in the room (spec §6.1).

The DAG's ``render_exam`` box is the one render stage that needs no LLM: the
pack already contains the question, the answer and the numbers a wrong model
could print, and the §5.10 rules -- sampled trace shape, pitfall distractors,
the ``FINAL ANSWER:`` closing -- are composition rules, not style. An optional
teacher trace-phrasing pass exists in the spec ("LLM only for trace phrasing
with locked numbers"); it is deliberately absent here: a stage that could
quietly start calling the teacher would make the byte-identity of the exam
slice -- the slice the board re-audits by recomposition -- a property of the
endpoint instead of a property of the corpus.

The stage mirrors :mod:`pipelines.v3.render.prose` where the shapes can
mirror: the same selection discipline (train rows only, sorted, ``limit``
after selection), the same two absences (no pack *file* is the DAG out of
order; a variant the pack guard refused is that guard's own verdict), the
same id-gated resume, the same ``append_unique`` writing. It differs in the
one thing it must never do: *re-ask*. A composed row either satisfies
:func:`verification.exam.compose_exam_item` or the composition raised; there
is no repair loop because there is no author to repair -- a collapse here is
a defect of the *pack*, and the dead letter says so.
"""

from __future__ import annotations

import os

from .. import row as rowlib
from .. import write
from ..verification.exam import EXAM_KIND, compose_exam_item
from .prose import _PACK_ENVELOPE, _family_of, row_id_from_coords

#: The system turn of an exam row: the protocol the student is trained to
#: answer in. Short on purpose -- §8.1 asks for a shortened identity, and an
#: exam item's contract is the answer form, not a persona.
EXAM_PROTOCOL = (
    "You are sitting a Cosimo verification item. Work the instrument the way "
    "the formula says; every figure you state must be one the statement "
    "authorises. Choose an option. Close with the line "
    "'FINAL ANSWER: <letter> -- <value> <unit>' and nothing after it."
)


def select_exam_jobs(jobs, *, limit=None, holdout=False) -> list:
    """The render-eligible exam slice of *jobs*: one cohort, sorted, capped.

    The prose selector's discipline, unchanged, for the same reasons (the
    fixture harnesses capture replies for exactly what a run will ask); the
    exam slice is deterministic, so a captured run and a live run must ask
    the same questions of the same packs. ``holdout`` selects the cohort
    rather than filtering one away -- see ``prose.select_prose_jobs``.
    """
    selected = sorted(
        (
            job
            for job in jobs
            if job.record_type == EXAM_KIND and bool(job.holdout) is bool(holdout)
        ),
        key=lambda job: (job.work_type, job.family, job.record_type, job.variant),
    )
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def build_exam_row(pack_line: dict, *, holdout: bool = False) -> dict:
    """One stored pack line -> ``{"row": .., "dead_letter": ..}``, no I/O.

    The pair shape of :func:`pipelines.v3.render.prose.render_prose_row`
    (exactly one of the two keys is non-``None``), but there is no attempt
    loop: the composition is a pure function of the pack, so "try again" is
    "compute the same bytes again". The only way this row dies is a pack
    whose pitfall derivations collapsed to fewer than the floor of distinct
    distractors -- and the dead letter records that as a *pack* finding.
    """
    pack = {k: v for k, v in pack_line.items() if k not in _PACK_ENVELOPE}
    stamp = pack_line.get("verification") or {}
    try:
        item = compose_exam_item(pack)
    except ValueError as exc:  # the floor of distinct distractors, mainly
        return {
            "row": None,
            "dead_letter": {
                "id": row_id_from_coords(
                    EXAM_KIND, pack["work_type"], _family_of(pack), pack["variant"]
                ),
                "record_type": EXAM_KIND,
                "reason": f"composition: {exc}",
                "work_type": pack["work_type"],
                "scenario_id": pack["scenario_id"],
            },
        }
    # No system turn on the shipped row. ``EXAM_PROTOCOL`` is the *factory's*
    # instruction about answer form, and the harness composes its own from
    # `prompt.exam_protocol` -- binding both would put two protocols in front
    # of one item, which is what `normalize_v3_record` was already stripping
    # back out on the way in. The item's prompt (the four labelled options)
    # lives in a named field instead of at message index 1: a column an
    # auditor can read beats an index into a list.
    messages = [
        {"role": "user", "content": item["question_text"]},
        {"role": "assistant", "content": item["answer_text"]},
    ]
    return {
        "row": rowlib.student_row(
            row_id=row_id_from_coords(
                EXAM_KIND, pack["work_type"], _family_of(pack), pack["variant"]
            ),
            kind=EXAM_KIND,
            pack=pack,
            holdout=holdout,
            answer=item["answer_text"],
            stamp=stamp,
            teacher=None,  # composed, not dictated: the row says so
            render={
                "kind": EXAM_KIND,
                "attempts": 0,
                "style": item["style"],
                "liturgy": item["liturgy"],
                "distractors": item["distractor_names"],
            },
            attempts=0,
            extra={
                "question_text": item["question_text"],
                "messages": messages,
                "options": item["options"],
                "answer_value": item["answer_value"],
                "answer_key": item["answer_key"],
                "unit": item["unit"],
            },
        ),
        "dead_letter": None,
    }


def run_exam_stage(out_dir: str, jobs, *, limit=None, holdout=False) -> dict:
    """Compose exam rows for *jobs*; write the shard + ``dead_letter/``; report."""
    selected = select_exam_jobs(jobs, limit=limit, holdout=holdout)
    shard = write.shard_kind(holdout)
    pack_cache: dict[str, dict | bool] = {}

    def pack_for(job):
        cached = pack_cache.get(job.work_type)
        if cached is None:
            path = write.path_for("fact_packs", job.work_type, out_dir)
            if not os.path.isfile(path):
                pack_cache[job.work_type] = False
                return None, False
            index = {}
            for line in write.read_jsonl(path):
                index[(line["work_type"], _family_of(line), line["variant"])] = line
            pack_cache[job.work_type] = index
            cached = index
        if cached is False:
            return None, False
        return cached.get((job.work_type, job.family, job.variant)), True

    report = {
        "jobs_seen": len(selected),
        "rendered": 0,
        "existing": 0,
        "dead_lettered": 0,
        "skipped_by_pack_gate": [],
        "missing_packs": [],
        "by_kind": {},
        "liturgy": 0,
        "bucket": shard,
        "teacher_logs": 0,
    }
    rows: list[dict] = []
    deads: list[dict] = []
    known = write.existing_ids(write.path_for(shard, EXAM_KIND, out_dir)) | (
        write.existing_ids(write.path_for("dead_letter", EXAM_KIND, out_dir))
    )
    seen: set[str] = set()
    for job in selected:
        rid = row_id_from_coords(EXAM_KIND, job.work_type, job.family, job.variant)
        if rid in seen:
            continue
        seen.add(rid)
        if rid in known:
            report["existing"] += 1
            report["by_kind"].setdefault(EXAM_KIND, {"rendered": 0, "existing": 0})[
                "existing"
            ] += 1
            continue
        pack_line, file_existed = pack_for(job)
        if pack_line is None:
            entry = {
                "id": rid,
                "work_type": job.work_type,
                "family": job.family,
                "record_type": EXAM_KIND,
                "variant": job.variant,
                "reason": (
                    "the pack guard rejected this parameter set (PackError at "
                    "the packs stage); the fact never existed, nothing to compose"
                    if file_existed
                    else "no fact-pack file for this work type at all; run the "
                    "packs stage for this plan before rendering"
                ),
            }
            (
                report["skipped_by_pack_gate"]
                if file_existed
                else report["missing_packs"]
            ).append(entry)
            continue
        outcome = build_exam_row(pack_line, holdout=job.holdout)
        cell = report["by_kind"].setdefault(EXAM_KIND, {"rendered": 0, "existing": 0})
        if outcome["row"] is not None:
            rows.append(outcome["row"])
            report["rendered"] += 1
            cell["rendered"] += 1
            if outcome["row"]["verification"]["render"]["liturgy"]:
                report["liturgy"] += 1
        else:
            deads.append(outcome["dead_letter"])
            report["dead_lettered"] += 1
    if rows:
        write.append_unique(write.path_for(shard, EXAM_KIND, out_dir), rows)
    if deads:
        write.append_unique(write.path_for("dead_letter", EXAM_KIND, out_dir), deads)
    return report
