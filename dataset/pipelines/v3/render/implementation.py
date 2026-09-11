"""The implementation renderer: composed record, authored limitations, proven
suite (spec §5.9).

Everything the row states about the instrument -- spec, question, reference,
suites, dirty fixture -- is :func:`compose_impl_item`'s, i.e. the pack's; the
teacher authors exactly one field, the statement of limitations, under the
same repair loop the prose lanes run (``config.IMPL_ATTEMPTS`` strikes, the
temperature ladder). The reference implementation is *not* asked for: it is
the reviewed table's, and a teacher-written answer next to a suite generated
from the same pack would be a collaboration nobody audited -- the shipped
``answer`` is the table's own instrument, byte for byte.

Two gates, in a deliberate order:

* the **sandbox runs before the teacher is billed**. A pack whose composed
  suite does not pass against the reference is a defect of the *table*, not
  a draft to repair; asking a model to fix limitations around an instrument
  that fails its own tests would bill for a dead row, so the stage measures
  first and dead-letters the pack with the failures logged;
* the **limitations gate is the teacher's** (:func:`limitations_violations`):
  fact-locked, floor-lengthed, forbidden-claim-free. It is the only field
  with an author, so it is the only field with attempts.
"""

from __future__ import annotations

import os

from .. import config
from .. import row as rowlib
from .. import write
from ..teacher import routing
from ..verification.implementation import (
    IMPL_KIND,
    compose_impl_item,
    impl_brief,
    limitations_violations,
    run_sandboxed,
)
from .prose import _PACK_ENVELOPE, row_id_from_coords


def select_impl_jobs(jobs, *, limit=None, holdout=False) -> list:
    """The render-eligible implementation slice: one cohort, sorted, capped."""
    selected = sorted(
        (
            job
            for job in jobs
            if job.record_type == IMPL_KIND and bool(job.holdout) is bool(holdout)
        ),
        key=lambda job: (job.work_type, job.family, job.record_type, job.variant),
    )
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _family(pack: dict) -> str:
    return pack["scenario_id"][len(pack["work_type"]) + 1 :]


def build_impl_row(teacher, pack_line: dict, *, holdout: bool = False) -> dict:
    """One stored pack line -> ``{"row": .., "dead_letter": ..}``.

    The pair shape of the prose builder; the failure taxonomy is richer:
    ``composition`` (the pack entails no record), ``sandbox`` (the composed
    suite does not pass the table's own instrument -- a table defect, no
    attempts spent), ``gate`` (the authored limitations drew violations after
    every attempt). ``TeacherError`` propagates: an outage is retried over
    the run, not dead-lettered through the corpus.
    """
    pack = {k: v for k, v in pack_line.items() if k not in _PACK_ENVELOPE}
    stamp = pack_line.get("verification") or {}
    rid = row_id_from_coords(
        IMPL_KIND, pack["work_type"], _family(pack), pack["variant"]
    )

    def dead(reason: str, **extra) -> dict:
        return {
            "row": None,
            "log": None,
            "dead_letter": {
                "id": rid,
                "record_type": IMPL_KIND,
                "reason": reason,
                "work_type": pack["work_type"],
                "scenario_id": pack["scenario_id"],
                **extra,
            },
        }

    try:
        item = compose_impl_item(pack)
    except ValueError as exc:
        return dead(f"composition: {exc}")

    passed, log = run_sandboxed(
        item["reference_code"],
        [*item["public_tests"], *item["hidden_tests"]],
        pack.get("inputs") or {},
    )
    if not passed:
        # A table defect, not a draft: no teacher was asked, nothing to repair.
        return dead(f"sandbox: {log}")

    route = routing.route(IMPL_KIND)
    brief = impl_brief(pack, item)
    exchange = list(brief)
    violations: list[str] = []
    attempts = 0
    while attempts < config.IMPL_ATTEMPTS:
        temperature = config.IMPL_TEMPERATURES[attempts]
        result = teacher.complete(
            exchange,
            model=route.model,
            temperature=temperature,
            max_tokens=route.max_tokens,
            think=route.think,
        )
        attempts += 1
        limitations = (result.text or "").strip()
        violations = limitations_violations(pack, limitations)
        if not violations:
            return {
                "row": rowlib.student_row(
                    row_id=rid,
                    kind=IMPL_KIND,
                    pack=pack,
                    holdout=holdout,
                    answer=item["reference_code"],
                    stamp=stamp,
                    teacher=result.to_verification(),
                    render={
                        "kind": IMPL_KIND,
                        "attempts": attempts,
                        "public_tests": len(item["public_tests"]),
                        "hidden_tests": len(item["hidden_tests"]),
                        "sandbox": "passed",
                    },
                    attempts=attempts,
                    extra={
                        # The spec is the *record's* prompt and it is the pack's,
                        # not the teacher's: `impl_brief` wraps it in the factory
                        # contract to ask for limitations, and that wrapper is
                        # what may not ship. The two turns kept here are the
                        # instrument's own -- ask and reference implementation.
                        "messages": [
                            {"role": "user", "content": item["spec"]},
                            {
                                "role": "assistant",
                                "content": item["reference_code"],
                            },
                        ],
                        "reference_code": item["reference_code"],
                        "spec": item["spec"],
                        "public_tests": item["public_tests"],
                        "hidden_tests": item["hidden_tests"],
                        "dirty_fixture": item["dirty_fixture"],
                        "limitations": limitations,
                    },
                ),
                "dead_letter": None,
                "log": rowlib.teacher_log(
                    row_id=rid,
                    kind=IMPL_KIND,
                    pack=pack,
                    messages=[*brief, {"role": "assistant", "content": limitations}],
                    result=result,
                ),
            }
        exchange = [
            *exchange,
            {"role": "assistant", "content": limitations},
            {
                "role": "user",
                "content": (
                    "Your passage drew these objections: "
                    + "; ".join(violations)
                    + ". Restate the limitations without them."
                ),
            },
        ]
    return dead(
        "gate: " + " | ".join(violations),
        attempts=attempts,
        temperatures=list(config.IMPL_TEMPERATURES[:attempts]),
        violations=violations,
        exchange=exchange,
    )


def run_impl_stage(out_dir: str, jobs, teacher, *, limit=None, holdout=False) -> dict:
    """Render implementation rows for *jobs*; write the shard + ``dead_letter/``."""
    selected = select_impl_jobs(jobs, limit=limit, holdout=holdout)
    shard = write.shard_kind(holdout)
    keep_logs = config.keep_teacher_messages()
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
                index[(line["work_type"], _family(line), line["variant"])] = line
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
        "bucket": shard,
        "teacher_logs": 0,
    }
    rows: list[dict] = []
    deads: list[dict] = []
    known = write.existing_ids(write.path_for(shard, IMPL_KIND, out_dir)) | (
        write.existing_ids(write.path_for("dead_letter", IMPL_KIND, out_dir))
    )
    seen: set[str] = set()
    for job in selected:
        rid = row_id_from_coords(IMPL_KIND, job.work_type, job.family, job.variant)
        if rid in seen:
            continue
        seen.add(rid)
        if rid in known:
            report["existing"] += 1
            report["by_kind"].setdefault(IMPL_KIND, {"rendered": 0, "existing": 0})[
                "existing"
            ] += 1
            continue
        pack_line, file_existed = pack_for(job)
        if pack_line is None:
            entry = {
                "id": rid,
                "work_type": job.work_type,
                "family": job.family,
                "record_type": IMPL_KIND,
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
        outcome = build_impl_row(teacher, pack_line, holdout=job.holdout)
        cell = report["by_kind"].setdefault(IMPL_KIND, {"rendered": 0, "existing": 0})
        if outcome["row"] is not None:
            rows.append(outcome["row"])
            report["rendered"] += 1
            cell["rendered"] += 1
            if keep_logs and outcome.get("log"):
                write.write_teacher_log(
                    write.teacher_log_path(IMPL_KIND, rid, out_dir), outcome["log"]
                )
                report["teacher_logs"] += 1
        else:
            deads.append(outcome["dead_letter"])
            report["dead_lettered"] += 1
    if rows:
        write.append_unique(write.path_for(shard, IMPL_KIND, out_dir), rows)
    if deads:
        write.append_unique(write.path_for("dead_letter", IMPL_KIND, out_dir), deads)
    return report
