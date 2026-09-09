"""Prose renderers: analysis, memo, grounded, critique, abstention (spec §5.5).

The repair loop in one paragraph (analysis spec §6.3 item 8, arch spec
§5.5): ask the routed teacher; run the prose gate; if it returns violations,
append the draft plus a repair turn naming every violation, drop the
temperature one notch, ask again -- ``config.PROSE_ATTEMPTS`` strikes total.
A row that survives is written to ``sft/``; a row that does not is written
whole (exchange and all) to ``dead_letter/``, never quietly, never retried
on a later run without the entry being deleted first. Three strikes is a
verdict about *this teacher on this brief*, and re-asking it at the same
temperatures would be hoping, not engineering.

Why the loop lives here and not in the client: the transport must keep
"one POST per call" or "what did the teacher answer" stops being a fact
(client.py's rule). The loop is a *generation* policy -- temperatures, repair
wording, strike count -- and those are corpus decisions, versioned with the
corpus. ``verification.render.attempts`` on every row is what makes that
auditable: an attempt histogram per lane is how you notice a teacher
degrading before 20k rows of it ship.

Holdout families are not rendered here: their packs are the gold bar
(PR4's eval slice), and prose for them must never pass through a training
shard -- the leakage axis would catch it, and refusing at the gate is
cheaper than catching it at the audit.
"""

from __future__ import annotations

import os

from .. import config
from .. import write
from ..seed import render_seed
from ..teacher import routing
from ..teacher.client import DEFAULT_MAX_TOKENS
from ..teacher.prompts import BRIEF_KINDS, render_brief, render_repair
from ..verification.prose import gate_violations

#: Fields a stored fact-pack line carries *around* the pack proper. Stripped
#: before the pack is handed to the brief builder: the teacher must see
#: exactly :meth:`FactPack.to_dict`, byte-for-byte the dict the replay
#: fixture hashed it as -- envelope fields in the brief would silently
#: invalidate every committed fixture entry.
_PACK_ENVELOPE = ("id", "verification")


def _family_of(pack: dict) -> str:
    """The pack's scenario family, recovered from its ``scenario_id``."""
    return pack["scenario_id"][len(pack["work_type"]) + 1 :]


def row_id_from_coords(kind: str, work_type: str, family: str, variant: int) -> str:
    """The id one (kind, work_type, family, variant) is *always* called.

    Public because ``verify_v3`` re-derives it from a row's coordinates: an
    id is the corpus's word for a coordinate, and a row that has been pasted
    between coordinates -- or hand-written -- fails this the same way a
    checksum fails, without anybody arguing about it.
    """
    return config.supervised_id(kind, render_seed(work_type, family, kind, variant))


def row_id(kind: str, pack: dict) -> str:
    return row_id_from_coords(
        kind, pack["work_type"], _family_of(pack), pack["variant"]
    )


def select_prose_jobs(jobs, *, types=None, limit=None) -> list:
    """The render-eligible slice of *jobs*: prose kinds, train only, ordered.

    Split out of the stage because the fixture harness must capture replies
    for *exactly* the jobs a run will ask about -- the replay keys are the
    request hashes, so an off-by-one in this filter is a fixture miss, which
    is precisely the loud kind of bug a shared selector prevents. Holdout
    families are dropped here, once: the gold bar renders through PR4's
    eval path, never through ``sft/``.
    """
    wanted = set(types) if types is not None else set(BRIEF_KINDS)
    unknown = wanted - set(BRIEF_KINDS)
    if unknown:
        raise ValueError(
            f"render handles prose types only {BRIEF_KINDS}; asked for "
            + ", ".join(sorted(unknown))
        )
    selected = sorted(
        (job for job in jobs if job.record_type in wanted and not job.holdout),
        key=lambda job: (job.work_type, job.family, job.record_type, job.variant),
    )
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def render_prose_row(teacher, pack_line: dict, *, kind: str) -> dict:
    """One stored pack line + one kind -> ``{"row": .., "dead_letter": ..}``.

    Never raises for a *content* failure -- a violation is data and goes to
    the dead letter. ``TeacherError`` (the endpoint misbehaved) propagates on
    purpose: the transport breaking is an outage to retry the run over, not
    a reason to dead-letter three thousand good scenarios.
    """
    if kind not in BRIEF_KINDS:
        raise ValueError(
            f"no prose renderer for record type {kind!r} (known: {BRIEF_KINDS})"
        )
    pack = {k: v for k, v in pack_line.items() if k not in _PACK_ENVELOPE}
    stamp = pack_line.get("verification") or {}
    route = routing.route(kind)
    messages = render_brief(pack, kind=kind)
    exchange: list[dict] = list(messages)
    violations: list[str] = []
    attempts = 0
    truncations = 0
    while attempts < config.PROSE_ATTEMPTS:
        temperature = config.PROSE_TEMPERATURES[attempts]
        # Two different failures share this ladder, and they want opposite
        # responses. A draft that broke the contract is usually the model
        # padding, and a cooler temperature helps. A draft that came back
        # *empty* is a reasoning teacher that spent its whole completion budget
        # thinking and never reached the answer -- cooling that does nothing at
        # all, and the first live runs proved it: three attempts, three empty
        # drafts, one dead letter, repeatedly. Truncation gets headroom instead.
        result = teacher.complete(
            exchange,
            model=route.model,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS * (1 + truncations),
            think=route.think,
        )
        attempts += 1
        if not (result.text or "").strip():
            truncations += 1
        violations = gate_violations(pack, result.text, kind)
        if not violations:
            return {
                "row": {
                    "id": row_id(kind, pack),
                    "record_type": kind,
                    "work_type": pack["work_type"],
                    "scenario_id": pack["scenario_id"],
                    "variant": pack["variant"],
                    "question": pack["question"],
                    "messages": [
                        *messages,
                        {"role": "assistant", "content": result.text},
                    ],
                    "answer": result.text,
                    "register": pack["register"],
                    "verification": {
                        "computed_by": stamp.get("computed_by", "unknown"),
                        "pack_seed": stamp.get("pack_seed", f"{pack['seed']:016x}"),
                        "teacher": result.to_verification(),
                        "render": {"kind": kind, "attempts": attempts},
                    },
                },
                "dead_letter": None,
            }
        exchange = render_repair(exchange, result.text, violations)
    return {
        "row": None,
        "dead_letter": {
            "id": row_id(kind, pack),
            "record_type": kind,
            "reason": "gate: " + " | ".join(violations),
            "work_type": pack["work_type"],
            "scenario_id": pack["scenario_id"],
            "attempts": attempts,
            "temperatures": list(config.PROSE_TEMPERATURES[:attempts]),
            "violations": violations,
            "exchange": exchange,
        },
    }


def run_render_stage(out_dir: str, jobs, teacher, *, types=None, limit=None) -> dict:
    """Render prose rows for *jobs*; write ``sft/`` + ``dead_letter/``; report.

    Same shape as ``stage.run_pack_stage`` on purpose -- filter by types
    *before* ``limit``, both before touching the disk; ids decide what is
    asked (a job already in ``sft`` or ``dead_letter`` is not asked again:
    the teacher bill is the budget this whole design defends), and the
    report is what the CLI prints and the CI gate reads.

    Two absences are not the same event: a work type with **no fact-pack
    file at all** means the packs stage never covered this plan -- a DAG
    ordering bug, ``missing_packs``, fatal. A *variant* absent from a
    present file is the pack guard's own verdict (``PackError`` on a
    parameter set that describes no scenario, already reported by the packs
    stage) -- ``skipped_by_pack_gate``, a recorded skip, not an error. The
    fact never existed; there is nothing to render and nobody to blame.
    """
    selected = select_prose_jobs(jobs, types=types, limit=limit)

    pack_cache: dict[str, dict[tuple[str, str, int], dict] | None] = {}

    def pack_for(job) -> tuple[dict | None, bool]:
        """``(line, file_existed)`` -- see the docstring's two absences."""
        cached = pack_cache.get(job.work_type)
        if cached is None:
            path = write.path_for("fact_packs", job.work_type, out_dir)
            existed = os.path.isfile(path)
            index = {}
            if existed:
                for line in write.read_jsonl(path):
                    index[(line["work_type"], _family_of(line), line["variant"])] = line
            pack_cache[job.work_type] = index if existed else False
            cached = index if existed else False
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
    }
    rows: dict[str, list[dict]] = {}
    deads: dict[str, list[dict]] = {}
    known: dict[str, frozenset[str]] = {}
    seen_ids: set[str] = set()
    for job in selected:
        kind = job.record_type
        rid = row_id_from_coords(kind, job.work_type, job.family, job.variant)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        if kind not in known:
            known[kind] = write.existing_ids(
                write.path_for("sft", kind, out_dir)
            ) | write.existing_ids(write.path_for("dead_letter", kind, out_dir))
        if rid in known[kind]:
            report["existing"] += 1
            report["by_kind"].setdefault(kind, {"rendered": 0, "existing": 0})[
                "existing"
            ] += 1
            continue
        pack_line, file_existed = pack_for(job)
        if pack_line is None:
            entry = {
                "id": rid,
                "work_type": job.work_type,
                "family": job.family,
                "record_type": kind,
                "variant": job.variant,
            }
            bucket = (
                report["skipped_by_pack_gate"]
                if file_existed
                else report["missing_packs"]
            )
            entry["reason"] = (
                "the pack guard rejected this parameter set (PackError at the "
                "packs stage); the fact never existed, nothing to render"
                if file_existed
                else "no fact-pack file for this work type at all; run the packs "
                "stage for this plan before rendering"
            )
            bucket.append(entry)
            continue
        outcome = render_prose_row(teacher, pack_line, kind=kind)
        cell = report["by_kind"].setdefault(kind, {"rendered": 0, "existing": 0})
        if outcome["row"] is not None:
            rows.setdefault(kind, []).append(outcome["row"])
            report["rendered"] += 1
            cell["rendered"] += 1
        else:
            deads.setdefault(kind, []).append(outcome["dead_letter"])
            report["dead_lettered"] += 1
    for kind in sorted(rows):
        write.append_unique(write.path_for("sft", kind, out_dir), rows[kind])
    for kind in sorted(deads):
        write.append_unique(write.path_for("dead_letter", kind, out_dir), deads[kind])
    return report
