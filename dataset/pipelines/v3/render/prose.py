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

A dead letter carries the whole ladder, not its last rung. Recording only
the final round was actively misleading: a run whose drafts ran [0 words,
163 words, 0 words] filed a post-mortem reading "0 words", hiding both the
one real draft and the only violations anybody could act on. ``history``
holds every attempt's temperature, budget, finish reason, word count and
verdict; ``best_attempt`` names the round worth reading; and ``reason``
distinguishes a teacher that broke the contract from one that never
finished thinking, because those want opposite fixes.

Holdout families **are** rendered here now, and that is the amendment's §E
change. They were dropped outright before, on the reasoning that prose for a
gold-bar family must never pass through a training shard -- which is true, and
which dropping them enforced by producing nothing at all. The cost was that
"unseen scenario family" had nothing to measure: the harness had to hold out
*shipped* families instead, so the generalisation number was taken on families
the model had trained on. A holdout job renders exactly like a train one and
lands in ``eval/`` rather than ``sft/`` -- one directory apart, which a glob
cannot confuse the way a boolean on a row could.

Two objects come out of a render, and only one is trainable. See
:mod:`pipelines.v3.row`: the student row carries the pack's question and the
visible answer, the teacher log carries the brief and the think text, and the
log is written only when ``COSIMO_V3_KEEP_TEACHER_MESSAGES=1``.
"""

from __future__ import annotations

import os

from .. import config
from .. import row as rowlib
from .. import write
from ..seed import render_seed
from ..teacher import routing
from ..teacher.prompts import BRIEF_KINDS, render_brief, render_repair
from ..verification.prose import gate_violations, missing_mentions

#: Fields a stored fact-pack line carries *around* the pack proper. Stripped
#: before the pack is handed to the brief builder: the teacher must see
#: exactly :meth:`FactPack.to_dict`, byte-for-byte the dict the replay
#: fixture hashed it as -- envelope fields in the brief would silently
#: invalidate every committed fixture entry.
#:
#: Re-exported from :mod:`pipelines.v3.row`, which is where the two-surface
#: split put it: three other renderers import it from here, and a second
#: definition of "what is envelope" is a second answer to the question.
_PACK_ENVELOPE = rowlib.PACK_ENVELOPE
_family_of = rowlib.family_of


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


def select_prose_jobs(jobs, *, types=None, limit=None, holdout=False) -> list:
    """The render-eligible slice of *jobs*: prose kinds, one cohort, ordered.

    Split out of the stage because the fixture harness must capture replies
    for *exactly* the jobs a run will ask about -- the replay keys are the
    request hashes, so an off-by-one in this filter is a fixture miss, which
    is precisely the loud kind of bug a shared selector prevents.

    ``holdout`` picks the cohort rather than filtering one away, which is the
    amendment's §E change. Holdout families used to be dropped here outright,
    and the consequence surfaced two subsystems later: the harness's
    ``holdout_scenario_families`` had to be pointed at *shipped* families
    instead, because ``fade_required`` and its four siblings had fact packs on
    disk and no rendered rows anywhere -- there was nothing downstream to hold
    out, so the generalisation measurement was being taken on families the
    model had trained on. They render now, and they render to ``eval/``.
    """
    wanted = set(types) if types is not None else set(BRIEF_KINDS)
    unknown = wanted - set(BRIEF_KINDS)
    if unknown:
        raise ValueError(
            f"render handles prose types only {BRIEF_KINDS}; asked for "
            + ", ".join(sorted(unknown))
        )
    selected = sorted(
        (
            job
            for job in jobs
            if job.record_type in wanted and bool(job.holdout) is bool(holdout)
        ),
        key=lambda job: (job.work_type, job.family, job.record_type, job.variant),
    )
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _best_attempt(history: list[dict]) -> dict | None:
    """The attempt worth reading, or ``None`` if the teacher never wrote a word.

    "Best" is the non-empty draft with the fewest violations, latest wins a
    tie -- a later attempt has seen the repair feedback, so when two rounds
    break the same number of rules the later one is the more informative
    failure. Empty drafts are not candidates at all: a truncated round has no
    faults of its own, only the word-budget floor its emptiness trips, and
    letting it win the comparison is exactly how the real draft got lost.
    """
    scored = [h for h in history if h["words"]]
    if not scored:
        return None
    return min(scored, key=lambda h: (len(h["violations"]), -h["attempt"]))


def _dead_letter_reason(history: list[dict], best: dict | None) -> str:
    """One line saying which of the two failures this was.

    A teacher that cannot obey the contract and a teacher that never finished
    thinking need opposite fixes -- reword the pack, or raise the budget --
    and a post-mortem that files both under "gate: word budget" costs a day
    to tell apart.
    """
    if best is None:
        budgets = " -> ".join(str(h["max_tokens"]) for h in history)
        return (
            f"truncated: {len(history)} attempts returned no text at all "
            f"(budgets {budgets}); the teacher never reached an answer"
        )
    return "gate: " + " | ".join(best["violations"])


def render_prose_row(
    teacher, pack_line: dict, *, kind: str, holdout: bool = False
) -> dict:
    """One stored pack line + one kind -> ``{"row", "dead_letter", "log"}``.

    Never raises for a *content* failure -- a violation is data and goes to
    the dead letter. ``TeacherError`` (the endpoint misbehaved) propagates on
    purpose: the transport breaking is an outage to retry the run over, not
    a reason to dead-letter three thousand good scenarios.

    Three keys now, not two. ``row`` is the student surface and carries no
    ``messages`` at all: a prose row's prompt *is* its question, and the brief
    that produced it -- the fact pack as JSON, the number policy, the word
    budget, the teacher's own system turn -- was never anything a student
    should be shown. ``log`` is that brief, for the operator who asked to keep
    it; the stage decides whether it reaches the disk.
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
    history: list[dict] = []
    attempts = 0
    truncations = 0
    while attempts < config.PROSE_ATTEMPTS:
        temperature = config.PROSE_TEMPERATURES[attempts]
        # The budget is the lane's, flat across attempts, and the escalating
        # ladder that used to live here is gone with the thing that forced it.
        #
        # That ladder existed for one failure: a *reasoning* teacher on the
        # prose lane spent its whole completion budget thinking and returned
        # `content: null`, so a later attempt -- carrying the failed draft and
        # the violation list on top of the brief -- needed more room than the
        # one before it. Doubling per attempt was the fix, and it worked. But
        # the cause was think=True on analysis, memo and grounded, which the
        # amendment removes: a lane that does not reason cannot run out of
        # budget before it reaches an answer, and 800 tokens is twice the
        # 400-word ceiling the widest prose band allows.
        #
        # Cooling the temperature stays, because the *other* failure the ladder
        # answered is real and unchanged: a draft that broke the contract is
        # usually the model padding, and cold models pad less.
        budget = routing.budget_for_attempt(route, truncations=truncations)
        result = teacher.complete(
            exchange,
            model=route.model,
            temperature=temperature,
            max_tokens=budget,
            think=route.think,
        )
        text = (result.text or "").strip()

        # A truncation is not a strike. The teacher ran out of room before it
        # wrote a word, which says nothing about whether it *can* satisfy the
        # contract -- and spending one of three gate attempts on it, then
        # sending a repair turn that quotes an empty draft, is how nine live
        # rows cost forty-eight minutes and produced nothing.
        #
        # So: give it more room, on the same brief, without cooling the
        # temperature (there is no draft to cool *toward*) and without counting
        # it against the contract. The lane's floor rises with it, so the next
        # row opens where this one ended up rather than rediscovering it.
        if routing.truncated(result) and truncations < config.PROSE_TRUNCATION_RETRIES:
            truncations += 1
            floor = routing.note_truncation(route.lane, budget)
            history.append(
                {
                    "attempt": f"{attempts + 1}t{truncations}",
                    "temperature": temperature,
                    "max_tokens": budget,
                    "finish_reason": result.finish_reason,
                    "words": 0,
                    "violations": [],
                    "note": f"truncated before writing; lane floor now {floor}",
                }
            )
            continue

        attempts += 1
        violations = gate_violations(pack, result.text, kind)
        history.append(
            {
                "attempt": attempts,
                "temperature": temperature,
                "max_tokens": budget,
                "finish_reason": result.finish_reason,
                "words": len(text.split()),
                "violations": violations,
            }
        )
        if not violations:
            rid = row_id(kind, pack)
            return {
                "row": rowlib.student_row(
                    row_id=rid,
                    kind=kind,
                    pack=pack,
                    holdout=holdout,
                    answer=rowlib.visible_answer(result.text),
                    stamp=stamp,
                    teacher=result.to_verification(),
                    render={"kind": kind, "attempts": attempts},
                    attempts=attempts,
                    # A shipped row cleared every axis by definition; recording
                    # the two lists empty is what lets the eval sidecar read a
                    # row's compliance without recomputing the pack to re-run
                    # the gate. `missing_mentions` is re-read rather than
                    # assumed so the field means "measured", not "assumed".
                    invented=[],
                    missing=missing_mentions(pack, result.text),
                    register_ok=True,
                ),
                "dead_letter": None,
                "log": rowlib.teacher_log(
                    row_id=rid,
                    kind=kind,
                    pack=pack,
                    messages=[
                        *exchange,
                        {"role": "assistant", "content": result.text},
                    ],
                    result=result,
                    history=history,
                ),
            }
        exchange = render_repair(exchange, result.text, violations)
    best = _best_attempt(history)
    return {
        "row": None,
        "log": None,
        "dead_letter": {
            "id": row_id(kind, pack),
            "record_type": kind,
            "reason": _dead_letter_reason(history, best),
            "work_type": pack["work_type"],
            "scenario_id": pack["scenario_id"],
            "attempts": attempts,
            "temperatures": list(config.PROSE_TEMPERATURES[:attempts]),
            "violations": (best or history[-1])["violations"],
            "best_attempt": None if best is None else best["attempt"],
            "history": history,
            "exchange": exchange,
        },
    }


def run_render_stage(
    out_dir: str, jobs, teacher, *, types=None, limit=None, holdout=False
) -> dict:
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
    selected = select_prose_jobs(jobs, types=types, limit=limit, holdout=holdout)
    shard = write.shard_kind(holdout)
    keep_logs = config.keep_teacher_messages()

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

    # A coordinate the human set already holds is not one the corpus renders:
    # the same teacher, seed and brief writes very nearly the same row, and
    # axis 13 calls that a leak of the held-out set into training. Read once
    # per stage rather than per job -- it is one small file.
    gold_barred = write.gold_bar_ids()

    report = {
        "jobs_seen": len(selected),
        "rendered": 0,
        "existing": 0,
        "gold_barred": 0,
        "dead_lettered": 0,
        "skipped_by_pack_gate": [],
        "missing_packs": [],
        "by_kind": {},
        "bucket": shard,
        "teacher_logs": 0,
    }
    rows: dict[str, list[dict]] = {}
    deads: dict[str, list[dict]] = {}
    known: dict[str, frozenset[str]] = {}
    seen_ids: set[str] = set()
    try:
        for job in selected:
            kind = job.record_type
            rid = row_id_from_coords(kind, job.work_type, job.family, job.variant)
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            if kind not in known:
                known[kind] = write.existing_ids(
                    write.path_for(shard, kind, out_dir)
                ) | write.existing_ids(write.path_for("dead_letter", kind, out_dir))
            if rid in gold_barred:
                # Counted apart from `existing`, because the two are different
                # events: one says "already generated", the other says "never
                # generate this one".
                report["gold_barred"] += 1
                continue
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
            outcome = render_prose_row(
                teacher, pack_line, kind=kind, holdout=job.holdout
            )
            cell = report["by_kind"].setdefault(kind, {"rendered": 0, "existing": 0})
            if outcome["row"] is not None:
                rows.setdefault(kind, []).append(outcome["row"])
                report["rendered"] += 1
                cell["rendered"] += 1
                if keep_logs and outcome.get("log"):
                    # Written as the row is made rather than in the finally
                    # block: the log's whole value is post-mortem, and a run
                    # that dies mid-stage is exactly the run whose transcripts
                    # somebody wants.
                    write.write_teacher_log(
                        write.teacher_log_path(kind, rid, out_dir), outcome["log"]
                    )
                    report["teacher_logs"] += 1
            else:
                deads.setdefault(kind, []).append(outcome["dead_letter"])
                report["dead_lettered"] += 1
    finally:
        # Whatever the teacher already answered is written even when a later
        # call takes the stage down. A transport outage still propagates --
        # that contract is deliberate -- but it may not also discard the rows
        # it was already billed for: the first five-lane live run lost forty
        # minutes of clean output to a timeout on a later job, and the resume
        # contract that was supposed to make that cheap never got the chance,
        # because nothing had reached the disk to resume from.
        for kind in sorted(rows):
            write.append_unique(write.path_for(shard, kind, out_dir), rows[kind])
        for kind in sorted(deads):
            write.append_unique(
                write.path_for("dead_letter", kind, out_dir), deads[kind]
            )
    return report
