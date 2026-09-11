"""The agentic renderer: teacher proposes, the oracle disposes (spec §5.4, §5.8).

The loop, exactly as §5.4 draws it -- ``teacher proposes assistant turn ->
validate tool_calls against the registry -> oracle executes with the
injected fault -> append tool role`` -- repeated until the teacher writes a
plain turn, then handed to :mod:`verification.agentic` for judgement. Three
policies worth stating, because they are the whole difference between a
corpus and a transcript dump:

* **Nothing executes that the registry has not seen.** A call naming an
  unregistered tool, argument bytes that will not parse, arguments the
  registered schema rejects -- any of those is a single strike and a dead
  letter, *pre*-execution. Mid-trajectory there is no repair to do: an
  executed-then-retracted call leaves an assistant turn lying about a server
  that never spoke, and the grammar of a shipped trajectory is not
  re-negotiable after the fact. The repair ladder exists for the one thing
  it can actually fix -- the final turn.
* **The loop pays the oracle no attention it has not earned.** The call
  budget (``config.AGENTIC_MAX_TOOL_CALLS``) and the proposal budget are
  enforced here, before execution: an argument-mismatch echo can legitimately
  send a live teacher around the same block twice, and a loop without a
  ceiling is the difference between a corpus and a meter running.
* **Faults are steered, not hidden.** The gate's
  ``expected_tool_contents`` is the single authority on which call was
  served the dirty block; the loop asks it for the element each new call
  earns, so loop and gate agree *by construction* -- one function, not one
  rule transcribed twice. That is what makes ``verify_v3``'s replay a
  byte-audit of history rather than a re-imagination of it.

Shipped rows carry the clean conversation (goal, calls, results, answer)
*without* the factory's system turn; repair rounds -- the desk's QA chatter --
stay out of training bytes and in the dead letter's audit trail, where
post-mortems need them. Holdout families render here now, exactly as in
:mod:`render.prose`, into ``eval/`` rather than ``sft/``: a gold family with no
rendered rows gave the generalisation measurement nothing to measure, which
was the more expensive of the two failures.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # dataset/
for _p in (_DATASET, os.path.dirname(_DATASET)):  # dataset; repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cosimo.tools import wire  # noqa: E402

from .. import config, write  # noqa: E402
from .. import row as rowlib  # noqa: E402
from ..oracle import faults, runtime  # noqa: E402
from ..teacher import routing  # noqa: E402
from ..teacher.prompts import render_agentic_brief, render_agentic_repair  # noqa: E402
from ..verification.agentic import expected_tool_contents, gate_violations  # noqa: E402
from .prose import _PACK_ENVELOPE, _family_of, row_id, row_id_from_coords  # noqa: E402

#: The one record type this renderer owns. Deliberately not in
#: ``prompts.BRIEF_KINDS``: prose asks for a voice, this asks for a
#: transcript -- different contract, different gate, different module.
KIND = "agentic"

#: The full registry in registry order: what the server would offer. The
#: teacher is shown every registered tool, not merely the ones this work
#: type's pack can answer with -- calling sideways into an honest empty
#: block is a lesson the corpus *wants* to teach, and §5.8's "a tool not in
#: the serving app does not appear in the corpus" is a rule about the
#: registry, not a per-render permission slip. Built once: the same object
#: rides every request body, so the replay keys of a fixture are stable.
_ADVERTISED = [runtime.SCHEMAS[name] for name in runtime.SCHEMA_NAMES]


def select_agentic_jobs(jobs, *, limit=None, holdout=False) -> list:
    """The render-eligible agentic slice of *jobs*: one cohort, ordered.

    Same discipline as ``prose.select_prose_jobs`` and shared with the fixture
    harness on purpose -- the replay keys are request hashes, so an
    off-by-one in this filter is a fixture miss, i.e. exactly the loud
    failure a shared selector prevents.

    ``holdout`` picks the cohort. The *rank* it yields is still the ordinal
    within that cohort, which is what the fault schedule keys off -- so the
    train slice's mix is untouched by holdout families now rendering, and the
    eval slice gets a mix of its own rather than inheriting a stride computed
    over rows it does not contain.
    """
    selected = sorted(
        (
            job
            for job in jobs
            if job.record_type == KIND and bool(job.holdout) is bool(holdout)
        ),
        key=lambda job: (job.work_type, job.family, job.record_type, job.variant),
    )
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def select_agentic_sample(
    jobs, *, limit: int, faulted: int, no_call: int
) -> list[tuple[int, object]]:
    """A *limit*-sized slice of the agentic slice with its mix pinned exact.

    The head window of ``select_agentic_jobs`` already runs 4 faulted / 4
    no-call per twenty ranks -- that is what rank arithmetic buys -- but a
    coordinate the pack guard refused is *examined* without ever shipping a
    row, and a head ``--limit`` window would let those refusals chew the
    mix. This walks the same shared selector's order, keeps a rank only if
    its mode still has budget and its coordinates are renderable, and stops
    at *limit*: the PR3 gate's "20 conversations, 4 with faults, 4 no-call"
    is then a count, not a prayer. Under-supplied quotas raise loudly --
    a plan whose agentic slice cannot field them is a plan to fix, not a
    sample to pad.

    The *(rank, job)* pairs come back together because the schedule is a
    function of the selector ordinal, and a sample that dropped the ordinal
    would drop the dirty-program with it: a faulted row rendered from the
    wrong rank replays against the wrong schedule and says so, loudly, at
    the gate -- which is the failure mode this pairing exists to make
    impossible at the source.
    """
    if faulted + no_call > limit:
        raise ValueError(
            f"sample of {limit} cannot hold {faulted} faulted and {no_call} no-call"
        )
    wanted = {
        "faulted": faulted,
        "no_call": no_call,
        "clean": limit - faulted - no_call,
    }
    taken = {"faulted": 0, "no_call": 0, "clean": 0}
    sample: list = []
    for rank, job in enumerate(select_agentic_jobs(jobs)):
        schedule = faults.schedule_of(rank)
        if taken[schedule.mode] >= wanted[schedule.mode]:
            continue
        try:
            from ..packs import PackError, compute_pack  # noqa: PLC041

            compute_pack(job.work_type, job.family, job.variant)
        except PackError:
            continue  # the guard's own verdict: the coordinate never existed
        sample.append((rank, job))
        taken[schedule.mode] += 1
        if len(sample) == limit:
            break
    if len(sample) < limit:
        raise ValueError(
            f"only {len(sample)} renderable agentic jobs for a sample of {limit} "
            f"(wanted {wanted}, got {taken}); raise the plan's variant counts -- "
            "the PR3 gate's mix is not negotiable by shrinking the sample"
        )
    return sample


def _shipped(messages: list[dict]) -> list[dict]:
    """The trajectory as the student row carries it: no factory system turn."""
    return [m for m in messages if m.get("role") != "system"]


def _parse_proposal(raw_calls) -> tuple[list[dict], list[str]]:
    """``(valid calls, problems)`` from the wire's ``tool_calls`` channel.

    Every problem is collected, none raises: the dead letter must show all of
    a teacher's missteps at once, the way the gate names all violations at
    once -- and *nothing here executes*, so a bad call cannot leak into the
    oracle even by accident.
    """
    calls: list[dict] = []
    problems: list[str] = []
    for raw in raw_calls or ():
        function = (raw or {}).get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                problems.append(f"call {name!r} carries arguments that are not json")
                continue
        if not isinstance(arguments, dict):
            problems.append(f"call {name!r} carries arguments that are not an object")
            continue
        if name not in runtime.SCHEMA_NAMES:
            problems.append(f"tool {name!r} is not in the registry")
            continue
        errors = runtime.validate_call(name, arguments)
        if errors:
            problems.append(
                f"call to {name!r} violates its schema: " + "; ".join(errors)
            )
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls, problems


def render_agentic_row(
    teacher, pack_line: dict, *, rank: int, holdout: bool = False
) -> dict:
    """One stored pack line -> ``{"row": .., "dead_letter": ..}``.

    ``rank`` is the job's ordinal in :func:`select_agentic_jobs` -- the
    schedule's only input, and the reason it is an argument rather than
    something derived here: the fixture harness and the gate count ordinals
    through the same shared selector, so the dirty-program schedule cannot
    silently disagree with the renderer about who was faulted.

    Never raises for a *content* failure: a violation is data and travels to
    the dead letter whole (exchange, calls, results, the works).
    ``TeacherError`` propagates -- an endpoint misbehaving is an outage to
    run the job over, not a verdict on three thousand good scenarios.
    """
    pack = {k: v for k, v in pack_line.items() if k not in _PACK_ENVELOPE}
    stamp = pack_line.get("verification") or {}
    schedule = faults.schedule_of(rank)
    route = routing.route(KIND)

    attempts = 0
    temperatures: list[float] = []

    def ask(messages: list[dict], temperature: float):
        """One turn of the loop, with the amendment's §B think split applied.

        The planner turn -- the first ask, before any tool has answered -- is
        where the trajectory is decided and is worth a chain of thought. Every
        turn after a tool result is a *reaction* to bytes the oracle just
        handed back: reasoning there re-derives a plan that is already fixed,
        and the corpus pays for it on every hop of every conversation. The
        budget follows the flag, so a tool-call turn costs the think-off cap.
        """
        nonlocal attempts
        attempts += 1
        temperatures.append(temperature)
        seen_results = any(m.get("role") == "tool" for m in messages)
        think = routing.agentic_turn_think(route, tool_results_seen=seen_results)
        return teacher.complete(
            messages,
            model=route.model,
            temperature=temperature,
            max_tokens=routing.budget_for(think),
            think=think,
            extra={"tools": _ADVERTISED},
        )

    working = list(
        render_agentic_brief(
            pack,
            mode=schedule.mode,
            tools=_ADVERTISED,
            max_calls=config.AGENTIC_MAX_TOOL_CALLS,
            message_band=(config.AGENTIC_MIN_MESSAGES, config.AGENTIC_MAX_MESSAGES),
        )
    )
    executed: list[dict] = []

    def dead_letter(reason: str, violations: list[str], exchange: list[dict]) -> dict:
        return {
            "row": None,
            "log": None,
            "dead_letter": {
                "id": row_id(KIND, pack),
                "record_type": KIND,
                "reason": f"{reason}: " + " | ".join(violations),
                "work_type": pack["work_type"],
                "scenario_id": pack["scenario_id"],
                "attempts": attempts,
                "temperatures": list(temperatures),
                "mode": schedule.mode,
                "fault": schedule.fault,
                "tool_calls": len(executed),
                "violations": violations,
                "exchange": exchange,
            },
        }

    # -- phase 1: the loop ----------------------------------------------------
    plain = None
    if schedule.mode == "no_call":
        # One ask, no execution. And no ladder either: a no-call job that
        # reaches for a tool has breached the shape of its row, not the
        # wording of its answer -- the same single-strike doctrine as an
        # unregistered call, because no repair can un-call what was asked
        # (spec §5.8: the call itself is the defect, "rejected =
        # unnecessary call"). The gate's own words are the verdict's text.
        plain = ask(working, config.AGENTIC_TEMPERATURES[0])
        if plain.tool_calls:
            audit = [
                {
                    "role": "assistant",
                    "content": plain.text or "",
                    "tool_calls": list(plain.tool_calls),
                }
            ]
            verdict = gate_violations(
                pack,
                _shipped(working + audit),
                mode=schedule.mode,
                fault=schedule.fault,
            )
            return dead_letter("call", verdict, [*working, *audit])
    else:
        # Bounded by the call budget alone, and provably so: every proposal
        # that does not close the loop executes at least one call, so the
        # budget check below ends this while inside seven iterations -- no
        # separate proposal counter guarding a case the arithmetic already
        # excludes.
        while True:
            result = ask(working, config.AGENTIC_TEMPERATURES[0])
            if not result.tool_calls:
                plain = result
                break
            calls, problems = _parse_proposal(result.tool_calls)
            if problems:
                # Single strike, pre-execution -- the module docstring.
                working.append(
                    wire.assistant_tool_call_message(
                        [
                            {
                                "name": (raw or {}).get("function", {}).get("name"),
                                "arguments": (raw or {})
                                .get("function", {})
                                .get("arguments", {}),
                            }
                            for raw in result.tool_calls
                        ],
                        content=result.text or "",
                    )
                )
                return dead_letter("call", problems, working)
            if len(executed) + len(calls) > config.AGENTIC_MAX_TOOL_CALLS:
                return dead_letter(
                    "budget",
                    [
                        f"{len(executed) + len(calls)} calls would exceed the budget of "
                        f"{config.AGENTIC_MAX_TOOL_CALLS}"
                    ],
                    working,
                )
            working.append(
                wire.assistant_tool_call_message(calls, content=result.text or "")
            )
            for call in calls:
                executed.append(call)
                # The gate's replay table *is* the execution log: the element
                # this call earned comes from the same function the board will
                # later re-run against the recomputed pack.
                content = expected_tool_contents(
                    pack, executed, mode=schedule.mode, fault=schedule.fault
                )[len(executed) - 1]
                working.append(wire.tool_result_message(call["name"], content))

    # -- phase 2: judge the final turn; one ladder, cooler rungs only --------
    def judge(reply) -> list[str]:
        # The judged transcript carries the proposal's tool_calls verbatim:
        # in the calling mode they are absent by construction (the loop broke
        # on a plain turn), but in no_call mode the call itself is the
        # violation, and a gate blind to it would certify waste.
        message = {"role": "assistant", "content": reply.text or ""}
        if reply.tool_calls:
            message["tool_calls"] = list(reply.tool_calls)
        # Judged on the bytes that will *ship*, which since §A no longer
        # include the factory's system turn. One policy, one text: a renderer
        # that graded the brief-bearing transcript and then wrote a stripped
        # one would be certifying something other than what it published, and
        # the board -- reading only the published row -- would be the first to
        # find out.
        return gate_violations(
            pack,
            _shipped(working + [message]),
            mode=schedule.mode,
            fault=schedule.fault,
        )

    final_reply = plain
    draft = (final_reply.text if final_reply is not None else "") or ""
    violations = judge(final_reply)
    audit = list(working)
    if violations:
        for temperature in config.AGENTIC_TEMPERATURES[1 : config.AGENTIC_ATTEMPTS]:
            audit = render_agentic_repair(audit, draft, violations)
            repair = ask(audit, temperature)
            if repair.tool_calls:
                return dead_letter(
                    "repair",
                    [
                        "the repair turn proposed new calls; a repair may rewrite "
                        "only the final answer -- the grammar of what ran is fixed"
                    ],
                    audit,
                )
            final_reply = repair
            draft = repair.text or ""
            violations = judge(repair)
            if not violations:
                break
        if violations:
            return dead_letter(
                "gate", violations, [*audit, {"role": "assistant", "content": draft}]
            )

    # -- phase 3: ship the clean conversation; repair chatter stays in the audit
    used = sorted({call["name"] for call in executed})
    rid = row_id(KIND, pack)
    # The one record type that keeps its ``messages``, because for a trajectory
    # the transcript *is* the target -- but without the factory's system turn.
    # ``AGENTIC_SYSTEM`` is the teacher's brief (its posture line announces the
    # mode, its number policy quotes the pack), and `normalize_v3_record` was
    # already stripping `role: system` back off on the way into training. The
    # amendment moves that strip to where the row is written, so the corpus and
    # the harness cannot disagree about what shipped.
    shipped = _shipped(working)
    return {
        "row": rowlib.student_row(
            row_id=rid,
            kind=KIND,
            pack=pack,
            holdout=holdout,
            answer=rowlib.visible_answer(draft),
            stamp=stamp,
            teacher=final_reply.to_verification(),
            render={
                "kind": KIND,
                "attempts": attempts,
                "mode": schedule.mode,
                "fault": schedule.fault,
                "tool_calls": len(executed),
                "temperatures": list(temperatures),
            },
            attempts=attempts,
            extra={
                "messages": [
                    *shipped,
                    {"role": "assistant", "content": rowlib.visible_answer(draft)},
                ],
                "tool_names": used,
                "tool_schemas": [
                    runtime.SCHEMAS[name]
                    for name in runtime.SCHEMA_NAMES
                    if name in used
                ],
            },
        ),
        "dead_letter": None,
        "log": rowlib.teacher_log(
            row_id=rid,
            kind=KIND,
            pack=pack,
            messages=[*working, {"role": "assistant", "content": draft}],
            result=final_reply,
        ),
    }


def run_agentic_stage(
    out_dir: str, jobs, teacher, *, limit=None, holdout=False
) -> dict:
    """Render agentic rows for *jobs*; write ``sft/`` + ``dead_letter/``; report.

    The prose stage's shape on purpose -- ids gate whatever is ever asked of
    the teacher, the two absences keep their separate verdicts (no pack
    *file* means the packs stage never covered this plan, fatal; a variant
    the pack guard refused is that guard's own data, recorded as a skip), and
    ``append_unique`` keeps a replay a replay. The ``no_call``/``faulted``/
    ``tool_calls`` tallies are the mix the PR3 gate reads -- 20 conversations,
    4 faulted, 4 no-call -- measured from what shipped, not sampled from
    hope, because the schedule is a function of the plan.
    """
    selected = select_agentic_jobs(jobs, limit=limit, holdout=holdout)
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
        "no_call": 0,
        "faulted": 0,
        "tool_calls": 0,
        "bucket": shard,
        "teacher_logs": 0,
    }
    rows: list[dict] = []
    deads: list[dict] = []
    seen_ids: set[str] = set()
    known = write.existing_ids(
        write.path_for(shard, KIND, out_dir)
    ) | write.existing_ids(write.path_for("dead_letter", KIND, out_dir))
    for rank, job in enumerate(selected):
        rid = row_id_from_coords(KIND, job.work_type, job.family, job.variant)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        if rid in known:
            report["existing"] += 1
            report["by_kind"].setdefault(KIND, {"rendered": 0, "existing": 0})[
                "existing"
            ] += 1
            continue
        pack_line, file_existed = pack_for(job)
        if pack_line is None:
            entry = {
                "id": rid,
                "work_type": job.work_type,
                "family": job.family,
                "record_type": KIND,
                "variant": job.variant,
                "reason": (
                    "the pack guard rejected this parameter set (PackError at the "
                    "packs stage); the fact never existed, nothing to render"
                    if file_existed
                    else "no fact-pack file for this work type at all; run the packs "
                    "stage for this plan before rendering"
                ),
            }
            (
                report["skipped_by_pack_gate"]
                if file_existed
                else report["missing_packs"]
            ).append(entry)
            continue
        outcome = render_agentic_row(teacher, pack_line, rank=rank, holdout=job.holdout)
        cell = report["by_kind"].setdefault(KIND, {"rendered": 0, "existing": 0})
        if outcome["row"] is not None:
            rows.append(outcome["row"])
            report["rendered"] += 1
            cell["rendered"] += 1
            render_field = outcome["row"]["verification"]["render"]
            report["tool_calls"] += render_field["tool_calls"]
            if render_field["mode"] == "no_call":
                report["no_call"] += 1
            elif render_field["mode"] == "faulted":
                report["faulted"] += 1
            if keep_logs and outcome.get("log"):
                write.write_teacher_log(
                    write.teacher_log_path(KIND, rid, out_dir), outcome["log"]
                )
                report["teacher_logs"] += 1
        else:
            deads.append(outcome["dead_letter"])
            report["dead_lettered"] += 1
    if rows:
        write.append_unique(write.path_for(shard, KIND, out_dir), rows)
    if deads:
        write.append_unique(write.path_for("dead_letter", KIND, out_dir), deads)
    return report
