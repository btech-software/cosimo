"""The preference stage: second tellings and named wrong models (spec §5.6-7).

Pairs are a *second pass over shipped rows*, never a second output of the
render loop: §5.7's order of operations is "generate pairs **after** the SFT
target exists", and the stage keeps it literally -- it reads ``sft/``, draws
which rows are paired from the pair's own seed (``verification.preference.
pair_draw``, recomputible without the teacher), and asks the teacher for
exactly two things a row cannot give itself: a paraphrase (the chosen side,
which must clear the whole prose gate *and* not be the target again) and a
fluent crime (the rejected side, which must commit the pitfall its name
names). Both travels live in the pair's shard; the SFT row it was born from
is never touched -- v1 nested the pair on the supervised row's own id, and
DPO read a duplicate where it should have read a margin.

The two asks carry two ladders. The chosen side repairs like prose (the same
gate, the same "name the error and cool it" loop) because it is prose. The
rejected side repairs against *evidence* -- the detector said it did not
commit the crime -- and its routing is the table's own ``rejected=True``
half: same model as the parent, think forced off. A teacher paid to produce
a pair whose sides differ only by a number is a teacher paid to produce the
v1 loss curve; the copy gate stops that bill before it is written.
"""

from __future__ import annotations

import os

from . import config, inventory, write
from .packs import PackError, compute_pack
from .teacher import routing
from .teacher.client import DEFAULT_MAX_TOKENS
from .verification.preference import (
    PREFER_KIND,
    chosen_brief,
    pair_draw,
    pair_id,
    pitfall_violations,
    rejected_brief,
    shingle_overlap,
)


def _family_of(row: dict) -> str:
    return row["scenario_id"][len(row["work_type"]) + 1 :]


def _licensed(plan: dict, work_type: str) -> list[str]:
    """The work type's pitfall licence, sorted -- the draw's only alphabet.

    An empty licence raises: a work type shipped for pairing with nothing
    licensable is a plan defect, and drawing from an empty list is how a
    stage gets a crash log instead of a finding.
    """
    licensed = sorted((plan.get(work_type) or {}).get("pitfalls") or [])
    if not licensed:
        raise ValueError(
            f"work type {work_type!r} licenses no pitfall; the plan and the "
            "preference contract have drifted apart"
        )
    return licensed


def select_pair_jobs(
    out_dir: str, plan: dict, *, types=None, limit=None
) -> list[tuple[str, dict, str]]:
    """``(parent_kind, row, pitfall)`` for every shipped row the draw pairs.

    Shared with the fixture harness for the reason every selector in this
    pipeline is shared: the replay keys are hashes of what the stage will
    actually ask, and two copies of a draw could not stay byte-equal by luck.
    The iteration order (kind, then coordinates) is part of the contract.
    """
    wanted = set(config.PREF_PROBABILITIES) if types is None else set(types)
    unknown = wanted - set(config.PREF_PROBABILITIES)
    if unknown:
        raise ValueError(
            "prefer pairs over "
            + ", ".join(sorted(config.PREF_PROBABILITIES))
            + "; asked for "
            + ", ".join(sorted(unknown))
        )
    selected: list[tuple[str, dict, str]] = []
    for kind in sorted(wanted):
        path = write.path_for("sft", kind, out_dir)
        if not os.path.isfile(path):
            continue
        rows = write.read_jsonl(path)
        for row in sorted(
            rows,
            key=lambda row: (
                row.get("work_type", "?"),
                row.get("scenario_id", "?"),
                row.get("variant", -1),
            ),
        ):
            paired, pitfall = pair_draw(
                row["work_type"],
                _family_of(row),
                kind,
                row["variant"],
                _licensed(plan, row["work_type"]),
            )
            if paired:
                selected.append((kind, row, pitfall))
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _repair(violations: list[str], ask: str) -> str:
    return "Your passage drew these objections: " + "; ".join(violations) + ". " + ask


def build_pair_row(teacher, kind: str, row: dict, pitfall: str) -> dict:
    """One shipped row + its drawn pitfall -> ``{"row": .., "dead_letter": ..}``.

    Content failures are data (dead letter, exchange and all); a transport
    failure propagates as an outage to retry the run over. The pack is
    recomputed here, not read from the row: the pair is entitled to the
    computer's figures, not to a copy of the render's assumptions about them.
    """
    family = _family_of(row)
    rid = pair_id(row["work_type"], family, kind, row["variant"])

    def dead(reason: str, **extra) -> dict:
        return {
            "row": None,
            "dead_letter": {
                "id": rid,
                "record_type": PREFER_KIND,
                "parent_kind": kind,
                "pitfall": pitfall,
                "reason": reason,
                "work_type": row["work_type"],
                "scenario_id": row["scenario_id"],
                **extra,
            },
        }

    try:
        pack = compute_pack(row["work_type"], family, row["variant"]).to_dict()
    except PackError as exc:
        return dead(f"pack: {exc}")

    messages = row.get("messages") or []
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        return dead(
            "parent: the SFT row is not the prose triad; a pair is not "
            "a repair kit for a malformed row"
        )
    prompt = [{"role": m["role"], "content": m["content"]} for m in messages[:2]]
    answer = str(messages[2]["content"])

    route_parent = routing.route(kind)
    route_rejected = routing.route(kind, rejected=True)
    attempts = {"chosen": 0, "rejected": 0}
    temperatures = {"chosen": [], "rejected": []}

    # -- the chosen side: the full prose gate, plus not being the target ----
    exchange = chosen_brief(prompt, answer)
    problems: list[str] = []
    chosen = ""
    chosen_stamp: dict = {}
    while attempts["chosen"] < config.PREF_ATTEMPTS:
        temperature = config.PREF_TEMPERATURES[attempts["chosen"]]
        result = teacher.complete(
            exchange,
            model=route_parent.model,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            think=route_parent.think,
        )
        attempts["chosen"] += 1
        temperatures["chosen"].append(temperature)
        chosen = (result.text or "").strip()
        problems = [
            f"chosen side: {p}" for p in gate_problems(pack, kind, chosen, answer)
        ]
        if not problems:
            chosen_stamp = result.to_verification()
            break
        exchange = [
            *exchange,
            {"role": "assistant", "content": chosen},
            {
                "role": "user",
                "content": _repair(
                    problems,
                    "Restate the answer once more in your own words, under the "
                    "same figures -- and change the telling, do not copy it.",
                ),
            },
        ]
    else:
        return dead(
            "gate: " + " | ".join(problems),
            attempts=dict(attempts),
            temperatures=dict(temperatures),
            exchange=exchange,
        )

    # -- the rejected side: fluent prose of one named crime -----------------
    exchange = rejected_brief(prompt, row["work_type"], pitfall)
    rejected_problems: list[str] = []
    rejected = ""
    rejected_stamp: dict = {}
    while attempts["rejected"] < config.PREF_ATTEMPTS:
        temperature = config.PREF_TEMPERATURES[attempts["rejected"]]
        result = teacher.complete(
            exchange,
            model=route_rejected.model,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            think=route_rejected.think,
        )
        attempts["rejected"] += 1
        temperatures["rejected"].append(temperature)
        rejected = (result.text or "").strip()
        rejected_problems = rejected_problems_of(pack, rejected, pitfall, chosen)
        if not rejected_problems:
            rejected_stamp = result.to_verification()
            break
        exchange = [
            *exchange,
            {"role": "assistant", "content": rejected},
            {
                "role": "user",
                "content": _repair(
                    rejected_problems,
                    f"Rewrite under the same defect ({pitfall}); the defect is "
                    "the lesson, so commit it visibly and keep the pack's "
                    "figures elsewhere.",
                ),
            },
        ]
    else:
        return dead(
            "gate: " + " | ".join(rejected_problems),
            attempts=dict(attempts),
            temperatures=dict(temperatures),
            exchange=exchange,
        )

    stamp = row.get("verification") or {}
    return {
        "row": {
            "id": rid,
            "record_type": PREFER_KIND,
            "source_sft_id": row["id"],
            "work_type": row["work_type"],
            "scenario_id": row["scenario_id"],
            "variant": row["variant"],
            "parent_kind": kind,
            "pitfall": pitfall,
            "question": pack["question"],
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "register": pack["register"],
            "verification": {
                "computed_by": stamp.get("computed_by", "unknown"),
                "pack_seed": stamp.get("pack_seed", f"{pack['seed']:016x}"),
                "teacher": {"chosen": chosen_stamp, "rejected": rejected_stamp},
                "render": {
                    "kind": PREFER_KIND,
                    "parent_kind": kind,
                    "pitfall": pitfall,
                    "attempts": dict(attempts),
                    "temperatures": dict(temperatures),
                    "shingle_overlap": {
                        "target": round(shingle_overlap(chosen, answer), 3),
                        "pair": round(shingle_overlap(chosen, rejected), 3),
                    },
                },
            },
        },
        "dead_letter": None,
    }


def gate_problems(pack: dict, kind: str, chosen: str, answer: str) -> list[str]:
    """The chosen side's objections: the prose gate, plus the copy clauses.

    The gate is ``verification.prose.gate_violations`` whole -- the chosen
    side claims to be a telling of the pack, and it is graded as one; the
    copy clauses are acceptance criterion 5 read literally: byte-equal to
    the target is the v1 disease, and a near copy (shingle overlap above
    the set threshold) is the same disease wearing a hat.
    """
    from .verification.prose import gate_violations  # late: one home for the gate

    problems = list(gate_violations(pack, chosen, kind))
    if chosen == answer:
        problems.append("the restatement is the SFT target byte for byte")
    else:
        overlap = shingle_overlap(chosen, answer)
        if overlap > config.PREF_MAX_SHINGLE_OVERLAP:
            problems.append(
                f"near copy of the SFT target: shingle overlap "
                f"{overlap:.2f} exceeds {config.PREF_MAX_SHINGLE_OVERLAP}"
            )
    return problems


def rejected_problems_of(
    pack: dict, rejected: str, pitfall: str, chosen: str
) -> list[str]:
    """The rejected side's objections: fluency, the crime, the contrast.

    Not the prose gate: the rejected side is *meant* to break the pack's
    rules, and the one rule it may not break is the pair's own -- it must
    differ from the chosen side by more than a number, which is the shingle
    clause again, aimed this time at the margin DPO will train on.
    """
    problems: list[str] = []
    words = len(rejected.split())
    if not config.PREF_MIN_REJECTED_WORDS <= words <= config.PREF_MAX_REJECTED_WORDS:
        problems.append(
            f"not fluent prose of the desk: {words} words, outside the band "
            f"{config.PREF_MIN_REJECTED_WORDS}-{config.PREF_MAX_REJECTED_WORDS}"
        )
    problems += pitfall_violations(pack, rejected, pitfall)
    overlap = shingle_overlap(chosen, rejected)
    if overlap > config.PREF_MAX_SHINGLE_OVERLAP:
        problems.append(
            f"near copy of the chosen side: shingle overlap {overlap:.2f} "
            f"exceeds {config.PREF_MAX_SHINGLE_OVERLAP} -- a margin of one "
            "number is not a contrast (v1's loss-0.0 pair)"
        )
    return problems


def run_prefer_stage(out_dir: str, teacher, *, types=None, limit=None) -> dict:
    """Pair the shipped rows; write ``preference/`` + ``dead_letter/``; report.

    The id gate is the bill's keeper: a pair already on the shard (or
    dead-lettered) is not asked for again, and the draw is recomputible from
    the plan, so a replay run discovers the whole pair set, asks the teacher
    for none of it, and costs nothing.
    """
    plan = inventory.load_plan(config.taxonomy_path())
    selected = select_pair_jobs(out_dir, plan, types=types, limit=limit)
    report = {
        "jobs_seen": len(selected),
        "rendered": 0,
        "existing": 0,
        "dead_lettered": 0,
        "skipped_by_pack_gate": [],
        "by_kind": {},
    }
    known = write.existing_ids(write.path_for("preference", "pairs", out_dir)) | (
        write.existing_ids(write.path_for("dead_letter", PREFER_KIND, out_dir))
    )
    pairs: list[dict] = []
    deads: list[dict] = []
    for kind, row, pitfall in selected:
        rid = pair_id(row["work_type"], _family_of(row), kind, row["variant"])
        if rid in known:
            report["existing"] += 1
            report["by_kind"].setdefault(kind, {"rendered": 0, "existing": 0})[
                "existing"
            ] += 1
            continue
        outcome = build_pair_row(teacher, kind, row, pitfall)
        cell = report["by_kind"].setdefault(kind, {"rendered": 0, "existing": 0})
        if outcome["row"] is not None:
            pairs.append(outcome["row"])
            report["rendered"] += 1
            cell["rendered"] += 1
        elif str(outcome["dead_letter"]["reason"]).startswith("pack:"):
            # The guard's verdict, not a teacher failure: the row shipped
            # against a plan the computers no longer agree with. Recorded in
            # the report, never in the shard (where it would gate the id
            # forever for a fact that never existed), and it bills nobody --
            # the recompute precedes every ask.
            report["skipped_by_pack_gate"].append(outcome["dead_letter"])
        else:
            deads.append(outcome["dead_letter"])
            report["dead_lettered"] += 1
    if pairs:
        write.append_unique(write.path_for("preference", "pairs", out_dir), pairs)
    if deads:
        write.append_unique(write.path_for("dead_letter", PREFER_KIND, out_dir), deads)
    return report
