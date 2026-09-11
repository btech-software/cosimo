#!/usr/bin/env python3
"""The §B bake-off: is think-on worth its cost on a prose lane?

The amendment sets the bar and this runs the measurement::

    Amendment test: 20 analysis rows think-off vs think-on. Promote think-on
    only if invented-number rate drops by >= 2 points. Otherwise it stays off.

Both arms ask the *same* teacher the *same* briefs over the *same* packs, so
the only difference on the wire is the ``thinking`` block and the completion
budget that goes with it (800 think-off, 2048 think-on -- a think-on arm at the
think-off budget would be measuring truncation, not reasoning).

Every row is graded through ``verification.prose.gate_violations``, which is
the gate the renderer would have applied, so "invented-number rate" here means
exactly what it means in the corpus rather than something reimplemented for a
report. The other axes are reported alongside because a change that halves
invented numbers while doubling dead letters is not an improvement, and a
one-number bake-off cannot see that.

Live by construction: there is nothing to learn from a fixture that answers
identically whatever the flag says. It refuses without ``COSIMO_V3_LIVE=1``.

    uv run --group corpus python dataset/tools/think_ablation.py --rows 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(_HERE)
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config, inventory  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.prose import select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import routing  # noqa: E402
from pipelines.v3.teacher.client import TeacherError, teacher_from_env  # noqa: E402
from pipelines.v3.teacher.prompts import render_brief  # noqa: E402
from pipelines.v3.verification.prose import (  # noqa: E402
    canonical_numbers,
    gate_violations,
    whitelist_for,
)
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402

#: Where the verdict is written. `dataset_build.sh full` reads this file and
#: refuses while it still says NOT YET MEASURED, which is the §F precondition
#: "think-off analysis has not been measured" made checkable.
REPORT_PATH = os.path.join(_DATASET, "progress", "v3_think_ablation.md")

#: The amendment's promotion bar, in points of invented-number rate.
PROMOTION_BAR_POINTS = 2.0


def _packs(kind: str, rows: int) -> list[dict]:
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    out: list[dict] = []
    for job in select_prose_jobs(jobs, types=(kind,), limit=None):
        try:
            out.append(compute_pack(job.work_type, job.family, job.variant).to_dict())
        except PackError:
            continue  # the guard's verdict; the renderer would never ask either
        if len(out) == rows:
            break
    return out


def _arm(teacher, packs: list[dict], kind: str, *, think: bool) -> dict:
    """One arm of the bake-off: same briefs, one flag different."""
    route = routing.route(kind)
    budget = routing.budget_for(think)
    results: list[dict] = []
    started = time.time()
    for pack in packs:
        try:
            reply = teacher.complete(
                render_brief(pack, kind=kind),
                model=route.model,
                temperature=config.PROSE_TEMPERATURES[0],
                max_tokens=budget,
                think=think,
            )
        except TeacherError as exc:
            results.append({"error": str(exc)})
            continue
        text = (reply.text or "").strip()
        offenders = invented_numbers(text, canonical_numbers(pack), whitelist_for(pack))
        results.append(
            {
                "scenario_id": pack["scenario_id"],
                "variant": pack["variant"],
                "words": len(text.split()),
                "empty": not text,
                "finish_reason": reply.finish_reason,
                "think_present": bool(reply.think),
                "invented": offenders,
                "violations": gate_violations(pack, reply.text, kind),
                "usage": dict(reply.usage or {}),
            }
        )
    graded = [r for r in results if "error" not in r]
    n = len(graded) or 1
    return {
        "think": think,
        "max_tokens": budget,
        "model": route.model,
        "n_asked": len(packs),
        "n_answered": len(graded),
        "errors": [r["error"] for r in results if "error" in r],
        "wall_seconds": round(time.time() - started, 1),
        # The headline. A *rate over rows*, not over tokens: the amendment's
        # bar is "two points", and a point is a row.
        "invented_number_rate": sum(1 for r in graded if r["invented"]) / n,
        "clean_rate": sum(1 for r in graded if not r["violations"]) / n,
        "empty_rate": sum(1 for r in graded if r["empty"]) / n,
        "median_words": statistics.median([r["words"] for r in graded] or [0]),
        "mean_completion_tokens": (
            sum(float(r["usage"].get("completion_tokens") or 0) for r in graded) / n
        ),
        "rows": graded,
    }


#: Above this empty-draft rate an arm has not measured anything. A rate is a
#: ratio over rows that produced text; an arm where most rows produced none is
#: reporting the teacher's budget, not its judgement.
DEGENERATE_EMPTY_RATE = 0.5


def degenerate(arm: dict) -> str | None:
    """Why this arm measured nothing, or ``None`` if it measured something.

    Guarding this is the whole difference between a bake-off and a number. The
    first live run of this tool returned ``invented_number_rate 0.000`` on both
    arms and a tidy "KEEP think-off" verdict -- from forty calls that every one
    of them returned *empty*, ``finish_reason: length``, the budget consumed to
    the token. Zero invented numbers because zero numbers. Had that been
    written as the verdict it would have satisfied §F's "think-off analysis has
    been measured" precondition and unblocked a full render on the strength of
    a measurement of nothing.
    """
    if not arm["n_answered"]:
        return "no row was answered at all"
    if arm["empty_rate"] > DEGENERATE_EMPTY_RATE:
        return (
            f"{arm['empty_rate']:.0%} of drafts came back empty at a "
            f"{arm['max_tokens']}-token budget -- the teacher spent the budget "
            "before it wrote a word, so this arm measured the cap, not the flag"
        )
    return None


def render_report(kind: str, off: dict, on: dict) -> str:
    spoiled = {
        name: why
        for name, why in (("think-off", degenerate(off)), ("think-on", degenerate(on)))
        if why
    }
    delta = (off["invented_number_rate"] - on["invented_number_rate"]) * 100.0
    promote = not spoiled and delta >= PROMOTION_BAR_POINTS
    verdict = (
        f"PROMOTE think-on for `{kind}`" if promote else f"KEEP think-off for `{kind}`"
    )
    lines = [
        "# v3 think ablation (amendment §B)",
        "",
        "The amendment's test, run: *20 analysis rows think-off vs think-on;",
        "promote think-on only if the invented-number rate drops by >= 2 points.*",
        "",
        f"- record type: `{kind}`",
        f"- teacher: `{off['model']}`",
        f"- rows asked per arm: {off['n_asked']}",
        f"- budgets: {off['max_tokens']} think-off / {on['max_tokens']} think-on",
        "",
        "| metric | think-off | think-on |",
        "| --- | ---: | ---: |",
    ]
    for label, key, fmt in (
        ("rows answered", "n_answered", "{:.0f}"),
        ("invented-number rate", "invented_number_rate", "{:.3f}"),
        ("gate-clean rate", "clean_rate", "{:.3f}"),
        ("empty-draft rate", "empty_rate", "{:.3f}"),
        ("median words", "median_words", "{:.0f}"),
        ("mean completion tokens", "mean_completion_tokens", "{:.0f}"),
        ("wall seconds", "wall_seconds", "{:.1f}"),
    ):
        lines.append(f"| {label} | {fmt.format(off[key])} | {fmt.format(on[key])} |")
    if spoiled:
        lines += [
            "",
            "## NOT YET MEASURED",
            "",
            "This run produced no usable comparison, so it carries no verdict and",
            "does not satisfy §F's precondition. `dataset_build.sh full` stays",
            "blocked, which is the point: a bake-off that measured nothing must",
            "not read like one that measured a tie.",
            "",
        ]
        for name, why in spoiled.items():
            lines.append(f"- **{name} arm**: {why}")
        lines += [
            "",
            "### What this usually means",
            "",
            "A teacher that reasons *unconditionally*. §B's budgets assume the",
            "think flag decides whether a chain of thought is produced -- 800",
            "tokens being twice the widest prose band once nothing reasons first.",
            "A model that reasons whatever the flag says spends that budget before",
            "it reaches an answer and returns `content: null` with",
            "`finish_reason: length`, identically in both arms.",
            "",
            "Check `think_present` in the per-row detail (`--json`). If it is true",
            "on the think-off arm, the flag is not reaching the model and the",
            "budgets need the teacher's reasoning overhead added to them:",
            "",
            "    COSIMO_V3_THINK_OVERHEAD=8000 COSIMO_V3_LIVE=1 \\",
            "      uv run --group corpus python dataset/tools/think_ablation.py",
            "",
            "Raising it is a statement about the *endpoint*, not about the corpus:",
            "the amendment's 800/2048 are what an answer costs, and the overhead is",
            "what this particular teacher spends before writing one.",
            "",
        ]
    lines += [
        "",
        f"**Δ invented-number rate: {delta:+.1f} points** "
        f"(bar: ≥ {PROMOTION_BAR_POINTS:.0f} points to promote)"
        + ("  — meaningless while an arm is degenerate, see above" if spoiled else ""),
        "",
    ]
    if not spoiled:
        lines += [f"## Verdict: {verdict}", ""]
    if spoiled:
        pass
    elif not promote:
        lines += [
            "Think stays off for the prose lanes. `routing.py` already reflects",
            "this; `COSIMO_V3_MEMO_THINK=1` remains the one operator override,",
            "and it is for re-running this measurement on the memo lane, not for",
            "turning reasoning back on because a run looked thin.",
            "",
        ]
    else:
        lines += [
            "Think-on clears the bar for this lane. Promoting it is a `routing.py`",
            "edit plus a re-run of every committed fixture (the think flag is part",
            "of the request body, so it rekeys the replay tables).",
            "",
        ]
    if off["errors"] or on["errors"]:
        lines += ["## Transport errors", ""]
        for arm, name in ((off, "think-off"), (on, "think-on")):
            for err in arm["errors"][:5]:
                lines.append(f"- {name}: {err}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--kind", default="analysis")
    parser.add_argument("--out", default=REPORT_PATH)
    parser.add_argument("--json", help="also write the per-row detail here")
    args = parser.parse_args(argv)

    if not config.live_enabled():
        print(
            "think_ablation needs a live teacher: a fixture answers identically "
            f"whatever the flag says. Set {config.LIVE_ENV}=1.",
            file=sys.stderr,
        )
        return 2

    packs = _packs(args.kind, args.rows)
    if len(packs) < args.rows:
        print(
            f"only {len(packs)} renderable {args.kind} packs for a {args.rows}-row "
            "arm; the bar is a rate, and a rate on a short sample is noise",
            file=sys.stderr,
        )
        return 1

    teacher = teacher_from_env(live=True)
    print(f"think-off arm: {len(packs)} {args.kind} rows ...")
    off = _arm(teacher, packs, args.kind, think=False)
    print(f"think-on  arm: {len(packs)} {args.kind} rows ...")
    on = _arm(teacher, packs, args.kind, think=True)

    report = render_report(args.kind, off, on)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        handle.write(report)
    print(report)
    print(f"wrote {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump({"off": off, "on": on}, handle, indent=1, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
