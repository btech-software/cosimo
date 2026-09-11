#!/usr/bin/env python3
"""How much room does this teacher actually need? Ask it, before rendering.

The expensive way to learn a teacher's budget is to render a slice and read the
dead letters. Nine live rows cost forty-eight minutes and produced nothing,
because a flat 4,000-token cap was sized from a single lucky probe while the
model's real appetite ranged 4,084-13,167 across rows. One sample of a quantity
that varies 3x is not a measurement.

This is the cheap way. It asks the *same brief* k times at a generous ceiling,
records how much each reply actually spent, and reports the distribution plus
the setting that covers it. Generous on purpose: a call that truncates teaches
you only that the ceiling was low, so the probe pays for headroom once rather
than paying for truncation k times.

What it prints is meant to be acted on:

* ``think_present`` on a think=False call -- whether the flag reaches the model
  at all. If true, §B's caps do not apply to this endpoint and the rest of the
  report says by how much;
* the completion-token spread, which is the thing a single sample hides;
* a recommended ``COSIMO_V3_THINK_OVERHEAD``, sized to the observed maximum
  rather than the median, because the median ships half the corpus.

    COSIMO_V3_LIVE=1 uv run --group corpus python dataset/tools/probe_teacher.py
"""

from __future__ import annotations

import argparse
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

#: The ceiling the probe asks at. Not a recommendation -- a measuring stick.
#: It has to sit above anything the teacher might want, or the probe measures
#: its own ceiling instead of the teacher's appetite.
PROBE_CEILING = 20000

#: Safety margin over the observed maximum. The spread between rows is the
#: whole problem, and k samples of a long-tailed quantity will under-report the
#: tail; 1.5x of the worst sample is cheap insurance against the row that was
#: not sampled.
HEADROOM = 1.5


def probe(kind: str, rows: int, ceiling: int) -> dict:
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    packs: list[dict] = []
    for job in select_prose_jobs(jobs, types=(kind,), limit=None):
        try:
            packs.append(compute_pack(job.work_type, job.family, job.variant).to_dict())
        except PackError:
            continue
        if len(packs) == rows:
            break

    route = routing.route(kind)
    teacher = teacher_from_env(live=True)
    samples: list[dict] = []
    print(
        f"probing {route.model} on {len(packs)} {kind} briefs at a "
        f"{ceiling}-token ceiling (think={route.think})\n",
        flush=True,
    )
    print(
        f"{'#':>2} {'wall':>6} {'finish':>9} {'completion':>11} {'words':>6} think",
        flush=True,
    )
    for index, pack in enumerate(packs, 1):
        started = time.time()
        try:
            reply = teacher.complete(
                render_brief(pack, kind=kind),
                model=route.model,
                temperature=config.PROSE_TEMPERATURES[0],
                max_tokens=ceiling,
                think=route.think,
            )
        except TeacherError as exc:
            print(f"{index:>2}  transport error: {exc}", flush=True)
            continue
        usage = reply.usage or {}
        spent = int(usage.get("completion_tokens") or 0)
        words = len((reply.text or "").split())
        samples.append(
            {
                "completion_tokens": spent,
                "words": words,
                "finish_reason": reply.finish_reason,
                "think_present": bool(reply.think),
                "wall": round(time.time() - started, 1),
            }
        )
        print(
            f"{index:>2} {samples[-1]['wall']:>5}s {reply.finish_reason:>9} "
            f"{spent:>11} {words:>6} {bool(reply.think)}",
            flush=True,
        )
    return {"kind": kind, "route": route, "ceiling": ceiling, "samples": samples}


def report(result: dict) -> str:
    samples = result["samples"]
    route = result["route"]
    if not samples:
        return "no samples: the teacher answered nothing, so nothing was measured."
    spent = sorted(s["completion_tokens"] for s in samples)
    answered = [s for s in samples if s["words"]]
    thinking = [s for s in samples if s["think_present"]]
    truncated = [s for s in samples if s["finish_reason"] == "length"]

    lines = [
        "",
        "=" * 66,
        f"teacher      : {route.model}   lane {route.lane}   think={route.think}",
        f"answered     : {len(answered)}/{len(samples)}",
        f"truncated    : {len(truncated)}/{len(samples)} at the {result['ceiling']} ceiling",
        f"completion   : min {spent[0]}  median {statistics.median(spent):.0f}  max {spent[-1]}"
        + (f"  (spread {spent[-1] / max(spent[0], 1):.1f}x)" if spent[0] else ""),
    ]

    honours = not thinking
    lines += [
        "",
        f"think flag honoured: {honours}"
        + ("" if honours else f"  ({len(thinking)}/{len(samples)} reasoned anyway)"),
    ]
    if honours:
        lines += [
            "",
            "This teacher does what §B assumes: the flag decides whether a chain",
            f"of thought exists. The lane cap ({route.max_tokens}) is the right",
            "budget and no overhead is needed.",
        ]
    else:
        need = int(spent[-1] * HEADROOM)
        overhead = max(0, need - route.max_tokens)
        lines += [
            "",
            "This teacher reasons whatever the flag says, so §B's caps do not",
            "apply to it -- they were sized for a lane with no chain of thought",
            "to run out of. Budget it explicitly:",
            "",
            f"    export COSIMO_V3_THINK_OVERHEAD={overhead}",
            "",
            f"({spent[-1]} observed worst case x {HEADROOM} headroom = {need}, "
            f"less the {route.max_tokens} lane cap.)",
            "",
            "Headroom over the *maximum*, not the median: the spread between",
            "rows is the failure mode, and a budget at the median truncates",
            "half the corpus. The renderer will also grow a row's budget on",
            "truncation and carry the lesson to later rows, so this only has to",
            "be approximately right.",
        ]
    if truncated:
        lines += [
            "",
            f"WARNING: {len(truncated)} sample(s) hit the {result['ceiling']} ceiling,",
            "so the maximum above is a floor, not the real appetite. Re-probe",
            "with --ceiling raised before trusting the recommendation.",
        ]
    lines.append("=" * 66)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", default="analysis")
    parser.add_argument(
        "--rows", type=int, default=4, help="samples; the point is the spread"
    )
    parser.add_argument("--ceiling", type=int, default=PROBE_CEILING)
    args = parser.parse_args(argv)

    if not config.live_enabled():
        print(
            f"probe_teacher needs a live teacher; set {config.LIVE_ENV}=1.",
            file=sys.stderr,
        )
        return 2
    print(report(probe(args.kind, args.rows, args.ceiling)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
