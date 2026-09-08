#!/usr/bin/env python3
"""Generate ``prose_fixture.json``: the offline dummy for the prose render stage.

The teacher_echo fixture (PR1) proves the *client* replays. This one proves
the *pipeline*: fifty real ``analysis`` jobs from the real plan, rendered
end to end through ``cli render`` with no network and no GPU, and audited by
``cli verify`` -- the arch spec §10 PR2 gate ("50 analysis rows,
invented-number rate 0 on the dummy") is a pytest over this file.

The answers are *composed, not captured*: question verbatim, one sentence per
``must_mention``, then fixed-point renderings of the pack's own computed
figures until the word budget is met. Every number it writes is a number the
pack itself authored, which is what makes the dummy clean under the same
:meth:`pipelines.v3.verification.prose.gate_violations` the live teacher
faces -- and the build below *asserts* that, on every entry, before writing
a byte. A dummy that flunks its own gate is a broken harness, and it must
fail the build, not the corpus.

Determinism: no wall clock, no RNG, no environment reads. The request bodies
pin the dummy's own model name (never the deployment's lane names), so the
bytes hash the same on every box; an offline run hits them by setting
``TEACHER_PROSE`` to :data:`DUMMY_MODEL`, exactly the convention
``make_teacher_echo.py`` established. Run from the repo root::

    .venv/bin/python dataset/tests/v3/fixtures/make_prose_fixture.py

The committed json must match byte-for-byte; test_prose_fixture enforces it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_TESTS = os.path.dirname(_HERE)
DATASET = os.path.dirname(os.path.dirname(_V3_TESTS))
for _p in (DATASET, os.path.dirname(DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config, inventory  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.prose import select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import routing  # noqa: E402
from pipelines.v3.teacher.client import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    build_body,
    canonical_request,
)
from pipelines.v3.teacher.prompts import WORD_BUDGETS, render_brief  # noqa: E402
from pipelines.v3.verification.prose import gate_violations  # noqa: E402

PROSE_FIXTURE_NAME = "prose_fixture.json"
DEFAULT_TYPES = ("analysis",)
DEFAULT_LIMIT = 50

#: The offline model name, pinned into every request body this file keys on.
#: Never a deployment lane name: the bytes of the replay table must not depend
#: on what a given box has its ``TEACHER_*`` env pointed at.
DUMMY_MODEL = "fixture-prose"


def _dec(value: float) -> str:
    """Fixed-point rendering, no sci notation, no thousands separators.

    ``%.10f`` then trimmed: every pack value the harness quotes is a number
    the pack's own ``assemble_numbers`` walked into ``allowed_numbers``, so
    the token only has to *round-trip inside the gate's tolerance*, and plain
    decimal notation is the spelling whose tokens the number gate parses
    without ambiguity.
    """
    text = f"{float(value):.10f}".rstrip("0")
    if text.endswith("."):
        text = text[:-1]
    return text or "0"


def compliant_text(pack: dict, kind: str) -> str:
    """A completion that satisfies the contract for *pack*, assembled locally.

    Mirrors what a good teacher answer looks like -- the question's answer is
    the question's facts, restated -- minus any freedom to be wrong: nothing
    here is outside the pack's fields, and every figure is the pack's own.
    """
    low, high = WORD_BUDGETS[kind]
    lines = [pack["question"].strip()]
    lines.extend(
        point.strip().rstrip(".") + "." for point in pack.get("must_mention") or []
    )
    facts = sorted(
        (
            key
            for key, value in (pack.get("computed") or {}).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ),
        key=lambda key: (
            # quote integers and ratios first: they are the figures an analysis
            # actually talks about; micro-plains (1e-11) are skipped below
            0 if abs(float(pack["computed"][key])) >= 1e-4 else 1,
            key,
        ),
    )

    def words() -> int:
        return sum(len(line.split()) for line in lines)

    position = 0
    guard = 0
    while words() < low and guard < 400:
        if not facts:
            break
        key = facts[position % len(facts)]
        position += 1
        guard += 1
        value = float(pack["computed"][key])
        spelling = _dec(value)
        if spelling == "0" and value != 0.0:
            continue  # too small to quote in plain decimals; skip, do not invent
        lines.append(f"The {key.replace('_', ' ')} stands at {spelling}, per the pack.")
    lines.append(
        "Every figure above is the pack's own; nothing here is drawn from outside it."
    )
    text = "\n".join(lines)
    count = len(text.split())
    if not low <= count <= high:
        raise AssertionError(
            f"dummy prose for {pack['scenario_id']} variant {pack['variant']} "
            f"lands at {count} words, outside the {kind} budget {low}-{high}; "
            "the pack offers too few quotable facts for this budget -- widen "
            "the plan or the budget deliberately, never silently"
        )
    return text


def request_body(pack: dict, kind: str) -> dict:
    """The exact body the render stage posts for *pack*/*kind*, first attempt."""
    return build_body(
        render_brief(pack, kind=kind),
        model=DUMMY_MODEL,
        temperature=config.PROSE_TEMPERATURES[0],
        max_tokens=DEFAULT_MAX_TOKENS,
        think=routing.route(kind).think,
    )


def reply_for(pack: dict, kind: str) -> dict:
    """The canned OpenAI-v1 reply -- self-inspected against the real gate."""
    text = compliant_text(pack, kind)
    problems = gate_violations(pack, text, kind)
    if problems:
        raise AssertionError(
            f"dummy prose for {pack['scenario_id']} variant {pack['variant']} "
            f"({kind}) fails its own gate: {problems}"
        )
    message = {"content": text, "role": "assistant"}
    if routing.route(kind).think:
        message["reasoning_content"] = "Checked line by line against the pack."
    return {
        "choices": [{"finish_reason": "stop", "message": message}],
        "model": DUMMY_MODEL,
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def build_fixture(types: tuple[str, ...], limit: int) -> dict:
    """The first *limit* renderable prose jobs of the real plan, as a replay table."""
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    # Select with limit=None and walk forward ourselves: the limit counts
    # *renderable* entries, so PackError-rejected coordinates (which the
    # render stage will never ask about -- no pack exists for them) do not
    # silently shrink the corpus below the requested size.
    selected = select_prose_jobs(jobs, types=types, limit=None)
    entries: dict[str, dict] = {}
    examined = 0
    for job in selected:
        examined += 1
        try:
            pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        except PackError:
            continue
        entries[canonical_request(request_body(pack, job.record_type))] = reply_for(
            pack, job.record_type
        )
        if len(entries) >= limit:
            break
    if len(entries) < limit:
        raise AssertionError(
            f"only {len(entries)} renderable {types} jobs in {examined} planned; "
            "raise the plan's variant counts -- the 50-row PR2 gate is not "
            "negotiable by shrinking the sample"
        )
    return {
        "entries": entries,
        "model": DUMMY_MODEL,
        "version": 1,
        "meta": {
            "entries": len(entries),
            "jobs_examined": examined,
            "limit": limit,
            "types": sorted(types),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(_HERE, PROSE_FIXTURE_NAME))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="comma list of prose record types (default: analysis)",
    )
    args = parser.parse_args(argv)
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    payload = build_fixture(types, args.limit)
    with open(args.out, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        f"wrote {args.out}: {payload['meta']['entries']} entries "
        f"({payload['meta']['jobs_examined']} plan jobs examined)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
