#!/usr/bin/env python3
"""Generate ``impl_fixture.json``: the offline dummy for the implementation stage.

The prose harness composes answers; this one composes the *one field the
implementation teacher is ever asked for* -- the statement of limitations --
and, unlike any other harness here, it must also *execute* to be honest:
every entry is admitted only after ``run_sandboxed`` has passed the composed
public + hidden suite against the composed reference. A dummy that shipped
entries for a suite no instrument passes would replay a broken table, and a
harness that cannot tell a broken table from a stale fixture is the reason
this file exists instead of a hand-written json.

The limitations passage is generic on purpose: no digits (nothing for the
fact-lock to catch or miss), no claim of coverage or warranty in any
direction. It is still self-inspected per pack through the very
:func:`limitations_violations` the live teacher faces -- a dummy that flunks
its own gate must fail this build, not the corpus.

Determinism: no wall clock, no RNG, no environment reads. Entries are keyed
on the dummy's own model name (:data:`DUMMY_MODEL`), never a deployment lane
name, so the bytes hash the same on every box; an offline run hits them by
setting ``TEACHER_REASONING`` to it, the convention the other fixtures set.
Run from the repo root::

    .venv/bin/python dataset/tests/v3/fixtures/make_impl_fixture.py

The committed json must match byte-for-byte; test_impl_fixture enforces it.
"""

from __future__ import annotations

import argparse
import json
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
from pipelines.v3.render.implementation import (  # noqa: E402
    IMPL_KIND,
    impl_brief,
    select_impl_jobs,
)
from pipelines.v3.teacher import routing  # noqa: E402
from pipelines.v3.teacher.client import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    build_body,
    canonical_request,
)
from pipelines.v3.verification.implementation import (  # noqa: E402
    compose_impl_item,
    limitations_violations,
    run_sandboxed,
)

IMPL_FIXTURE_NAME = "impl_fixture.json"

#: The whole planned train implementation slice, deliberately: the family caps
#: admit 30 implementation rows and the pack guards veto the contradictory
#: draws among them, so the fixture captures the renderable remainder. A
#: future veto is safe (an unused entry costs bytes); a future un-veto would
#: be a replay miss, which is the loud kind -- and the drift test would see
#: it first, as a byte the harness grew that the committed file lacks.
DEFAULT_LIMIT = 30

#: The offline model name, pinned into every request body this file keys on.
DUMMY_MODEL = "fixture-impl"

#: The one passage, generic by design: it states what the instrument is and
#: is not without touching a single figure, so the fact-lock has nothing to
#: weigh and the forbidden wall nothing to catch on every pack in the corpus.
_LIMITATIONS = (
    "The instrument reads only the inputs it is handed and models exactly "
    "the arithmetic those inputs define; it takes no market view, offers no "
    "forecast, and gives no warranty of any kind. Readings it cannot parse "
    "are refused, not estimated."
)


def limitations_text(pack: dict) -> str:
    """The canned passage, self-inspected against the live gate per pack."""
    problems = limitations_violations(pack, _LIMITATIONS)
    if problems:
        raise AssertionError(
            f"limitations passage flunks its own gate on "
            f"{pack['scenario_id']} variant {pack['variant']}: {problems}"
        )
    return _LIMITATIONS


def request_body(pack: dict) -> dict:
    """The exact body the stage posts for *pack*, first attempt."""
    item = compose_impl_item(pack)
    return build_body(
        impl_brief(pack, item),
        model=DUMMY_MODEL,
        temperature=config.IMPL_TEMPERATURES[0],
        max_tokens=DEFAULT_MAX_TOKENS,
        think=routing.route(IMPL_KIND).think,
    )


def reply_for(pack: dict) -> dict:
    """The canned reply -- after the suite has actually run green on this pack."""
    item = compose_impl_item(pack)
    passed, log = run_sandboxed(
        item["reference_code"],
        [*item["public_tests"], *item["hidden_tests"]],
        pack.get("inputs") or {},
    )
    if not passed:
        raise AssertionError(
            f"composed suite fails its own reference on "
            f"{pack['scenario_id']} variant {pack['variant']}: {log}; "
            "the table is broken -- fix impl_references, do not bake a fixture "
            "over the corpse"
        )
    text = limitations_text(pack)
    message = {"content": text, "role": "assistant"}
    if routing.route(IMPL_KIND).think:
        message["reasoning_content"] = "Checked against the pack: no figure restated."
    return {
        "choices": [{"finish_reason": "stop", "message": message}],
        "model": DUMMY_MODEL,
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def build_fixture(limit: int) -> dict:
    """The renderable part of the first *limit* planned implementation jobs."""
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    selected = select_impl_jobs(jobs, limit=limit)
    entries: dict[str, dict] = {}
    covered: set[str] = set()
    examined = 0
    for job in selected:
        examined += 1
        try:
            pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        except PackError:
            continue  # the guard's verdict: the fact never existed, nothing to key
        entries[canonical_request(request_body(pack))] = reply_for(pack)
        covered.add(job.work_type)
    if not entries or len(entries) * 2 < examined:
        raise AssertionError(
            f"{examined - len(entries)} of {examined} planned draws drawn were "
            f"vetoed by the pack guards: more than half the slice is being "
            f"rejected. That is a computer defect, not a verdict -- fix the "
            "bands before asking for a fixture"
        )
    return {
        "entries": entries,
        "model": DUMMY_MODEL,
        "version": 1,
        "meta": {
            "entries": len(entries),
            "jobs_examined": examined,
            "limit": limit,
            "record_type": IMPL_KIND,
            "work_types": sorted(covered),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(_HERE, IMPL_FIXTURE_NAME))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args(argv)
    payload = build_fixture(args.limit)
    with open(args.out, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        f"wrote {args.out}: {payload['meta']['entries']} entries "
        f"({payload['meta']['jobs_examined']} plan jobs examined, "
        f"work types {', '.join(payload['meta']['work_types'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
