#!/usr/bin/env python3
"""Regenerate ``examples/v3/<record_type>.jsonl`` -- one student row each (§H).

What this replaces: ``dataset/pipelines/v3/example_generation.jsonl``, nine
rows, all of them ``analysis``, every one carrying ``messages`` that opened
with the teacher's system turn and the fact pack as JSON. It was the only
committed picture of "what a v3 row looks like", and what it pictured was the
labelling protocol -- so a reader learning the corpus from it learned the wrong
shape, and the amendment's §A calls it out by name.

Eight files now, one per record type, each holding exactly one row, each row
produced by the *real* renderer through a scripted teacher. That last part is
the point: an example hand-written to look right is an example that drifts the
first time the schema moves. These are generated from the same builders the
corpus uses, so a schema change either regenerates them or fails loudly here.

The teacher transcripts the old file carried are not thrown away -- they move
to ``examples/v3/_teacher_logs/``, which is where the two-surface split says
they belong: readable, gitignored in a working tree, and outside every glob
that walks the corpus.

Run from the repo root::

    uv run --group corpus python dataset/examples/v3/make_examples.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
for _p in (
    _DATASET,
    os.path.dirname(_DATASET),
    os.path.join(_DATASET, "tests", "v3", "fixtures"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_agentic_fixture as agentic_harness  # noqa: E402
import make_prose_fixture as prose_harness  # noqa: E402

from pipelines.v3 import row as rowlib  # noqa: E402
from pipelines.v3.render.agentic import KIND as AGENTIC_KIND  # noqa: E402
from pipelines.v3.render.agentic import render_agentic_row  # noqa: E402
from pipelines.v3.render.exam import build_exam_row  # noqa: E402
from pipelines.v3.render.implementation import build_impl_row  # noqa: E402
from pipelines.v3.render.prose import render_prose_row  # noqa: E402
from pipelines.v3.stage import pack_record  # noqa: E402
from pipelines.v3.teacher.client import Teacher  # noqa: E402
from pipelines.v3.verification.implementation import IMPL_KIND  # noqa: E402
from pipelines.v3.teacher.prompts import BRIEF_KINDS  # noqa: E402

#: The coordinate each example is drawn from. One per record type, spread
#: across work types on purpose: eight examples of one computer would show the
#: schema and hide the range, and the range is half of what an example is for.
COORDS = {
    "analysis": ("execution.tca.arrival", "large_cap_intraday", 0),
    "memo": ("portfolio.attribution.brinson_carino", "global_equity_long", 0),
    "grounded": ("risk.market.var_es", "rates_book", 0),
    "critique": ("valuation.equity.multiples", "specialty_retail", 0),
    "abstention": ("valuation.equity.dcf", "mature_consumer", 0),
    "exam": ("execution.tca.arrival", "large_cap_intraday", 0),
    IMPL_KIND: ("execution.tca.arrival", "large_cap_intraday", 0),
    # The agentic example is the one that has to name a *rank*: the fault
    # schedule is a function of the job's ordinal in the shared selector, and a
    # trajectory rendered at the wrong rank replays against the wrong schedule.
    # Rank 0 is a `no_call` job -- the row that teaches "the pack already has
    # the answer, so calling a tool is waste" -- which is the agentic behaviour
    # least visible in a transcript and therefore the one worth showing.
    AGENTIC_KIND: ("execution.tca.arrival", "large_cap_intraday", 0),
}

#: The agentic example's selector ordinal. Not derived from COORDS: the ordinal
#: is a property of the whole plan's walk, and pinning it here keeps the example
#: reproducible when the plan's variant counts move.
AGENTIC_RANK = 0


class Scripted:
    """A teacher that answers with one prepared text, whatever it is asked."""

    def __init__(self, text: str):
        self._text = text

    def post(self, body: dict) -> dict:  # noqa: ARG002 -- the ask is irrelevant
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": self._text},
                }
            ],
            "model": "example-scripted",
            "usage": {"total_tokens": max(1, len(self._text.split()))},
        }


def _write(kind: str, row: dict, log: dict | None) -> None:
    path = os.path.join(_HERE, f"{kind}.jsonl")
    with open(path, "w", encoding="utf8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
    leaked = rowlib.carries_teacher_brief(row)
    assert not leaked, f"{kind}: the example carries the teacher fingerprint"
    assert row.get("verified") is True, f"{kind}: example does not claim verified"
    assert "fact_pack" in row, f"{kind}: example carries no fact pack"
    print(f"wrote {path}  ({row['id']})")
    if log is not None:
        logs = os.path.join(_HERE, "_teacher_logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, f"{kind}.json"), "w", encoding="utf8") as handle:
            json.dump(log, handle, indent=1, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


def main() -> int:
    for kind, (work_type, family, variant) in COORDS.items():
        pack_line = pack_record(work_type, family, variant)
        pack = rowlib.strip_envelope(pack_line)
        if kind in BRIEF_KINDS:
            teacher = Teacher(Scripted(prose_harness.compliant_text(pack, kind)))
            outcome = render_prose_row(teacher, pack_line, kind=kind)
        elif kind == "exam":
            outcome = build_exam_row(pack_line)
        elif kind == AGENTIC_KIND:
            outcome = render_agentic_row(
                agentic_harness.ScriptedTeacher(pack, AGENTIC_RANK, {}),
                pack_line,
                rank=AGENTIC_RANK,
            )
        else:
            teacher = Teacher(
                Scripted(
                    "The instrument assumes every reading is finite and stated in "
                    "the units the specification names; it models neither the "
                    "execution schedule nor any regime change in the inputs, and "
                    "it refuses rather than extrapolates when a reading is absent."
                )
            )
            outcome = build_impl_row(teacher, pack_line)
        if outcome["row"] is None:
            raise SystemExit(
                f"{kind}: the example coordinate dead-lettered "
                f"({outcome['dead_letter'].get('reason')}); pick another in COORDS"
            )
        _write(kind, outcome["row"], outcome.get("log"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
