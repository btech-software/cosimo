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

# The completion budget is part of every request body this file hashes, and
# `client.DEFAULT_MAX_TOKENS` reads COSIMO_V3_MAX_TOKENS at import time. A
# developer with that set -- `.env.example` ships it -- would otherwise
# capture their own budget into the committed table and rekey every entry.
# Dropping it here keeps the promise this module's docstring makes: the bytes
# hash the same on every box.
os.environ.pop("COSIMO_V3_MAX_TOKENS", None)

from pipelines.v3 import config, inventory  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.prose import select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import routing  # noqa: E402
from pipelines.v3.teacher.client import build_body, canonical_request  # noqa: E402
from pipelines.v3.teacher.prompts import KIND_SENTENCE_CAPS, points_for  # noqa: E402
from pipelines.v3.teacher.prompts import render_brief, word_budget  # noqa: E402
from pipelines.v3.verification.prose import gate_violations  # noqa: E402

PROSE_FIXTURE_NAME = "prose_fixture.json"
DEFAULT_TYPES = ("analysis",)
DEFAULT_LIMIT = 50

#: How many *holdout*-family rows ride along, so `eval-slice` runs offline.
#: Smaller than the train arm on purpose: this half exists to prove the path,
#: not to be a corpus.
DEFAULT_HOLDOUT_LIMIT = 15

#: The offline model name, pinned into every request body this file keys on.
#: Never a deployment lane name: the bytes of the replay table must not depend
#: on what a given box has its ``TEACHER_*`` env pointed at.
DUMMY_MODEL = "fixture-prose"


def _dec(value: float) -> str:
    """Fixed-point rendering, no sci notation, no thousands separators.

    Trimmed fixed-point: every pack value the harness quotes is a number the
    pack's own ``assemble_numbers`` walked into ``allowed_numbers``, so the token
    only has to *round-trip inside the gate's tolerance*, and plain decimal
    notation is the spelling whose tokens the number gate parses without
    ambiguity.

    Capped at ``config.PROSE_MAX_DECIMALS`` because the dummy has to satisfy the
    same prose gate a live teacher does, and the gate now refuses a figure spelled
    past desk precision. The cap bites on real packs: 1,522 published figures
    carry 9-12 decimals, so rendering at ``%.10f`` made the dummy quote portfolio
    weights like ``0.472041725693`` -- which the build assertion below caught the
    moment the axis existed. Rounding here keeps it inside the gate's 0.5%
    tolerance by a wide margin.
    """
    text = f"{float(value):.{config.PROSE_MAX_DECIMALS}f}".rstrip("0")
    if text.endswith("."):
        text = text[:-1]
    return text or "0"


#: How many figures one composed sentence carries. Three, because the dummy
#: has to clear a 120-word floor *and* a 12-sentence desk_chat ceiling at the
#: same time, and one fact per sentence cannot do both: 120 words of ten-word
#: sentences is twelve sentences before a single must_mention anchor is
#: written. Packing the facts is also simply what the register looks like --
#: a desk writes "cost is 28.16 bp on 0.048555 participation, 26.66 of it
#: impact", not three sentences with one number each.
FACTS_PER_SENTENCE = 3

#: How each register signs off. `ic_memo` must make a call (see
#: `verification.register`), `risk_committee` constrains rather than directs,
#: and `desk_chat` simply stops -- which is the register.
_CLOSINGS = {
    "ic_memo": (
        "Our call is to hold the position at its current weight and revisit on "
        "the next print; every figure above is stated as given."
    ),
    "risk_committee": (
        "The limit stands over this horizon on the assumption stated, and the "
        "exposure is reported rather than directed, on the figures as given."
    ),
    "desk_chat": (
        "Every figure above is as given; nothing here is drawn from anywhere else."
    ),
    "": "Every figure above is as given; nothing here is drawn from anywhere else.",
}


def fit_sentences(lines: list[str], cap: int | None) -> list[str]:
    """Merge *lines* until there are at most *cap* of them.

    The dummy writes one sentence to a line, which is what makes it easy to
    read and what put a ten-sentence ``grounded`` answer against an
    eight-sentence ceiling the moment the kinds grew one. Merging rather than
    dropping, because every line here is carrying something the gate checks --
    a ``must_mention`` anchor, a figure, the register's closing move -- and a
    harness that satisfied a ceiling by deleting contract points would be
    proving the gate against a text that no longer meets it.

    Shared with the preference harness: the two dummies have to clear the same
    gate, and a second copy of this would drift from the first.
    """
    if not cap or len(lines) <= cap:
        return lines
    group = math.ceil(len(lines) / cap)
    merged: list[str] = []
    for start in range(0, len(lines), group):
        chunk = [line.strip().rstrip(".") for line in lines[start : start + group]]
        merged.append("; ".join(part for part in chunk if part) + ".")
    return merged


def compliant_text(pack: dict, kind: str) -> str:
    """A completion that satisfies the contract for *pack*, assembled locally.

    Mirrors what a good teacher answer looks like -- the question's facts, in
    the answer's own words -- minus any freedom to be wrong: nothing here is
    outside the pack's fields, and every figure is the pack's own *canonical*
    spelling.

    It no longer opens by quoting the question, and that is the amendment
    biting on its own harness. The question prints roundings the answer may not
    claim -- ``1.21%`` for a canonical ``0.0121``, ``3.0 bp`` for a spread of
    ``3`` -- so a dummy that restated the question verbatim was committing
    exactly the two defects §D adds gates for, and the build assertion below
    caught it the moment they existed. Which is the harness working: a fixture
    that could not pass the live gate was never a fixture for the live gate.
    """
    low, high = word_budget(kind, pack.get("register") or "")
    if kind == "abstention":
        # The abstention lane is asked a question the figures cannot reach, so
        # the dummy declines it. It used to answer the pack's own question and
        # the gate had no way to tell, which is exactly the failure the live
        # lane shipped until the packs grew a gap of their own.
        missing = (points_for(pack, kind) or ["the figure this needs"])[0]
        lines = [
            f"This cannot be answered from what is given as of {pack['as_of']}.",
            f"The {missing} is not among the figures, and nothing here stands "
            "in for it.",
            "Supply that figure and the question becomes answerable; without "
            "it any number offered would be invented rather than measured.",
        ]
        text = "\n".join(lines)
        count = len(text.split())
        if not low <= count <= high:
            raise AssertionError(
                f"dummy abstention for {pack.get('scenario_id')} lands at "
                f"{count} words, outside {low}-{high}"
            )
        return text
    lines = [
        f"Answering the {pack['work_type']} question as of {pack['as_of']}, "
        "on the figures given and no others."
    ]
    lines.extend(point.strip().rstrip(".") + "." for point in points_for(pack, kind))
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
        clause: list[str] = []
        while len(clause) < FACTS_PER_SENTENCE and guard < 400:
            key = facts[position % len(facts)]
            position += 1
            guard += 1
            value = float(pack["computed"][key])
            # The pack's own spelling wins where it has one: §A.3 made that a
            # gate, and a dummy that wrote `0.048555` for a `4.86%` figure was
            # committing the defect the gate exists to catch -- which the
            # build assertion below caught the hour the axis was added.
            spelling = (pack.get("display") or {}).get(key) or _dec(value)
            if spelling == "0" and value != 0.0:
                continue  # too small to quote in plain decimals; skip, do not invent
            clause.append(f"{key.replace('_', ' ')} at {spelling}")
        if not clause:
            break
        lines.append(
            "Reading the figures directly, " + ", ".join(clause) + ", each as computed."
        )
    # The closing move, chosen by register. Not decoration: `ic_memo` now has a
    # positive requirement -- a committee memo that surveys and stops is a
    # research note with the wrong label -- and a dummy that could not satisfy
    # it was never a dummy for the live gate. Giving the three registers three
    # different closings also gives the fixture corpus real separation on the
    # profile axis, instead of one voice wearing three labels, which is the
    # collapse axis 16 exists to report.
    lines.append(_CLOSINGS.get(pack.get("register") or "", _CLOSINGS[""]))
    lines = fit_sentences(lines, KIND_SENTENCE_CAPS.get(kind))
    text = "\n".join(lines)
    count = len(text.split())
    if not low <= count <= high:
        where = pack.get("scenario_id") or pack.get("work_type") or "<pack>"
        raise AssertionError(
            f"dummy prose for {where} variant {pack.get('variant')} lands at "
            f"{count} words, outside the {kind} budget {low}-{high} with "
            f"{len(pack.get('must_mention') or [])} points to cover: either the "
            "pack offers too few quotable facts to reach the floor, or it "
            "carries more points than the ceiling can hold. Both are authoring "
            "errors -- widen the plan or the budget deliberately, never silently"
        )
    return text


def request_body(pack: dict, kind: str) -> dict:
    """The exact body the render stage posts for *pack*/*kind*, first attempt."""
    route = routing.route(kind)
    return build_body(
        render_brief(pack, kind=kind),
        model=DUMMY_MODEL,
        temperature=config.PROSE_TEMPERATURES[0],
        # The lane's cap, not the client's flat default: the budget is part of
        # every body this file hashes, and the amendment moved it onto the
        # route. A fixture keyed on 16384 would miss every real request.
        max_tokens=route.max_tokens,
        think=route.think,
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


def _capture(types, limit: int, *, holdout: bool, entries: dict) -> tuple[int, int]:
    """Fill *entries* with up to *limit* renderable jobs of one cohort.

    Returns ``(captured, examined)``. Select with ``limit=None`` and walk
    forward: the limit counts *renderable* entries, so PackError-rejected
    coordinates -- which the render stage never asks about, no pack existing
    for them -- do not silently shrink the table below the requested size.
    """
    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    selected = select_prose_jobs(jobs, types=types, limit=None, holdout=holdout)
    captured = examined = 0
    for job in selected:
        examined += 1
        try:
            pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        except PackError:
            continue
        entries[canonical_request(request_body(pack, job.record_type))] = reply_for(
            pack, job.record_type
        )
        captured += 1
        if captured >= limit:
            break
    return captured, examined


def build_fixture(
    types: tuple[str, ...], limit: int, holdout_limit: int = DEFAULT_HOLDOUT_LIMIT
) -> dict:
    """The real plan's first renderable prose jobs of *both* cohorts.

    The holdout half is new, and it is what makes the operator's `eval-slice`
    mode runnable offline. Amendment §E renders holdout families into
    ``eval/``, and that path sat on the stage order with no fixture behind it:
    the table only ever captured ``holdout=False``, so the one command whose
    whole job is to produce the eval tree could not be exercised without
    billing a teacher. A replay table that covers only the paths already easy
    to run is a replay table that stops covering the interesting ones.

    Both cohorts share one table, keyed by request hash as always, so no entry
    can be served to the wrong cohort: the coordinates differ, so the packs
    differ, so the briefs differ.
    """
    entries: dict[str, dict] = {}
    captured, examined = _capture(types, limit, holdout=False, entries=entries)
    if captured < limit:
        raise AssertionError(
            f"only {captured} renderable {types} train jobs in {examined} planned; "
            "raise the plan's variant counts -- the 50-row PR2 gate is not "
            "negotiable by shrinking the sample"
        )
    held, held_examined = _capture(types, holdout_limit, holdout=True, entries=entries)
    if held < holdout_limit:
        raise AssertionError(
            f"only {held} renderable {types} holdout jobs in {held_examined} "
            "planned; the eval tree needs a fixture behind it or `eval-slice` "
            "can only be run live"
        )
    return {
        "entries": entries,
        "model": DUMMY_MODEL,
        "version": 1,
        "meta": {
            "entries": len(entries),
            # `jobs_examined` keeps its established meaning -- how far the
            # *train* walk went -- because six tests slice
            # `select_prose_jobs(limit=None)[:jobs_examined]` with it to
            # rebuild exactly the job list this table answers. Summing both
            # cohorts into it would have handed every one of them a window
            # wider than the table, which is a fixture miss reported as a
            # render failure. The holdout walk reports beside it.
            "jobs_examined": examined,
            "holdout_jobs_examined": held_examined,
            "limit": limit,
            "holdout_limit": holdout_limit,
            "train_rows": captured,
            "holdout_rows": held,
            "types": sorted(types),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(_HERE, PROSE_FIXTURE_NAME))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--holdout-limit", type=int, default=DEFAULT_HOLDOUT_LIMIT)
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="comma list of prose record types (default: analysis)",
    )
    args = parser.parse_args(argv)
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    payload = build_fixture(types, args.limit, args.holdout_limit)
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
