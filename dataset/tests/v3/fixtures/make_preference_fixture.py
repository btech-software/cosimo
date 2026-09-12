#!/usr/bin/env python3
"""Generate ``preference_fixture.json``: the offline dummy for the preference stage.

The pair stage asks the teacher for two things no other lane asks for: a
*paraphrase of the truth* that is not the truth again, and *fluent prose of
a named wrong model*. This harness composes both locally, deterministically,
and admits each only through the very gates the stage will run them through
(``prefer.gate_problems``, ``prefer.rejected_problems_of``) -- a dummy that
flunks its own gate must fail this build, not the corpus, exactly the
prose fixture's doctrine. The chosen is the same material in a different
telling (figures quoted through their pack's own spellings, points
reversed, question verbatim); the rejected is the chosen's sibling with one
crime committed visibly, per pitfall detector of the contract.

The parent rows are reconstructed, not rendered: the harness walks the
plan's renderable prose jobs in the stage's own selection order, and the
answer of each is the prose harness's ``compliant_text`` -- byte-equal to
what the shipped corpus holds when the render stage replays the prose
fixture (the same pinned dummy model names, the same seed). Entries are
keyed on the dummy's own model name (:data:`DUMMY_MODEL`), never a lane
name, so the bytes hash the same on every box; an offline run hits them by
pointing ``COSIMO_V3_TEACHER_FIXTURE`` at this file and pinning **both**
lane names to it::

    COSIMO_V3_TEACHER_FIXTURE=.../preference_fixture.json \
    TEACHER_PROSE=fixture-pref TEACHER_REASONING=fixture-pref \
    ... cli prefer --types analysis --limit 24

The ``--limit`` that bounds the stage bounds the same deterministic walk
here, so the captured set is exactly what the stage will ask for -- the
prose fixture's ``jobs_examined`` convention, applied to pairs.

Determinism: no wall clock, no RNG beyond the seed-derived draw itself
(``verification.preference.pair_draw`` is the single source of which rows
pair and what crime they carry -- the harness never re-derives it by hand).
Run from the repo root::

    .venv/bin/python dataset/tests/v3/fixtures/make_preference_fixture.py

The committed json must match byte-for-byte; test_preference_fixture enforces
it, and the gate test then runs the real stage over the real shards.
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
for _p in (DATASET, os.path.dirname(DATASET), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The completion budget is part of every request body this file hashes, and
# `client.DEFAULT_MAX_TOKENS` reads COSIMO_V3_MAX_TOKENS at import time. A
# developer with that set -- `.env.example` ships it -- would otherwise
# capture their own budget into the committed table and rekey every entry.
# Dropping it here keeps the promise this module's docstring makes: the bytes
# hash the same on every box.
os.environ.pop("COSIMO_V3_MAX_TOKENS", None)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.prefer import gate_problems, rejected_problems_of  # noqa: E402
from pipelines.v3.render.prose import select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import routing  # noqa: E402
from pipelines.v3.teacher.client import (  # noqa: E402
    build_body,
    canonical_request,
)

from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402
from pipelines.v3.verification.preference import (  # noqa: E402
    WRONG_MODEL_SIGNS,
    chosen_brief,
    pair_draw,
    rejected_brief,
)
from pipelines.v3.verification.prose import stems, whitelist_for  # noqa: E402

PREF_FIXTURE_NAME = "preference_fixture.json"
DUMMY_MODEL = "fixture-pref"
#: The stage's ``--limit`` truncates the same deterministic walk this harness
#: walks, so a stage run at any limit at or below :data:`DEFAULT_LIMIT` asks
#: for a subset of what is captured -- capture wider than any one run asks,
#: deep enough that the walk visits every work type and therefore every
#: licensed crime: ``overconfident_abstention_fail`` is only licensed by the
#: var_es families, ``look_ahead`` only by the dcf ones, and a fixture that has
#: never met a detector has never tested it.
#:
#: 240, up from 120. The depth needed is a property of the *plan*, not a
#: constant: the walk visits work types in sorted order, so how far it must go
#: to reach ``valuation.equity.dcf`` -- the sole licensor of ``look_ahead`` --
#: depends on how many rows the three work types ahead of it contribute. The
#: amendment's WIP counts left every family untruncated by the family cap, and
#: 120 no longer reached the fourth work type. ``test_preference_fixture``
#: asserts the coverage rather than the number, so a plan edit that moves it
#: again fails loudly here instead of silently retiring a detector.
DEFAULT_LIMIT = 240
DEFAULT_TYPES = tuple(sorted(config.PREF_PROBABILITIES))

#: The figure from memory the ``tool_skip`` rejected side quotes, tried in
#: this order until one is found that no pack figure explains. A fixed
#: table, not a roll: the bytes of the fixture must not depend on the box.
_MEMORY_FIGURES = ("8677.13", "6103.27", "7431.19", "4271.41", "9213.56")
#: The future print the ``look_ahead`` rejected side consults: later than
#: every date in ``packs.base.AS_OF_DATES``, which is asserted per pack at
#: build rather than trusted from the pool's readme.
_LOOK_AHEAD_DATE = "2027-06-30"


def _dec(value: float) -> str:
    """Fixed-point rendering through the prose harness's own instrument."""
    return prose_harness._dec(value)


def _quoteable_figures(pack: dict) -> list[tuple[str, float]]:
    """The pack's computed figures that survive a plain-decimal spelling.

    In the order of the prose harness's quoting rule (magnitudes first,
    micro-plaines skipped): a figure that would print ``0`` while being
    nonzero is not quotable, and a paraphrase that misquotes one is not a
    paraphrase.
    """
    computed = pack.get("computed") or {}
    figures = sorted(
        (
            (key, float(value))
            for key, value in computed.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ),
        key=lambda pair: (0 if abs(pair[1]) >= 1e-4 else 1, pair[0]),
    )
    kept: list[tuple[str, float]] = []
    for key, value in figures:
        if _dec(value) == "0" and value != 0.0:
            continue
        kept.append((key, value))
    return kept


def compose_chosen(pack: dict, kind: str) -> str:
    """The second telling: same material, different wires.

    The figures are quoted through the pack's own spellings (the invented-
    number gate leaves no other freedom), the points are kept verbatim --
    the coverage gate reads them anywhere, so they travel in reverse order
    with the figures first -- and the question rides along once, so that
    the two texts share their long shingles nowhere else. The template is
    the prose harness's with the sections reordered and the leading words
    replaced, which is what keeps the shingle overlap far under the pair's
    ceiling instead of near it.
    """
    from pipelines.v3.teacher.prompts import (  # late: the tables
        KIND_SENTENCE_CAPS,
        word_budget,
    )

    low, high = word_budget(kind, pack.get("register") or "")
    figures = _quoteable_figures(pack)
    # Three figures to a sentence, matching the prose harness. One apiece put
    # a desk_chat paraphrase at fifteen sentences against the §C ceiling of
    # twelve -- the register gate's first catch on its own build, and the right
    # catch: a second telling that runs longer than the desk speaks is not the
    # same register, whatever the pack's label says.
    figure_lines = [
        "This telling reads "
        + ", ".join(
            f"{key.replace('_', ' ')} at {(pack.get('display') or {}).get(key) or _dec(value)}"
            for key, value in figures[i : i + 3]
        )
        + "."
        for i in range(0, len(figures), 3)
    ]
    tail_lines = [
        point.strip().rstrip(".") + "."
        for point in reversed(pack.get("must_mention") or [])
    ] + [
        # Not the question verbatim, which is what this line used to carry.
        # A question prints the roundings the answer may not claim -- `1.89%`
        # of a canonical `0.0189` -- so quoting it made the paraphrase commit
        # the rounding drift §D now gates, and the build assertion below said
        # so. The pair's distance from its target comes from the reordering
        # and the figure spellings, not from a block of shared prose.
        f"The scenario is a {pack['work_type']} read, as of {pack['as_of']}.",
        # The register's own closing move, borrowed from the prose harness:
        # `ic_memo` must make a call, and a paraphrase that dropped it would
        # flunk the gate its target passed. A second telling changes the
        # telling, not the register.
        prose_harness._CLOSINGS.get(pack.get("register") or "", ""),
        "Figures as the pack prints them; the arranging is mine.",
    ]

    def words(lines: list[str]) -> int:
        return sum(len(line.split()) for line in lines)

    # Trim before you fill: the figures are the most expendable section
    # (their absence invents nothing and omits no contract point), so the
    # head of that section goes first when the text overshoots -- but never
    # below two of them, a telling with no figures at all is not a telling
    # of a quantified pack.
    while words(figure_lines + tail_lines) > high and len(figure_lines) > 2:
        figure_lines.pop(0)
    lines: list[str] = list(figure_lines) + list(tail_lines)
    #: Padding with no digits of its own: a filler that carried a numeral
    #: would be the paraphrase inventing one, caught by the very gate this
    #: build is proving against.
    #:
    #: Long, not terse, and that is the register gate's doing. Each filler is
    #: one sentence, so short ones bought words at the price of sentences --
    #: and a desk_chat memo has a 200-word floor under a 15-sentence ceiling,
    #: which three-line padding cannot reach. Roughly twenty words apiece
    #: clears both.
    fillers = (
        "Restated for the pair: these are the pack's own numbers arranged in "
        "the desk's own words, with nothing added and nothing quietly dropped.",
        "Nothing in this telling reaches beyond the fact pack, and nothing the "
        "pack asks to be covered has been left out of it on the way through.",
        "The figures stand exactly as they were computed; only the ordering of "
        "the argument around them has moved, which is the whole of the change.",
    )
    filler = 0
    while words(lines) < low and filler < 60:
        lines.insert(
            max(0, len(lines) - 2),
            fillers[filler % len(fillers)],
        )
        filler += 1
    # The kinds that are short by construction -- a citation and a refusal --
    # cap sentences as well as words, and this dummy writes one sentence to a
    # line. Merge rather than drop: every line is carrying a contract point.
    lines = prose_harness.fit_sentences(lines, KIND_SENTENCE_CAPS.get(kind))
    text = "\n".join(lines)
    count = len(text.split())
    if not low <= count <= high:
        raise AssertionError(
            f"paraphrase for {pack['scenario_id']} variant {pack['variant']} "
            f"({kind}) lands at {count} words, outside the budget {low}-{high}"
        )
    return text


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _silence_a_contract_point(pack: dict, base: str) -> str | None:
    """*base* with one ``must_mention`` point genuinely dropped, or ``None``.

    Deleting the line that quoted the anchor used to be enough, because the
    gate matched anchors verbatim. It matches on content-word stems now (see
    ``verification.prose.missing_mentions``, and the anchors it forced the
    packs to shorten), so a dropped line leaves the point's own words strewn
    through the rest of the answer and the "ignored" constraint reads as
    covered -- which is how this maker's own build assertion caught it.

    So silence the point the way the gate reads it: strike the stems that
    belong to this point *and to no other*, wherever they occur. Restricting
    to exclusive stems is what keeps the crime singular -- a shared word
    would take a second contract point down with it, and the pair would then
    be evidence for a different accusation than the one it is labelled with.
    """
    points = list(pack.get("must_mention") or [])
    for index, point in enumerate(points):
        mine = stems(point)
        if not mine:
            continue
        if not mine <= stems(base):
            return base  # already silenced; the detector will see it so
        others = set().union(*(stems(p) for i, p in enumerate(points) if i != index))
        exclusive = mine - others
        if not exclusive:
            continue
        kept = [w for w in base.split() if not stems(w) & exclusive]
        return " ".join(kept)
    return None


def compose_rejected(
    pack: dict, work_type: str, pitfall: str, answer: str, chosen: str
) -> str:
    """The parent's telling with one crime committed visibly.

    The base is the shipped SFT text -- fluent, on-pack, already the length
    of the desk -- because the rejected side's job is not to be sloppy, it
    is to be *wrong in this one named way* (analysis spec §5.7: the v2
    rejected sides failed DPO because they read as templates of nonsense).
    The mutation is the pitfall's own signature, per the detector it must
    satisfy; everything the crime does not touch keeps the pack's figures.
    """
    base = answer
    if pitfall == "wrong_assumption":
        signs = WRONG_MODEL_SIGNS.get(work_type) or ()
        if not signs:
            raise AssertionError(f"no wrong model on file for {work_type!r}")
        return base + f" Say it as {signs[0]} and the figures follow that model."
    if pitfall == "false_precision":
        figures = _quoteable_figures(pack)
        if not figures:
            raise AssertionError(f"{pack['scenario_id']}: no figure to over-quote")
        key, value = figures[0]
        return (
            base
            + f" Printed to eight decimals, {key.replace('_', ' ')} is {float(value):.8f}."
        )
    if pitfall == "ignored_constraint":
        silenced = _silence_a_contract_point(pack, base)
        if silenced is None:
            raise AssertionError(
                f"{pack['scenario_id']} ({pitfall}): no contract point to silence "
                "-- every point shares all its content words with another, so "
                "dropping one drops two and the pair stops being about this crime"
            )
        return silenced
    if pitfall == "look_ahead":
        if not str(pack.get("as_of") or "") < _LOOK_AHEAD_DATE:
            raise AssertionError(
                f"{pack['scenario_id']}: as-of {pack.get('as_of')!r} is not "
                f"before the future print {_LOOK_AHEAD_DATE!r}; the pool moved"
            )
        return base + f" Confirmed against the {_LOOK_AHEAD_DATE} print."
    if pitfall == "overconfident_abstention_fail":
        figures = _quoteable_figures(pack)
        if not figures:
            raise AssertionError(f"{pack['scenario_id']}: nothing to commit to")
        key, value = figures[0]
        return (
            base
            + f" The {key.replace('_', ' ')} settles at {_dec(value)}, and that is the answer."
        )
    if pitfall == "wrong_register":
        figures = _quoteable_figures(pack)
        closing = (
            f"{figures[0][0].replace('_', ' ')} = {_dec(figures[0][1])}"
            if figures
            else "as the pack prints"
        )
        return (
            base + f"\nASSUMPTIONS: as the pack prints them. Step 1. the figures stand."
            f" FINAL ANSWER: {closing}"
        )
    if pitfall == "tool_skip":
        allowed = pack.get("allowed_numbers") or []
        whitelist = whitelist_for(pack)
        for figure in _MEMORY_FIGURES:
            if invented_numbers(f"the tally reads {figure} today", allowed, whitelist):
                return (
                    base
                    + f" The desk's own tally, from memory, adds {figure} of slippage."
                )
        raise AssertionError(
            f"{pack['scenario_id']}: every memorised figure happens to be a pack "
            "figure; widen _MEMORY_FIGURES -- never ship a tool_skip that is not one"
        )
    raise AssertionError(f"no composition for pitfall {pitfall!r}")


def _prompt(pack: dict, kind: str) -> list[dict]:
    """The turn the pair answers into -- the student's question, as the stage
    builds it.

    One user turn, not the parent row's first two messages. Those were the
    teacher's system turn and the JSON contract, and copying them onto every
    pair put the labelling protocol into the preference config as squarely as
    it sat in the supervised one (§A). ``kind`` is unused now and kept in the
    signature because the callers key their tables on (pack, kind) pairs.
    """
    del kind
    return [{"role": "user", "content": pack["question"]}]


def request_body_chosen(pack: dict, kind: str, answer: str) -> dict:
    return build_body(
        chosen_brief(_prompt(pack, kind), answer, kind),
        model=DUMMY_MODEL,
        temperature=config.PREF_TEMPERATURES[0],
        # The lane's cap, not the client's flat default: the amendment moved
        # the completion budget onto the route (think-off 800, think-on 2048),
        # and the budget is part of every body this table hashes.
        max_tokens=routing.route(kind).max_tokens,
        think=routing.route(kind).think,
    )


def request_body_rejected(pack: dict, kind: str, pitfall: str) -> dict:
    return build_body(
        rejected_brief(_prompt(pack, kind), pack["work_type"], pitfall),
        model=DUMMY_MODEL,
        temperature=config.PREF_TEMPERATURES[0],
        # The lane's cap, not the client's flat default: the amendment moved
        # the completion budget onto the route (think-off 800, think-on 2048),
        # and the budget is part of every body this table hashes.
        max_tokens=routing.route(kind, rejected=True).max_tokens,
        think=routing.route(kind, rejected=True).think,
    )


def reply(text: str, think: bool) -> dict:
    message = {"content": text, "role": "assistant"}
    if think:
        message["reasoning_content"] = "Checked against the pack, line by line."
    return {
        "choices": [{"finish_reason": "stop", "message": message}],
        "model": DUMMY_MODEL,
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def build_fixture(types: tuple[str, ...], limit: int) -> dict:
    """The first *limit* pairs of the plan, as a replay table.

    Two orders must be reconciled, and this is the whole comment of this
    function. ``prefer.select_pair_jobs`` -- and with it the stage's
    ``--limit`` -- is **kind-major**: the list is ``abstention`` rows, then
    ``analysis`` rows, ... and the limit truncates that concatenation, so a
    run at limit L asks for the L-prefix of it. This capture is
    **round-robin**: one pair from each kind in turn, until the limit, so
    that a shallow walk still meets every work type and every licensed
    crime. The reconciliation is a prefix property: so long as each kind's
    captured run is at least as long as the deepest prefix the stage will
    take of that kind (true at the committed ``DEFAULT_LIMIT``, where
    earlier kinds exhaust and later kinds keep their full captured runs),
    every request the stage will ever make at that limit is in the table.
    A stage asked to run beyond the committed limit misses loudly -- as
    every miss here does -- and the fix is to regenerate deeper, never to
    let a pair render from a stale table.
    """
    from collections import deque  # late: the walk's one instrument

    plan = inventory.load_plan(config.taxonomy_path())
    jobs = inventory.expand_jobs(plan)
    lanes: list[tuple[str, deque]] = []
    examined = 0
    for kind in sorted(types):
        drawn = deque()
        for job in select_prose_jobs(jobs, types=(kind,), limit=None):
            try:
                pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
            except PackError:
                continue  # the guard's verdict: that row was never shipped
            licensed = sorted((plan.get(job.work_type) or {}).get("pitfalls") or [])
            paired, pitfall = pair_draw(
                job.work_type, job.family, kind, job.variant, licensed
            )
            if paired:
                drawn.append((job, pack, pitfall))
        examined += len(drawn)
        if drawn:
            lanes.append((kind, drawn))
    entries: dict[str, dict] = {}
    pairs = 0
    covered: set[str] = set()
    crimes: set[str] = set()
    while pairs < limit and any(drawn for _, drawn in lanes):
        for kind, drawn in lanes:
            if not drawn:
                continue
            _job, pack, pitfall = drawn.popleft()
            answer = prose_harness.compliant_text(pack, kind)
            chosen = compose_chosen(pack, kind)
            problems = gate_problems(pack, kind, chosen, answer)
            if problems:
                raise AssertionError(
                    f"paraphrase flunks its own gate on {pack['scenario_id']} "
                    f"({kind}): {problems}"
                )
            rejected = compose_rejected(
                pack, pack["work_type"], pitfall, answer, chosen
            )
            problems = rejected_problems_of(pack, rejected, pitfall, chosen)
            if problems:
                raise AssertionError(
                    f"rejected side of {pitfall!r} flunks its own detector on "
                    f"{pack['scenario_id']} ({kind}): {problems}"
                )
            entries[canonical_request(request_body_chosen(pack, kind, answer))] = reply(
                chosen, routing.route(kind).think
            )
            entries[canonical_request(request_body_rejected(pack, kind, pitfall))] = (
                reply(rejected, False)
            )
            pairs += 1
            covered.add(kind)
            crimes.add(pitfall)
            if pairs >= limit:
                break
    if not pairs:
        raise AssertionError(
            f"not one pair drawn from {examined} renderable rows of {sorted(types)}; "
            "the plan or the probabilities moved -- the sample is not negotiable "
            "by shrinking it silently"
        )
    return {
        "entries": entries,
        "model": DUMMY_MODEL,
        "version": 1,
        "meta": {
            "entries": len(entries),
            "pairs": pairs,
            "jobs_examined": examined,
            "limit": limit,
            "types": sorted(covered),
            "pitfalls": sorted(crimes),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(_HERE, PREF_FIXTURE_NAME))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="comma list of parent record types (default: every paired kind)",
    )
    args = parser.parse_args(argv)
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    payload = build_fixture(types, args.limit)
    with open(args.out, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        f"wrote {args.out}: {payload['meta']['pairs']} pairs "
        f"({payload['meta']['entries']} entries over "
        f"{payload['meta']['jobs_examined']} renderable rows; "
        f"crimes {', '.join(payload['meta']['pitfalls'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
