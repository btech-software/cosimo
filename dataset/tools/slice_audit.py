#!/usr/bin/env python3
"""Is the last slice good enough to scale from? (amendment §F)

``dataset_build.sh full`` refuses unless four things are true of the tree
already on disk, and this is the reader that answers them. It exists as a file
rather than as shell because three of the four are questions about *row
content*, and a corpus gate written in grep is a corpus gate that will one day
pass on a file it could not parse.

The four, in the amendment's own words:

* the invented-number rate on the last slice is 0;
* no row contradicts its own pack (amendment §D: the schedule called the risk
  below the cap, a negative drift called a gain, a winning effect named out of
  pieces that do not add up);
* ``analysis`` and ``grounded`` on one scenario do not open on the same
  sentence (§F: two record types that share a first sentence are one record
  type with two names);
* no two rows of a cell are near-duplicates (the amendment does not name this
  one; it is the failure the numeric gates are blind to by construction, and
  scaling a repainted slice multiplies the repaint);
* no student row contains the factory fingerprint;
* think-off analysis has been measured (``v3_think_ablation.md`` exists and
  records a verdict, not a placeholder);
* the gold bar is present -- checked by the shell, because a missing file is a
  question shell can answer and this reader would only be repeating it.

Exit code is always 0: this is a *report*, and the caller decides. It prints
``slice_audit: ready`` on the last line when every check passed, which is the
one string ``dataset_build.sh`` greps for.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(_HERE)
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config, row as rowlib, write  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.teacher.prompts import BRIEF_KINDS  # noqa: E402
from pipelines.v3.verification.contradictions import (  # noqa: E402
    contradiction_violations,
    tag_of,
)
from pipelines.v3.verification.prose import (  # noqa: E402
    canonical_numbers,
    whitelist_for,
)
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402
from pipelines.v3.verify_v3 import (  # noqa: E402
    _measure_corpus_near_dup,
    _measure_register_separation,
)

#: Where the think ablation is recorded. A path, not a flag: the bar for
#: promoting think-on is a measurement somebody wrote down, and a boolean in an
#: env var is exactly the kind of evidence that gets set to 1 to unblock a run.
ABLATION_PATH = os.path.join(_DATASET, "progress", "v3_think_ablation.md")

#: The string a filled-in ablation must not still contain.
ABLATION_PLACEHOLDER = "NOT YET MEASURED"


def audit(out_dir: str) -> list[str]:
    """Every reason this slice is not ready to scale from, as sentences."""
    problems: list[str] = []
    rows: list[dict] = []
    for kind in (*BRIEF_KINDS, "agentic", "exam", "implementation"):
        path = write.path_for("sft", kind, out_dir)
        try:
            rows.extend(write.read_jsonl(path))
        except ValueError as exc:
            problems.append(f"{path} does not parse: {exc}")
    if not rows:
        problems.append(f"no student rows under {out_dir}/sft -- nothing to read")
        return problems

    print(f"slice_audit: {len(rows)} student rows under {out_dir}/sft")

    # -- 1. the factory fingerprint on a trainable surface -------------------
    leaked = [r.get("id", "?") for r in rows if rowlib.carries_teacher_brief(r)]
    print(f"slice_audit: teacher_leak_rate {len(leaked) / len(rows):.4f}")
    if leaked:
        problems.append(
            f"{len(leaked)} of {len(rows)} rows carry "
            f"{config.TEACHER_FINGERPRINT!r} (first: {leaked[0]})"
        )

    # -- 2. what the recomputed pack says about each row --------------------
    #
    # Against the recomputed pack rather than the row's stored `fact_pack`: the
    # stored copy is what the row *claims*, and a slice audit that trusted the
    # claim would be measuring the renderer's opinion of itself.
    #
    # One recompute, every question asked of it -- the shape `_check_row` uses
    # on the board. A fact computer is not free, and a second pass over the
    # same rows to ask a second question would double the cost of the audit
    # for nothing.
    offenders: list[str] = []
    contradicted: list[str] = []
    tags: dict[str, int] = {}
    ungradeable = 0
    for record in rows:
        try:
            pack = compute_pack(
                str(record.get("work_type")),
                rowlib.family_of(record),
                int(record.get("variant", -1)),
            ).to_dict()
        except (PackError, TypeError, ValueError):
            ungradeable += 1
            continue
        answer = str(record.get("answer") or "")
        tokens = invented_numbers(answer, canonical_numbers(pack), whitelist_for(pack))
        if tokens:
            offenders.append(f"{record.get('id', '?')}: {', '.join(tokens[:3])}")
        # §D: the claims the pack's own arithmetic refutes. Invisible to every
        # numeric axis above, because not one of them invents a number.
        for violation in contradiction_violations(pack, answer):
            tag = tag_of(violation) or "untagged"
            tags[tag] = tags.get(tag, 0) + 1
            contradicted.append(f"{record.get('id', '?')}: {violation}")
    graded = len(rows) - ungradeable
    rate = len(offenders) / graded if graded else 1.0
    print(f"slice_audit: invented_number_rate {rate:.4f} over {graded} graded rows")
    if offenders:
        for line in offenders[:5]:
            print(f"  invented: {line}")
        problems.append(
            f"{len(offenders)} of {graded} rows quote a number their pack does "
            "not authorise; §F wants that at zero before the budget is spent"
        )
    if ungradeable:
        problems.append(
            f"{ungradeable} rows could not be regraded (their coordinates no "
            "longer compute) -- a slice you cannot audit is not a slice to scale"
        )
    print(f"slice_audit: contradictions {len(contradicted)} {tags or ''}".rstrip())
    if contradicted:
        for line in contradicted[:5]:
            print(f"  contradiction: {line}")
        problems.append(
            f"{len(contradicted)} rows contradict their own pack "
            f"({', '.join(f'{tag} x{n}' for tag, n in sorted(tags.items()))}); "
            "§D wants those repaired, not shipped"
        )

    # -- 2c. analysis and grounded as two jobs, not two labels --------------
    shared = _shared_openings(rows)
    print(f"slice_audit: shared analysis/grounded openings {len(shared)}")
    if shared:
        for line in shared[:5]:
            print(f"  shared opening: {line}")
        problems.append(
            f"{len(shared)} scenarios open their analysis and their grounded "
            "row on the same sentence -- §F wants the two record types to be "
            "two jobs, and a shared first sentence is the symptom that they "
            "are one brief with two names"
        )

    # -- 3. holdout rows in the training tree -------------------------------
    leaks = [r.get("id", "?") for r in rows if r.get("holdout")]
    if leaks:
        problems.append(
            f"{len(leaks)} holdout rows are sitting in sft/ (first: {leaks[0]}); "
            "they belong in eval/ (§E)"
        )

    # -- 4. repaint, which the numeric gates cannot see ---------------------
    #
    # Every figure in a repainted row is impeccable -- it came off the same fact
    # computer -- so the invented-number rate above says nothing about it. This
    # is the one question the operator actually has before scaling: am I about
    # to buy two thousand copies of the twenty rows I read?
    board = {
        "corpus near-dup": {"checked": 0, "failures": [], "note": None},
        "register separation": {"checked": 0, "failures": [], "note": None},
    }
    _measure_corpus_near_dup(board, rows)
    _measure_register_separation(board, rows)
    dup = board["corpus near-dup"]
    print(
        f"slice_audit: near-dup {len(dup['failures'])} findings"
        + (f" [{dup['note']}]" if dup.get("note") else "")
    )
    print(f"slice_audit: registers {board['register separation'].get('note')}")
    if dup["failures"]:
        problems.append(
            f"{len(dup['failures'])} near-duplicate pairs in the slice -- the "
            "scenario is being repainted, and scaling multiplies the repaint"
        )

    # -- 5. the think ablation ----------------------------------------------
    if not os.path.isfile(ABLATION_PATH):
        problems.append(
            f"no think ablation at {ABLATION_PATH}: §B says think-off analysis "
            "has to be measured before a full render, not assumed"
        )
    else:
        with open(ABLATION_PATH, encoding="utf8") as handle:
            body = handle.read()
        if ABLATION_PLACEHOLDER in body:
            problems.append(
                f"{ABLATION_PATH} still reads {ABLATION_PLACEHOLDER!r}: the "
                "20-row bake-off has not been run"
            )
    return problems


#: How many opening words of two answers have to match before they are "the
#: same sentence". Eight, because the failure looks like this: a TCA analysis
#: and a TCA grounded row that both open "28.16 bp of arrival, 26.66 bp impact,
#: 1.5 bp half-spread" -- identical for far longer than eight words -- while two
#: genuinely different openings rarely agree past three or four.
_OPENING_WORDS = 8


def _opening(text: str) -> str:
    """The first :data:`_OPENING_WORDS` words of *text*, normalised for compare."""
    words = "".join(
        char if char.isalnum() or char.isspace() or char in ".%-" else " "
        for char in str(text or "").casefold()
    ).split()
    return " ".join(words[:_OPENING_WORDS])


def _shared_openings(rows: list[dict]) -> list[str]:
    """Scenarios whose ``analysis`` and ``grounded`` rows start the same way.

    Keyed on the *pack*, not on the scenario id: a scenario id names a family,
    and two variants of one family are two different sets of numbers that would
    of course open differently. What §F asks is narrower and sharper -- the
    same pack, answered twice, for two different jobs.
    """
    by_coord: dict[tuple, dict[str, str]] = {}
    for record in rows:
        kind = record.get("record_type")
        if kind not in ("analysis", "grounded"):
            continue
        coord = (
            record.get("work_type"),
            rowlib.family_of(record),
            record.get("variant"),
        )
        by_coord.setdefault(coord, {})[kind] = _opening(record.get("answer"))
    out: list[str] = []
    for coord, openings in sorted(by_coord.items(), key=lambda item: str(item[0])):
        if len(openings) < 2:
            continue
        if openings["analysis"] and openings["analysis"] == openings["grounded"]:
            work_type, family, variant = coord
            out.append(
                f"{work_type}/{family} variant {variant}: {openings['analysis']!r}"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="corpus root (default: config.out_dir())")
    args = parser.parse_args(argv)
    problems = audit(os.path.abspath(args.out or config.out_dir()))
    for problem in problems:
        print(f"slice_audit: NOT READY -- {problem}")
    print("slice_audit: ready" if not problems else "slice_audit: not ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
