#!/usr/bin/env python3
"""Assemble the few hundred rows a smoke SFT should train on, and no more.

Two rules, and the second is the one that makes the experiment worth running:

1. **v3 student rows only.** Nothing from v1/v2 -- the question is whether *this*
   corpus teaches anything, and a run that quietly included the 71k exam set
   would answer a different one. (`dataset.mix` is already `[]`; this is the
   belt.)
2. **Nothing on a trap coordinate.** `suites/traps.jsonl` asks its questions of
   four specific packs. Train on those packs and the eval measures memorisation
   -- the model saw the numbers -- and a false "the traps improved" is the
   expensive mistake, because it greenlights spending teacher tokens on more
   rows. Two of the four are the configured holdout families and would be
   dropped by `01_prepare_data` anyway; the other two carry the gold bar, and
   the fence exists for exactly this reason.

What is left is thin, and that is a finding rather than a flaw in the script:
the committed corpus sits on eight pack coordinates, half of which the traps
claim. Exam rows render offline with no teacher, so the stage is topped up from
coordinates the traps do not touch -- real v3 rows, different scenarios, which
is what makes the trap questions a generalisation test rather than a recall
test.

    uv run --group corpus python dataset/tools/smoke_corpus.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(_HERE)
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Files under examples/ that are not training material, whatever they contain.
#: `live_analysis` is the v3.1 capture -- it predates the amendment's briefs and
#: would teach the long-essay style the whole rewrite removed; `_rejected` is a
#: regression fixture; `live_three_registers` is the before-picture.
EXCLUDED_FILES = ("_rejected", "live_analysis", "live_three_registers")

#: The one-row-per-record-type files `make_examples.py` writes. They are the
#: *scripted* teacher -- prose assembled by the fixture harness to prove the
#: schema -- and examples/v3/README.md says in as many words: "do not train on
#: analysis.jsonl. That row is the scripted teacher, and it reads like one. It
#: proves the schema and would poison the voice if it reached SFT."
#:
#: It does. A 40-step smoke that included them produced an adapter answering a
#: VaR question with "Answering the risk.market.var_es question as of
#: 2025-12-31, on the figures given and without any others" -- the harness's
#: own opening line, reproduced verbatim, followed by fluent nonsense about
#: drift being "drawn subtractively". Eight of that run's sixteen rows were
#: these files.
SCRIPTED_EXAMPLES = (
    "analysis.jsonl",
    "memo.jsonl",
    "grounded.jsonl",
    "critique.jsonl",
    "abstention.jsonl",
    "exam.jsonl",
    "agentic.jsonl",
    "implementation.jsonl",
)


def trap_coordinates(suite_path: str) -> set[tuple[str, int]]:
    """``(scenario_id, variant)`` for every pack the trap suite draws on."""
    coords: set[tuple[str, int]] = set()
    if not os.path.isfile(suite_path):
        return coords
    with open(suite_path, encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scenario_id"):
                coords.add((row["scenario_id"], int(row.get("variant", 0))))
    return coords


def collect(examples_dir: str, goldbar: str, traps: set) -> tuple[list, dict]:
    rows, dropped = [], collections.Counter()
    seen: set[str] = set()
    sources = sorted(
        os.path.join(examples_dir, name)
        for name in os.listdir(examples_dir)
        if name.endswith(".jsonl")
        and name not in SCRIPTED_EXAMPLES
        and not any(token in name for token in EXCLUDED_FILES)
    )
    # The gold bar is read only to *exclude* its ids: those rows are the
    # held-out human set, and a training row that near-duplicates one is what
    # verify_v3 axis 13 fails a corpus for.
    gold_ids: set[str] = set()
    if os.path.isfile(goldbar):
        with open(goldbar, encoding="utf8") as handle:
            gold_ids = {json.loads(line)["id"] for line in handle if line.strip()}
    for path in sources:
        with open(path, encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                if row["id"] in gold_ids:
                    dropped["gold bar"] += 1
                    continue
                if (row["scenario_id"], row["variant"]) in traps:
                    dropped["trap coordinate"] += 1
                    continue
                rows.append(row)
    return rows, dict(dropped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="corpus root to write")
    parser.add_argument(
        "--keep-trap-coordinates",
        action="store_true",
        help="train on the packs the traps ask about. The eval then measures "
        "recall as much as method -- useful as a red light (traps flat even "
        "then means the rows teach nothing) and misleading as a green one",
    )
    parser.add_argument(
        "--suite",
        default=os.path.join(_DATASET, "..", "jobs", "fine-tune", "suites", "traps.jsonl"),
    )
    args = parser.parse_args()

    traps = (
        set()
        if args.keep_trap_coordinates
        else trap_coordinates(os.path.abspath(args.suite))
    )
    rows, dropped = collect(
        os.path.join(_DATASET, "examples", "v3"),
        os.path.join(_DATASET, "goldbar", "gold_bar_v3.jsonl"),
        traps,
    )
    sft_dir = os.path.join(os.path.abspath(args.out), "sft")
    os.makedirs(sft_dir, exist_ok=True)
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_kind[row["record_type"]].append(row)
    for kind, kind_rows in sorted(by_kind.items()):
        with open(os.path.join(sft_dir, f"{kind}.jsonl"), "w", encoding="utf8") as fh:
            for row in kind_rows:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"trap coordinates held out: {len(traps)}")
    for coord in sorted(traps):
        print(f"  {coord[0]} v{coord[1]}")
    print(f"dropped: {dropped or 'nothing'}")
    print(f"wrote {len(rows)} rows to {sft_dir}")
    for kind, kind_rows in sorted(by_kind.items()):
        print(f"  {kind:16s} {len(kind_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
