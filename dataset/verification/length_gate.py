"""Blocking gate: the response-length distribution of the supervised targets.

The brief makes this a design constraint rather than an outcome, and the reason is
measured: the base model is a long chain-of-thought reasoner averaging ~750 tokens,
and training on the first corpus compressed it to **120**. A corpus whose targets
are uniformly short trains that ability out of the model whatever else it scores.

So the threshold is on the **mixed** set: exam traces are legitimately short, and
the `analysis` records are what carry the tail. Per-type figures are reported
alongside so a collapse in one type cannot hide inside the average.

Token counts are approximate -- the real tokenizer ships in the fine-tune image,
not on the host. `gates.approx_tokens` uses words x 1.3, which tracks the Phi-4
tokenizer closely enough to gate a distribution. Every figure is labelled.

Run standalone:
    python3 verification/length_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates import (  # noqa: E402
    Result, approx_tokens, by_type, load_records, percentiles, supervised_text,
)

# The brief: "a corpus whose p95 target length is under 400 tokens has already
# failed". 800 is the stated target for the long-form fraction; p95 is where it
# has to show up.
MIXED_P95_FLOOR = 800

# Per-type expectations. `None` means reported but not enforced -- an abstention
# response is *supposed* to be short, and gating it long would be wrong.
PER_TYPE_P50_FLOOR = {
    "analysis": 500,
    "exam": None,
    "abstention": None,
    "agentic": None,
    "implementation": None,
    "preference": None,
}


def run(records=None, mixed_p95_floor=MIXED_P95_FLOOR, enforce_mixed=True):
    """Measure the length distribution; enforce the mixed-set floor when asked.

    `enforce_mixed=False` is for the smoke corpus, which generates exactly one
    variant per generator and therefore carries a deliberately unrepresentative
    type mix (38% abstention against a shipped 13%). The mixed-set p95 floor is
    a property of the *shipped* composition, so applying it to a corpus built
    under different weights measures the sampling, not the data. Per-type floors
    still apply there -- those are composition-independent.
    """
    result = Result("response length")
    records = load_records() if records is None else records
    if not records:
        result.warn("no records to measure")
        return result, {}

    grouped = by_type(records)
    stats = {}
    for rtype, recs in sorted(grouped.items()):
        lengths = [approx_tokens(supervised_text(r)) for r in recs]
        stats[rtype] = percentiles(lengths)

    mixed = [approx_tokens(supervised_text(r)) for r in records]
    stats["_mixed"] = percentiles(mixed)
    result.checked = len(records)

    if enforce_mixed and stats["_mixed"]["p95"] < mixed_p95_floor:
        result.fail(
            "corpus",
            f"mixed-set p95 is {stats['_mixed']['p95']} approx tokens, below the "
            f"{mixed_p95_floor} floor. The first corpus collapsed the base model "
            f"from ~750 to 120 tokens; a short-tailed corpus does it again.",
        )

    for rtype, floor in PER_TYPE_P50_FLOOR.items():
        if floor is None or rtype not in stats:
            continue
        if stats[rtype]["p50"] < floor:
            result.fail(
                rtype,
                f"p50 is {stats[rtype]['p50']} approx tokens, below the {floor} "
                f"floor for this record type",
            )
    return result, stats


def print_table(stats):
    print(f"  {'record_type':<16}{'n':>7}{'mean':>8}{'p50':>7}{'p90':>7}"
          f"{'p95':>7}{'max':>8}")
    for rtype, s in sorted(stats.items()):
        label = "MIXED" if rtype == "_mixed" else rtype
        print(f"  {label:<16}{s['n']:>7}{s['mean']:>8}{s['p50']:>7}"
              f"{s['p90']:>7}{s['p95']:>7}{s['max']:>8}")
    print("  (approximate tokens: words x 1.3; the real tokenizer is in the "
          "fine-tune image)")


def main():
    print("=== Gate: response-length distribution ===")
    result, stats = run()
    if stats:
        print_table(stats)
    ok = result.report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
