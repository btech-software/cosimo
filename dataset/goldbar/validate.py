"""The gold bar's own gate: schema, composition, and contamination.

The gold bar is the *reference* the A/B in `eval/ab_eval.py` scores against, which
makes it the definition of the target. The brief is blunt about why that matters:
last time the bar was a set of exam questions, so the target was exam items, and
the model faithfully became an exam solver. Build the bar first and build it right.

What this checks:

  1. valid JSONL, one object per line
  2. required fields, and correct types. The v1 file predates the `record_type`
     discriminator, so every legacy transcript is untyped; `--repair` backfills it
     as `exam`, which is what all twelve of them are. It also rewrites any value
     stored as a Python repr rather than JSON, should one appear.
  3. composition against the target table -- a bar that is 90% exam items
     re-creates the failure it exists to prevent
  4. `FINAL ANSWER:` only on exam transcripts
  5. no overlap with `jobs/fine-tune/suites/*.jsonl` -- the bar and the eval suites
     are independent instruments; contaminating one with the other collapses two
     measurements into one
  6. length -- an exemplary open-ended answer is not 40 tokens

Run:
    python3 goldbar/validate.py
    python3 goldbar/validate.py --repair    # rewrite str()-serialized values as JSON
"""
import argparse
import ast
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "verification"))

from gates import Result, approx_tokens, percentiles  # noqa: E402
import suite_overlap  # noqa: E402

GOLD_PATH = os.path.join(BASE, "goldbar", "gold_bar.jsonl")

# The brief asks for ~200 transcripts of what an excellent quant assistant actually
# says. Mirrors the objective, not the exam corpus.
TARGET_COMPOSITION = {
    "analysis": 70,
    "abstention": 30,
    "agentic": 30,
    "implementation": 25,
    "exam": 45,
}
TARGET_TOTAL = sum(TARGET_COMPOSITION.values())

REQUIRED = ("id", "record_type", "program", "topic", "subtopic", "question")

# Minimum supervised-answer length per type, in approximate tokens. An exemplary
# open-ended answer that is shorter than this is not exemplary.
MIN_TOKENS = {"analysis": 400, "abstention": 60, "implementation": 100,
              "agentic": 0, "exam": 0}


def load(path=GOLD_PATH):
    """Parse the gold bar, reporting the line number of any bad row."""
    records, problems = [], []
    if not os.path.exists(path):
        return records, [f"{path} does not exist"]
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                problems.append(f"line {line_num}: not valid JSON ({exc})")
    return records, problems


def _looks_stringified(value):
    """A Python repr that was written where JSON was meant."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (
        stripped in ("True", "False", "None")
        or (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("{") and stripped.endswith("}"))
    )


def repair(path=GOLD_PATH):
    """Rewrite `str()`-serialized values as real JSON, in place.

    The v1 file was written with `str()` instead of `json.dumps`, so booleans
    arrived as `"True"` and lists as their Python repr. Values are recovered with
    `ast.literal_eval`, which parses Python literals and executes nothing.
    """
    records, problems = load(path)
    if problems:
        raise SystemExit("cannot repair a file that does not parse:\n  "
                         + "\n  ".join(problems))
    fixed = 0
    for rec in records:
        for key, value in list(rec.items()):
            if _looks_stringified(value):
                try:
                    rec[key] = ast.literal_eval(value.strip())
                    fixed += 1
                except (ValueError, SyntaxError):
                    pass
        # Legacy transcripts predate the record_type discriminator. All twelve
        # v1 entries are exam items, and an untyped transcript cannot be paired
        # by type in the A/B.
        if not rec.get("record_type"):
            rec["record_type"] = "exam"
            fixed += 1
        # A distractor equal to the correct answer is never valid -- it makes the
        # item unanswerable and it is the axis-4 rule verify_all enforces on the
        # corpus. Six of the twelve v1 transcripts shipped with one.
        answer = str(rec.get("answer") or "")
        distractors = rec.get("distractors")
        if isinstance(distractors, list) and answer:
            deduped = [d for d in distractors if str(d) != answer]
            if len(deduped) != len(distractors):
                rec["distractors"] = deduped
                fixed += 1
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return fixed, len(records)


def answer_text(rec):
    if rec.get("record_type") == "implementation":
        return f"{rec.get('code', '')}\n{rec.get('answer', '')}"
    return rec.get("answer") or ""


def run(path=GOLD_PATH):
    result = Result("gold bar")
    records, problems = load(path)
    for problem in problems:
        result.fail("file", problem)
    result.checked = len(records)
    if not records:
        return result, {}

    seen = set()
    counts = {}
    for rec in records:
        rid = rec.get("id", "?")
        rtype = rec.get("record_type")
        counts[rtype] = counts.get(rtype, 0) + 1
        for field in REQUIRED:
            if not rec.get(field):
                result.fail(rid, f"missing required field {field!r}")
        if rid in seen:
            result.fail(rid, "duplicate id")
        seen.add(rid)
        for key, value in rec.items():
            if _looks_stringified(value):
                result.fail(rid, f"{key!r} is a str()-serialized Python literal "
                                 f"({value[:40]!r}); run --repair")
        if rtype != "exam" and "FINAL ANSWER:" in answer_text(rec):
            result.fail(rid, f"{rtype} transcript carries 'FINAL ANSWER:'")
        if rtype == "exam":
            answer = str(rec.get("answer") or "")
            distractors = rec.get("distractors") or []
            if isinstance(distractors, list) and answer in [str(d) for d in distractors]:
                result.fail(rid, "distractors contain the correct answer")
        floor = MIN_TOKENS.get(rtype, 0)
        if floor and approx_tokens(answer_text(rec)) < floor:
            result.fail(rid, f"{rtype} answer is "
                             f"{approx_tokens(answer_text(rec))} approx tokens, "
                             f"below the {floor} floor for an exemplar")

    # Composition. Reported always; failed only once the bar is near full size, so
    # a partially built bar is not spammed with failures while it is being written.
    if len(records) >= TARGET_TOTAL * 0.9:
        for rtype, target in TARGET_COMPOSITION.items():
            actual = counts.get(rtype, 0)
            if actual < target * 0.8:
                result.fail("composition",
                            f"{rtype}: {actual} transcripts against a target of "
                            f"{target}")
    else:
        result.warn(f"{len(records)}/{TARGET_TOTAL} transcripts; composition not "
                    f"enforced until the bar is near full size")

    overlap, _ = suite_overlap.run(records=records, write_report=False)
    for failure in overlap.failures:
        result.fail("suite-overlap", failure)

    stats = {
        "total": len(records),
        "by_type": dict(sorted(counts.items())),
        "target": TARGET_COMPOSITION,
        "answer_tokens": percentiles(
            [approx_tokens(answer_text(r)) for r in records]),
    }
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true",
                        help="rewrite str()-serialized values as JSON, in place")
    args = parser.parse_args()

    if args.repair:
        fixed, total = repair()
        print(f"repaired {fixed} stringified value(s) across {total} records")

    print("=== Gold bar ===")
    result, stats = run()
    if stats:
        print(f"  {stats['total']} transcripts (target {TARGET_TOTAL})")
        for rtype, target in sorted(TARGET_COMPOSITION.items()):
            actual = stats["by_type"].get(rtype, 0)
            bar = "#" * int(20 * min(1.0, actual / target))
            print(f"    {rtype:<16}{actual:>4} / {target:<4} {bar}")
        print(f"  answer length (approx tokens): {stats['answer_tokens']}")
    ok = result.report()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
