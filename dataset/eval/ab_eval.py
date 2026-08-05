"""Gate 3: blind A/B against the gold bar, on the six axes from the brief.

The previous version of this file scored five axes -- `correctness`,
`reasoning_depth`, `numerical_accuracy`, `educational_value`, `no_hallucination` --
with a hand-rolled heuristic. Two of the brief's axes were missing entirely:
**calibration** and **format appropriateness**. Format appropriateness is the axis
the first corpus failed on, so the one instrument that could have caught the
failure was not measuring it.

The six axes, and why each is here:

  correctness             numerical and conceptual
  reasoning_depth         does it explain *why*, or only *what*
  calibration             does it state assumptions and flag what it cannot know
  format_appropriateness  does the shape of the answer fit the question asked
  terminology_validity    every technical term real and correctly used
  no_hallucination        nothing invented

A batch must win on **all six**. The brief is explicit that winning on correctness
and losing on format appropriateness is a failure, "exactly what shipped last time".

Two scorers:

* `heuristic`  - offline, no API. Cheap, runs in CI, catches gross regressions.
                 It cannot judge whether an analysis is *good*, and does not
                 pretend to: it returns a neutral 3 on the axes it cannot observe.
* `critic`     - a frontier model with the rubric below, reading real records and
                 real gold transcripts, never summaries. This is the real gate.

Because the critic needs an API the dataset box may not have, the harness always
writes the blind A/B task file to disk. That file is the unit of work: score it
here, or hand it to a critic elsewhere and read the scores back.

Run:
    python3 eval/ab_eval.py --prepare              # write the blind A/B task file
    python3 eval/ab_eval.py --score heuristic      # offline scoring
    python3 eval/ab_eval.py --report               # aggregate whatever scores exist
    python3 eval/ab_eval.py --print-rubric         # the critic prompt
"""
import argparse
import json
import os
import random
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "verification"))

from gates import by_type, load_records, record_type, supervised_text  # noqa: E402

GOLD_PATH = os.path.join(BASE, "goldbar", "gold_bar.jsonl")
TASKS_PATH = os.path.join(BASE, "eval", "ab_tasks.jsonl")
SCORES_PATH = os.path.join(BASE, "eval", "ab_scores.jsonl")
REPORT_PATH = os.path.join(BASE, "eval", "ab_results.json")

AXES = (
    "correctness",
    "reasoning_depth",
    "calibration",
    "format_appropriateness",
    "terminology_validity",
    "no_hallucination",
)

RUBRIC = """You are judging two responses to the same finance question. One comes
from a reference set of excellent assistant transcripts; the other is a candidate.
You are not told which is which, and their order is randomised.

Score each response 1-5 on each axis, independently:

  correctness             Numerically and conceptually right. A confident wrong
                          number scores 1 regardless of how well written it is.
  reasoning_depth         Does it explain *why*, or only state *what*? An answer
                          that gives a correct number with no mechanism is shallow.
  calibration             Does it state its assumptions and flag what it cannot
                          know? Answering an underspecified question as though it
                          were fully specified scores 1, however fluent.
  format_appropriateness  Does the shape fit the question? An exam-style
                          ASSUMPTIONS / Step 1 / FINAL ANSWER trace in reply to an
                          open-ended judgement question scores 1. So does a
                          rambling essay in reply to a computation.
  terminology_validity    Every technical term real and used correctly. Watch for
                          a real eponym welded to the wrong concept -- the
                          individual words will look fine ("Durbin-Watson
                          duration", "Sharpe-Sortino tail invariance").
  no_hallucination        Nothing invented: no fabricated data, citations, terms
                          or tool outputs.

Return strict JSON and nothing else:
{"a": {axis: score, ...}, "b": {axis: score, ...}, "notes": "<one sentence>"}
"""


# ---------------------------------------------------------------------------
# preparing the blind comparison
# ---------------------------------------------------------------------------

def load_gold():
    if not os.path.exists(GOLD_PATH):
        return []
    gold = []
    with open(GOLD_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                gold.append(json.loads(line))
    return gold


def prepare(samples_per_type=10, seed=3407):
    """Pair sampled records with gold transcripts, order randomised and recorded.

    Pairing prefers a gold transcript of the same record_type, then the same
    topic. Comparing an analysis record against an exam gold item would measure
    the type difference rather than the quality difference.
    """
    rng = random.Random(seed)
    gold = load_gold()
    if not gold:
        raise SystemExit(
            f"no gold bar at {GOLD_PATH}. It is the reference this gate compares "
            "against -- build it before scoring."
        )
    gold_by_type = {}
    for g in gold:
        gold_by_type.setdefault(record_type(g), []).append(g)

    records = load_records()
    tasks = []
    for rtype, recs in sorted(by_type(records).items()):
        pool = gold_by_type.get(rtype) or gold
        for rec in rng.sample(recs, min(samples_per_type, len(recs))):
            same_topic = [g for g in pool if g.get("topic") == rec.get("topic")]
            reference = rng.choice(same_topic or pool)
            # Randomise which slot the candidate occupies and record it, so the
            # scorer never learns which is which but the aggregator can recover it.
            candidate_is_a = rng.random() < 0.5
            cand, ref = supervised_text(rec), supervised_text(reference)
            tasks.append({
                "task_id": f"ab_{len(tasks):05d}",
                "record_type": rtype,
                "record_id": rec.get("id"),
                "gold_id": reference.get("id"),
                "topic": rec.get("topic"),
                "question": rec.get("question") or rec.get("prompt") or "",
                "a": cand if candidate_is_a else ref,
                "b": ref if candidate_is_a else cand,
                "candidate_slot": "a" if candidate_is_a else "b",
            })
    with open(TASKS_PATH, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return tasks


# ---------------------------------------------------------------------------
# the offline heuristic scorer
# ---------------------------------------------------------------------------

_EXAM_SHAPE = re.compile(r"ASSUMPTIONS:|FINAL ANSWER:|(?:^|\n)\s*Step \d+\.", re.M)
_HEDGE = re.compile(
    r"\b(I would need|I need|cannot|can't|it depends|assuming|assumption|"
    r"unless|before I can|not enough|which .{0,25}\?)\b", re.I)
_WHY = re.compile(r"\b(because|since|the reason|which means|so that|therefore|"
                  r"why|drives|implies|otherwise)\b", re.I)
_COMPUTE_Q = re.compile(r"\bcompute\b|\bcalculate\b|what is the\b|estimate the\b", re.I)

# Axes no offline check can see. Returned as a neutral 3 rather than guessed, so
# the heuristic cannot manufacture a win on a dimension it is blind to.
_UNOBSERVABLE = 3


def _heuristic_axes(text, question):
    """Per-axis scores from the observable surface only.

    Scores length, shape, hedging and explanation markers. It does **not** judge
    whether the finance is right -- nothing offline can -- so correctness,
    terminology and hallucination come back neutral.
    """
    words = len(text.split())
    steps = len(_EXAM_SHAPE.findall(text))
    open_ended = not _COMPUTE_Q.search(question)

    if open_ended and steps >= 3:
        # An exam-shaped answer to an open-ended judgement question is precisely
        # the failure the first run shipped.
        fmt = 1
    elif not open_ended and steps == 0 and words > 400:
        fmt = 2
    else:
        fmt = 4

    return {
        "correctness": _UNOBSERVABLE,
        "reasoning_depth": min(5, 1 + len(_WHY.findall(text))),
        "calibration": min(5, 1 + 2 * len(_HEDGE.findall(text))),
        "format_appropriateness": fmt,
        "terminology_validity": _UNOBSERVABLE,
        "no_hallucination": _UNOBSERVABLE,
    }


def score_heuristic(tasks):
    scored = [{
        "task_id": t["task_id"],
        "scorer": "heuristic",
        "a": _heuristic_axes(t["a"], t["question"]),
        "b": _heuristic_axes(t["b"], t["question"]),
    } for t in tasks]
    with open(SCORES_PATH, "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    return scored


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(tasks, scores):
    by_task = {t["task_id"]: t for t in tasks}
    wins = {axis: {"win": 0, "loss": 0, "tie": 0} for axis in AXES}
    per_type = {}
    for row in scores:
        task = by_task.get(row["task_id"])
        if not task:
            continue
        cand_slot = task["candidate_slot"]
        ref_slot = "b" if cand_slot == "a" else "a"
        bucket = per_type.setdefault(
            task["record_type"],
            {axis: {"win": 0, "loss": 0, "tie": 0} for axis in AXES},
        )
        for axis in AXES:
            cand, ref = row[cand_slot].get(axis, 0), row[ref_slot].get(axis, 0)
            key = "win" if cand > ref else "loss" if cand < ref else "tie"
            wins[axis][key] += 1
            bucket[axis][key] += 1

    def rate(counts):
        """Ties count as half.

        The bar is "not worse than the reference", not "strictly beats it" -- a
        candidate that matches an excellent transcript has not failed. Counting a
        tie as a loss would also make every axis the scorer cannot observe read
        0%, which says "lost" when the truth is "not measured".
        """
        total = sum(counts.values())
        if not total:
            return 0.0
        return round((counts["win"] + 0.5 * counts["tie"]) / total, 3)

    report = {
        "n_tasks": len(scores),
        "scorers": sorted({r.get("scorer", "?") for r in scores}),
        "axes": {a: {**wins[a], "win_rate": rate(wins[a])} for a in AXES},
        "by_record_type": {
            rtype: {a: {**c[a], "win_rate": rate(c[a])} for a in AXES}
            for rtype, c in sorted(per_type.items())
        },
    }
    # The gate is all six axes, not an average: a batch that wins on correctness
    # and loses on format appropriateness has failed.
    report["failing_axes"] = [a for a in AXES if report["axes"][a]["win_rate"] < 0.5]
    report["passing_axes"] = [a for a in AXES if a not in report["failing_axes"]]
    report["pass"] = not report["failing_axes"]
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    return report


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--score", choices=("heuristic",))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--samples-per-type", type=int, default=10)
    parser.add_argument("--print-rubric", action="store_true")
    args = parser.parse_args()

    if args.print_rubric:
        print(RUBRIC)
        return 0

    if args.prepare:
        tasks = prepare(args.samples_per_type)
        print(f"wrote {len(tasks)} blind A/B tasks to {TASKS_PATH}")
        return 0

    tasks = read_jsonl(TASKS_PATH)
    if args.score:
        if not tasks:
            raise SystemExit(f"no tasks at {TASKS_PATH}; run --prepare first")
        print(f"scored {len(score_heuristic(tasks))} tasks -> {SCORES_PATH}")

    if args.report or args.score:
        scores = read_jsonl(SCORES_PATH)
        if not scores:
            raise SystemExit(f"no scores at {SCORES_PATH}")
        report = aggregate(tasks, scores)
        print(f"\n=== Gate 3: blind A/B vs gold bar ({report['n_tasks']} tasks, "
              f"scorer={','.join(report['scorers'])}) ===")
        print(f"  {'axis':<26}{'win':>6}{'loss':>6}{'tie':>6}{'win_rate':>10}")
        for axis in AXES:
            row = report["axes"][axis]
            print(f"  {axis:<26}{row['win']:>6}{row['loss']:>6}{row['tie']:>6}"
                  f"{row['win_rate']:>9.1%}")
        print(f"\n  failing axes: {report['failing_axes'] or 'none'}")
        print(f"  GATE: {'PASS' if report['pass'] else 'FAIL'}")
        if report["scorers"] == ["heuristic"]:
            print("  NOTE: the heuristic scorer returns a neutral 3 on correctness,"
                  "\n        terminology and hallucination. Those three axes are not"
                  "\n        measured here -- run the critic for the real gate.")
        print(f"  written to {REPORT_PATH}")
        return 0 if report["pass"] else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
