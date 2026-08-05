"""Blocking gate: no generated record may near-duplicate an eval suite prompt.

`jobs/fine-tune/suites/{open_ended,calibration,agentic}.jsonl` are held-out
measurement instruments -- the only evaluation that measures the actual objective
rather than exam accuracy. Generating near-duplicates of those prompts
contaminates them, and a contaminated instrument cannot be un-contaminated: every
cross-round comparison built on it becomes meaningless.

The brief asks for the check to be **explicit and recorded**, not merely
performed, so this writes its result to `verification/suite_overlap.json`
alongside failing the run.

Similarity is token-set Jaccard on content words. That is deliberately crude and
deliberately sensitive: a false positive costs one reworded generator, a false
negative costs the evaluation.

Run standalone:
    python3 verification/suite_overlap.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates import BASE, Result, load_records  # noqa: E402

SUITES_DIR = os.path.join(
    os.path.dirname(BASE), "jobs", "fine-tune", "suites"
)
SUITE_FILES = ("open_ended.jsonl", "calibration.jsonl", "agentic.jsonl")
REPORT_PATH = os.path.join(BASE, "verification", "suite_overlap.json")

# Jaccard above this is treated as the same prompt reworded.
THRESHOLD = 0.6

_STOP = {
    "the", "a", "an", "of", "for", "is", "are", "was", "were", "and", "or", "to",
    "in", "on", "with", "that", "this", "at", "by", "as", "its", "it", "be",
    "what", "how", "why", "which", "do", "does", "you", "your", "i", "me", "my",
    "we", "our", "can", "would", "should", "if", "from", "about", "into", "s",
}


def tokens(text):
    words = re.findall(r"[a-z0-9']+", str(text).lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_suite_prompts():
    """Every prompt in the held-out assistant-eval suites."""
    prompts = []
    for name in SUITE_FILES:
        path = os.path.join(SUITES_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("prompt") or row.get("question") or ""
                if text:
                    prompts.append((name, row.get("id", "?"), text, tokens(text)))
    return prompts


def run(records=None, threshold=THRESHOLD, write_report=True):
    result = Result("suite overlap")
    suite = load_suite_prompts()
    if not suite:
        result.warn(f"no suite files found under {SUITES_DIR}; check skipped")
        return result, {"suite_prompts": 0, "checked": 0, "matches": []}

    records = load_records() if records is None else records
    matches = []
    for rec in records:
        text = rec.get("question") or rec.get("prompt") or ""
        if not text:
            continue
        result.checked += 1
        rec_tokens = tokens(text)
        for suite_name, suite_id, suite_text, suite_tokens in suite:
            score = jaccard(rec_tokens, suite_tokens)
            if score >= threshold:
                matches.append({
                    "record_id": rec.get("id"),
                    "generator": (rec.get("metadata") or {}).get("generator"),
                    "suite": suite_name,
                    "suite_id": suite_id,
                    "jaccard": round(score, 3),
                })
                result.fail(
                    rec.get("id", "?"),
                    f"jaccard {score:.2f} against {suite_name}:{suite_id} -- "
                    f"generating near-duplicates of a held-out suite prompt "
                    f"contaminates the only evaluation that measures the objective",
                )
                break

    report = {
        "threshold": threshold,
        "suite_files": list(SUITE_FILES),
        "suite_prompts": len(suite),
        "checked": result.checked,
        "matches": matches,
        "clean": not matches,
    }
    if write_report:
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
    return result, report


def main():
    print("=== Gate: eval-suite overlap ===")
    result, report = run()
    print(f"  {report['suite_prompts']} suite prompts, "
          f"{report['checked']} records, threshold {report['threshold']}")
    ok = result.report()
    if report["suite_prompts"]:
        print(f"  recorded to {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
