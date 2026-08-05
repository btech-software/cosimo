"""Shared plumbing for the verification gates.

Record loading, the per-record-type field contract, and a small result type the
gates report through. `verify_all.py` composes everything here into the full-scan
regression runner; each gate is also runnable on its own.
"""
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pipelines import core  # noqa: E402

# The fields that make each record type useful. A record missing one of these
# parses as JSON and teaches nothing, which is how the first attempt shipped
# agentic records with no conversation.
REQUIRED_FIELDS = {
    "exam": ("question", "answer", "reasoning_trace"),
    "analysis": ("question", "answer"),
    "abstention": ("question", "answer"),
    "agentic": ("question", "answer", "tool_schemas", "conversation"),
    "implementation": ("question", "answer", "code"),
    "preference": ("prompt", "chosen", "rejected", "mode"),
}

VALID_DEFECTS = ("underspecified", "unanswerable", "false_premise")

# `FINAL ANSWER:` is a grading contract, not a house style. Putting it on every
# record is what taught the model to answer every question as an exam item.
FINAL_ANSWER_TAG = "FINAL ANSWER:"
EXAM_ONLY_MARKERS = (FINAL_ANSWER_TAG,)


class Result:
    """Accumulates failures for one gate."""

    def __init__(self, name):
        self.name = name
        self.failures = []
        self.warnings = []
        self.checked = 0

    def fail(self, record_id, message):
        self.failures.append(f"{record_id}: {message}")

    def warn(self, message):
        self.warnings.append(message)

    @property
    def ok(self):
        return not self.failures

    def report(self, limit=8):
        status = "PASS" if self.ok else "FAIL"
        print(f"  {self.name:<26} {status}  checked={self.checked} "
              f"failures={len(self.failures)}")
        for line in self.failures[:limit]:
            print(f"      {line}")
        if len(self.failures) > limit:
            print(f"      ... and {len(self.failures) - limit} more")
        for line in self.warnings[:limit]:
            print(f"      WARN {line}")
        return self.ok


def shard_files(shards_dir=None):
    root = shards_dir or core.SHARDS_DIR
    return sorted(glob.glob(os.path.join(root, "*", "*.jsonl")))


def load_records(shards_dir=None):
    """Every record on disk, tagged with the file it came from."""
    records = []
    for path in shard_files(shards_dir):
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num} is not valid JSON: {exc}")
                rec["_source"] = os.path.basename(path)
                records.append(rec)
    return records


def record_type(rec):
    return rec.get("record_type", "exam")


def by_type(records):
    grouped = {}
    for rec in records:
        grouped.setdefault(record_type(rec), []).append(rec)
    return grouped


def supervised_text(rec):
    """The text that would become the supervised target for this record.

    What the model is actually trained to produce, which is what the length and
    terminology gates need to see.
    """
    rtype = record_type(rec)
    if rtype == "exam":
        return f"{rec.get('reasoning_trace', '')}\n\n{rec.get('answer', '')}"
    if rtype == "preference":
        return rec.get("chosen", "")
    if rtype == "agentic":
        return "\n".join(
            str(turn.get("content") or "")
            for turn in rec.get("conversation", [])
            if turn.get("role") == "assistant"
        )
    if rtype == "implementation":
        return f"{rec.get('code', '')}\n\n{rec.get('answer', '')}"
    return rec.get("answer", "")


def approx_tokens(text):
    """Token estimate without a tokenizer.

    The real tokenizer lives in the fine-tune image, not on the host. Words x 1.3
    tracks the Phi-4 tokenizer closely enough for a distribution gate; every
    figure derived from it is labelled approximate.
    """
    return int(len(str(text).split()) * 1.3)


def percentiles(values):
    if not values:
        return {"n": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0}
    ordered = sorted(values)

    def at(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": ordered[-1],
    }
