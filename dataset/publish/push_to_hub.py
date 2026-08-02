"""Publish the Cosimo dataset to the Hugging Face Hub.

Builds two configs on the Hub repo:
  - default:            one split per program (cfa_level_i ... frm_part_2),
                        full record schema from FORMAT.md, Parquet on the Hub.
  - preference_pairs:   flattened DPO/ORPO-ready rows (prompt/chosen/rejected)
                        from the ~35% of records carrying a preference_pair.

Run from the repository root:
    <venv>/bin/python publish/push_to_hub.py

Requires: datasets, huggingface_hub; an authenticated `hf` login.
The repo is created PRIVATE; flip to public on the Hub after reviewing.
"""

import glob
import json
import os
import sys

from datasets import Dataset, DatasetDict

REPO_ID = "btech-software/cosimo-cfa-frm-71k"

PROGRAMS = {
    "CFA_Level_I": "cfa_level_i",
    "CFA_Level_II": "cfa_level_ii",
    "CFA_Level_III": "cfa_level_iii",
    "FRM_Part_1": "frm_part_1",
    "FRM_Part_2": "frm_part_2",
}

VERIFICATION_KEYS = (
    "method",
    "template",
    "seed",
    "recomputed",
    "answer_matches_recomputation",
    "flawed_answer_concrete",
)
METADATA_KEYS = (
    "topic",
    "subtopic",
    "difficulty",
    "question_type",
    "pitfalls_addressed",
    "source",
    "seed",
    "generator",
    "generator_version",
)


def normalize(rec: dict) -> dict:
    """Force every record onto one schema so Arrow type inference is stable."""
    ver = rec.get("verification") or {}
    meta = rec.get("metadata") or {}
    pref = rec.get("preference_pair")
    return {
        "id": rec["id"],
        "program": rec["program"],
        "topic": rec.get("topic"),
        "subtopic": rec.get("subtopic"),
        "difficulty": rec.get("difficulty"),
        "question_type": rec.get("question_type"),
        "question": rec["question"],
        "answer": str(rec["answer"]),
        "distractors": [str(d) for d in (rec.get("distractors") or [])],
        "reasoning_trace": rec.get("reasoning_trace"),
        "verified": bool(rec.get("verified", False)),
        "verification": {
            "method": ver.get("method"),
            "template": ver.get("template"),
            "seed": ver.get("seed"),
            "recomputed": ver.get("recomputed"),
            "answer_matches_recomputation": ver.get("answer_matches_recomputation"),
            "flawed_answer_concrete": ver.get("flawed_answer_concrete"),
        },
        "metadata": {
            "topic": meta.get("topic"),
            "subtopic": meta.get("subtopic"),
            "difficulty": meta.get("difficulty"),
            "question_type": meta.get("question_type"),
            "pitfalls_addressed": [str(p) for p in (meta.get("pitfalls_addressed") or [])],
            "source": meta.get("source"),
            "seed": meta.get("seed"),
            "generator": meta.get("generator"),
            "generator_version": meta.get("generator_version"),
        },
        "preference_pair": (
            {
                "chosen": pref.get("chosen"),
                "rejected": pref.get("rejected"),
                "pitfall": pref.get("pitfall"),
            }
            if pref
            else None
        ),
    }


def load_program(program: str) -> list[dict]:
    records = []
    paths = sorted(glob.glob(os.path.join("shards", program, "*.jsonl")))
    if not paths:
        sys.exit(f"no shards found for {program} — run from the repository root")
    for path in paths:
        with open(path) as fh:
            for line in fh:
                records.append(normalize(json.loads(line)))
    return records


def main() -> None:
    splits = {}
    all_records = []
    for program, split in PROGRAMS.items():
        records = load_program(program)
        all_records.extend(records)
        splits[split] = Dataset.from_list(records)
        print(f"{split}: {len(records)} records")

    total = sum(len(d) for d in splits.values())
    assert total == 71000, f"expected 71000 records, got {total}"

    pref_rows = [
        {
            "id": r["id"],
            "program": r["program"],
            "topic": r["topic"],
            "subtopic": r["subtopic"],
            "difficulty": r["difficulty"],
            "question_type": r["question_type"],
            "prompt": r["question"],
            "chosen": r["preference_pair"]["chosen"],
            "rejected": r["preference_pair"]["rejected"],
            "pitfall": r["preference_pair"]["pitfall"],
            "answer": r["answer"],
        }
        for r in all_records
        if r["preference_pair"] is not None
    ]
    print(f"preference_pairs: {len(pref_rows)} rows")

    ds = DatasetDict(splits)
    ds.push_to_hub(REPO_ID, config_name="default", private=True)

    prefs = DatasetDict({"train": Dataset.from_list(pref_rows)})
    prefs.push_to_hub(REPO_ID, config_name="preference_pairs")

    print(f"pushed to https://huggingface.co/datasets/{REPO_ID} (private)")


if __name__ == "__main__":
    main()
