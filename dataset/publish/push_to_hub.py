"""Publish the Cosimo v2 corpus to the Hugging Face Hub.

Two configs:

  default      one split per program (cfa_level_i ... frm_part_2), all supervised
               record types (exam, analysis, abstention, agentic, implementation)
  preference   standalone DPO/ORPO pairs from shards/preference/, in the
               cosimopref_ id namespace -- disjoint from every supervised id by
               construction

Schema decisions, and why:

  * Heterogeneous nested fields (`conversation`, `tool_schemas`, `verification`,
    `metadata`) are JSON-encoded strings. Their key sets differ per record type,
    and letting Arrow infer a union struct across 240+ shards is how a dataset
    ends up unloadable on the split that happens to lack a key.
  * The embedded `preference_pair` on exam records is **dropped**. Its `chosen`
    side is byte-identical to the record's own reasoning trace -- the exact
    collision that made the first DPO run a zero-gradient no-op. The published
    preference data is the standalone config, whose ids no SFT row can share.

The verification suite is a precondition, not a suggestion: the script runs
verify_all and refuses to publish on a non-zero exit. --skip-gates exists for
debugging the publish path itself and prints a warning you should not ignore.

Run from dataset/:
    python3 publish/push_to_hub.py --repo-id <org>/<name> --dry-run
    python3 publish/push_to_hub.py --repo-id <org>/<name>            # private
    python3 publish/push_to_hub.py --repo-id <org>/<name> --public

Requires: datasets, huggingface_hub; an authenticated login (hf auth login,
or HF_TOKEN in the environment).
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SHARDS = os.path.join(BASE, "shards")
CARD_TEMPLATE = os.path.join(BASE, "publish", "dataset_card.md")

PROGRAMS = {
    "CFA_Level_I": "cfa_level_i",
    "CFA_Level_II": "cfa_level_ii",
    "CFA_Level_III": "cfa_level_iii",
    "FRM_Part_1": "frm_part_1",
    "FRM_Part_2": "frm_part_2",
}
PREFERENCE_DIR = "preference"

# The default config's schema. Every record is normalised onto exactly these
# columns; absent fields become empty strings / lists so Arrow sees one type.
DEFAULT_COLUMNS = (
    "id", "record_type", "program", "topic", "subtopic", "difficulty",
    "question_type", "question", "answer", "distractors", "reasoning_trace",
    "code", "test_code", "conversation", "tool_schemas",
    "verified", "verification", "metadata",
)

PREFERENCE_COLUMNS = (
    "id", "record_type", "program", "topic", "subtopic", "difficulty",
    "question_type", "mode", "prompt", "chosen", "rejected", "pitfall",
    "contains_intentional_fabrication", "verified", "verification", "metadata",
)

_JSON_FIELDS = {"conversation", "tool_schemas", "verification", "metadata"}


def _text(value):
    return "" if value is None else str(value)


def normalize_default(rec):
    row = {}
    for col in DEFAULT_COLUMNS:
        value = rec.get(col)
        if col == "record_type" and not value:
            value = "exam"
        if col in _JSON_FIELDS:
            row[col] = json.dumps(value, ensure_ascii=False) if value else ""
        elif col == "distractors":
            row[col] = [str(d) for d in (value or [])]
        elif col == "verified":
            row[col] = bool(value)
        else:
            row[col] = _text(value)
    # preference_pair deliberately not carried -- see module docstring.
    return row


def normalize_preference(rec):
    row = {}
    for col in PREFERENCE_COLUMNS:
        value = rec.get(col)
        if col in _JSON_FIELDS:
            row[col] = json.dumps(value, ensure_ascii=False) if value else ""
        elif col in ("verified", "contains_intentional_fabrication"):
            row[col] = bool(value)
        else:
            row[col] = _text(value)
    return row


def load_dir(program):
    records = []
    paths = sorted(glob.glob(os.path.join(SHARDS, program, "*.jsonl")))
    if not paths:
        raise SystemExit(f"no shards for {program} under {SHARDS} -- generate first")
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def run_gates():
    """verify_all is the publish precondition. Non-zero exit blocks the push."""
    print("running verification gates (verify_all)...")
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE, "verification", "verify_all.py")],
        cwd=BASE,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "verify_all FAILED -- refusing to publish a corpus that does not "
            "pass its own gates. Fix the failures or use --skip-gates to debug "
            "the publish path only."
        )


def corpus_stats(splits, pref_rows):
    """Measured numbers for the dataset card. Never hand-written."""
    by_type = collections.Counter()
    subtopics = collections.defaultdict(set)
    generators = set()
    for rows in splits.values():
        for row in rows:
            by_type[row["record_type"]] += 1
            subtopics[row["program"]].add(row["subtopic"])
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            generators.add(meta.get("generator"))
    total = sum(by_type.values())

    # Taxonomy coverage, measured against the declared taxonomy.
    declared = {}
    try:
        with open(os.path.join(BASE, "taxonomy", "taxonomy.json")) as f:
            tax = json.load(f)
        for prog, spec in tax.get("programs", {}).items():
            names = set()
            for topic in spec.get("topics", []):
                for s in topic.get("subtopics", []):
                    names.add(s if isinstance(s, str) else s.get("name"))
            declared[prog] = names
    except Exception:
        pass
    covered = sum(
        len(declared.get(p, set()) & subtopics.get(p, set())) for p in declared
    )
    declared_total = sum(len(v) for v in declared.values())

    modes = collections.Counter(r["mode"] for r in pref_rows)
    return {
        "total": total,
        "by_type": dict(by_type.most_common()),
        "by_program": {s: len(r) for s, r in splits.items()},
        "preference_rows": len(pref_rows),
        "preference_modes": dict(sorted(modes.items())),
        "generators": len(generators),
        "coverage": f"{covered}/{declared_total}",
        "coverage_pct": round(100 * covered / declared_total, 1) if declared_total else 0,
    }


def _cite_key(repo_id):
    """A BibTeX-safe key from the repo id: no slashes, dots or hyphens."""
    name = repo_id.split("/")[-1]
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def render_card(stats, repo_id):
    import datetime

    with open(CARD_TEMPLATE) as f:
        card = f.read()
    type_rows = "\n".join(
        f"| `{t}` | {n:,} | {100 * n / stats['total']:.1f}% |"
        for t, n in stats["by_type"].items()
    )
    prog_rows = "\n".join(
        f"| {s} | {n:,} |" for s, n in stats["by_program"].items()
    )
    mode_rows = "\n".join(
        f"| `{m}` | {n:,} |" for m, n in stats["preference_modes"].items()
    )
    return (
        card.replace("{{REPO_ID}}", repo_id)
        .replace("{{TOTAL}}", f"{stats['total']:,}")
        .replace("{{TYPE_TABLE}}", type_rows)
        .replace("{{PROGRAM_TABLE}}", prog_rows)
        .replace("{{PREF_ROWS}}", f"{stats['preference_rows']:,}")
        .replace("{{MODE_TABLE}}", mode_rows)
        .replace("{{GENERATORS}}", str(stats["generators"]))
        .replace("{{COVERAGE}}", stats["coverage"])
        .replace("{{COVERAGE_PCT}}", str(stats["coverage_pct"]))
        .replace("{{CITE_KEY}}", _cite_key(repo_id))
        .replace("{{YEAR}}", str(datetime.date.today().year))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True,
                        help="Hub dataset id, e.g. btech-software/cosimo-v2")
    parser.add_argument("--public", action="store_true",
                        help="create public (default: private; flip on the Hub "
                             "after review)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate everything, push nothing")
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    if args.skip_gates:
        print("WARNING: --skip-gates -- publishing without the verification "
              "suite. Debug use only.")
    else:
        run_gates()

    splits = {}
    for program, split in PROGRAMS.items():
        rows = [normalize_default(r) for r in load_dir(program)]
        splits[split] = rows
        print(f"  {split}: {len(rows):,} rows")

    pref_rows = [normalize_preference(r) for r in load_dir(PREFERENCE_DIR)]
    print(f"  preference: {len(pref_rows):,} rows")

    # Id-space invariant, enforced at the door regardless of what the gates did.
    supervised_ids = {r["id"] for rows in splits.values() for r in rows}
    overlap = supervised_ids & {r["id"] for r in pref_rows}
    if overlap:
        raise SystemExit(f"{len(overlap)} preference ids collide with supervised "
                         f"ids (e.g. {sorted(overlap)[:3]}); refusing to publish")

    stats = corpus_stats(splits, pref_rows)
    card = render_card(stats, args.repo_id)
    card_out = os.path.join(BASE, "publish", "dataset_card_generated.md")
    with open(card_out, "w") as f:
        f.write(card)
    print(f"\ncomposition: {stats['by_type']}")
    print(f"taxonomy coverage: {stats['coverage']} ({stats['coverage_pct']}%)")
    print(f"card -> {card_out}")

    if args.dry_run:
        print("\n--dry-run: nothing pushed")
        return

    from datasets import Dataset, DatasetDict

    ds = DatasetDict({s: Dataset.from_list(rows) for s, rows in splits.items()})
    ds.push_to_hub(args.repo_id, config_name="default", private=not args.public)
    prefs = DatasetDict({"train": Dataset.from_list(pref_rows)})
    prefs.push_to_hub(args.repo_id, config_name="preference")

    # The card: keep the YAML front matter push_to_hub generated (the viewer
    # needs it), replace the prose body with the measured card.
    from huggingface_hub import HfApi

    api = HfApi()
    front = ""
    try:
        readme = api.hf_hub_download(args.repo_id, "README.md", repo_type="dataset")
        text = open(readme).read()
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                front = text[: end + 3] + "\n\n"
    except Exception:
        pass
    api.upload_file(
        path_or_fileobj=(front + card).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"\npushed to https://huggingface.co/datasets/{args.repo_id} "
          f"({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
