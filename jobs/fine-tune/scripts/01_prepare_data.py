#!/usr/bin/env python3
"""Download, normalise, split and render the Cosimo corpus into data/processed/.

Reads every source in `dataset.hub_id` + `dataset.mix`, assigns deterministic
splits, renders every training example through the harness chat template, writes
the JSONL files the training and evaluation scripts consume, and runs a
validation gate that fails loudly on leakage, on a rendering fault, and on a
configuration that did not do what it said it would do.

The gate checks the *output* against the *configuration*, not only against
itself: a holdout family that matched no record, an empty split, a generator
label that disagrees with its verification template, or a preference pair whose
rejected side carries a format cue are all failures, not warnings.

Three properties of the mixed corpus shape the code below:

* **Five record types, one grading contract.** `FINAL ANSWER:` and the exam
  protocol belong to `exam` rows only. The other four types are why v2 exists;
  rendering them in exam shape would rebuild the style collapse v1 produced.
* **Only exam rows are gradeable.** `grading.grade_cosimo` reads a final-answer
  value, which a 900-token analysis does not have, so the two evaluation slices
  are exam-only and non-exam records never reach a `test` split.
* **Preference rows may or may not share ids with supervised rows.** v2's do
  not (the `cosimopref_` namespace); v1's do, which is what
  `data.preference_holdout_frac` exists to work around.

Example:
    ./scripts/01_prepare_data.py --force
    ./scripts/01_prepare_data.py --limit 500 --force   # fast smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from cosimo_ft import chat, data_schema, grading, runlog, splits  # noqa: E402
from cosimo_ft import config as config_mod  # noqa: E402

LOGGER = logging.getLogger("prepare_data")

# Written into paths.processed_dir. The names are a contract: benchmarks.py
# reads the two eval_* files, the training scripts read the rest.
SFT_FILES = {splits.TRAIN: "sft_train.jsonl", splits.VAL: "sft_val.jsonl"}
EVAL_FILES = {
    splits.TEST: "eval_cosimo_test.jsonl",
    splits.UNSEEN_STEMS: "eval_cosimo_unseen_stems.jsonl",
}
PREF_FILES = {splits.TRAIN: "pref_train.jsonl", splits.VAL: "pref_val.jsonl"}

# Namespaces the preference-holdout draw so it cannot correlate with the
# prompt.variation_rate draw, which thresholds the unsalted digest of the same id.
PREFERENCE_HOLDOUT_SALT = "preference-holdout"
DATA_FILES = (
    tuple(SFT_FILES.values()) + tuple(EVAL_FILES.values()) + tuple(PREF_FILES.values())
)
MANIFEST_FILE = "split_manifest.json"
# Provenance written beside the data: which config produced it, in which
# environment. data/processed/ is gitignored, so without these a --set override
# is unrecoverable once the shell closes.
CONFIG_FILE = "resolved_config.yaml"
ENV_FILE = "env.json"
OUTPUT_FILES = DATA_FILES + (MANIFEST_FILE, CONFIG_FILE, ENV_FILE)

# A validation failure usually affects thousands of rows; report enough to
# diagnose it without printing a novel.
MAX_REPORTED_PROBLEMS = 20

# MCQ options are embedded in the question text as "A. <value>" lines.
OPTION_LINE_RE = re.compile(r"^\s*\(?([A-D])\)?[.):]\s*(.+?)\s*$", re.MULTILINE)


# --------------------------------------------------------------------------
# loading (heavy imports stay inside the functions)
# --------------------------------------------------------------------------


def load_tokenizer(tokenizer_id: str, revision: str | None = None) -> Any:
    """Load the tokenizer whose chat template every example is rendered with."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)


def resolve_dataset_sha(hub_id: str, revision: str | None) -> str | None:
    """Resolve a branch name to the Hub commit sha that was actually read.

    ``revision: main`` is a moving target: two runs months apart would otherwise
    record identical provenance while reading different data.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(hub_id, revision=revision).sha
    except Exception as exc:
        LOGGER.warning(
            "could not resolve %s@%s to a commit sha (%r); provenance will record "
            "the requested revision only",
            hub_id,
            revision,
            exc,
        )
        return None


def rows_fingerprint(rows: list[dict]) -> dict:
    """Content fingerprint of the row set actually used.

    A digest of the ids, computed *after* any subsampling, so it describes the
    rows that were prepared rather than the local ``datasets`` cache state.
    """
    digest = hashlib.sha256()
    for row_id in sorted(str(row.get("id", "")) for row in rows):
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\0")
    return {"n": len(rows), "ids_sha256": digest.hexdigest()[:16]}


def dataset_sources(cfg: dict) -> list[dict]:
    """The ordered list of Hub corpora to merge.

    ``dataset.hub_id`` is the primary source and stays a plain string so the
    manifest, the docs and ``--set dataset.hub_id=...`` keep working; each entry
    of ``dataset.mix`` adds another corpus. Order is priority order: the
    cross-source question dedupe keeps the first occurrence.
    """
    primary = {
        "hub_id": config_mod.get(cfg, "dataset.hub_id"),
        "revision": config_mod.get(cfg, "dataset.revision"),
        "preference_config": config_mod.get(cfg, "dataset.preference_config"),
        "max_share": None,
    }
    if not primary["hub_id"]:
        raise ValueError("dataset.hub_id is not set")
    sources = [primary]
    for entry in config_mod.get(cfg, "dataset.mix") or []:
        if not isinstance(entry, dict) or not entry.get("hub_id"):
            raise ValueError(
                f"every dataset.mix entry needs a hub_id, got {entry!r}"
            )
        sources.append(
            {
                "hub_id": entry["hub_id"],
                "revision": entry.get("revision", "main"),
                "preference_config": entry.get("preference_config"),
                "max_share": entry.get("max_share"),
            }
        )
    seen = [source["hub_id"] for source in sources]
    if len(set(seen)) != len(seen):
        raise ValueError(f"a corpus is listed twice in dataset.hub_id/mix: {seen}")
    return sources


def load_source_rows(
    source: dict,
    *,
    limit: int | None = None,
    seed: int = 3407,
) -> tuple[list[dict], list[dict], dict]:
    """Load one corpus's supervised and preference rows, plus row fingerprints.

    ``limit`` takes a *seeded, order-preserving sample* of each split rather than
    a head slice: the Hub stores contiguous ~1000-row blocks per generator, so
    ``[:N]`` would yield a single generator per program and a smoke run that
    exercises almost none of the stratification, the record types or the holdout
    families.

    Preference rows are limited the same way when their ids are disjoint from
    the supervised rows (v2), and by joining on the sampled ids when they are
    shared (v1) so the two configs stay consistent.
    """
    from datasets import load_dataset

    hub_id = source["hub_id"]
    revision = source["revision"]
    fingerprints: dict[str, dict] = {}

    default_rows: list[dict] = []
    loaded = load_dataset(hub_id, "default", revision=revision)
    for split_name in sorted(loaded):
        dataset = loaded[split_name]
        picked = subsample(list(range(len(dataset))), limit, seed)
        if len(picked) < len(dataset):
            dataset = dataset.select(picked)
        rows = [dict(row) for row in dataset]
        for row in rows:
            row["_source"] = hub_id
        fingerprints[f"{hub_id}/default/{split_name}"] = rows_fingerprint(rows)
        default_rows.extend(rows)
        LOGGER.info("loaded %s default/%s: %d rows", hub_id, split_name, len(rows))

    pref_config = source.get("preference_config")
    pref_rows: list[dict] = []
    if not pref_config:
        LOGGER.info("%s: no preference_config, its pairs are not used", hub_id)
        return default_rows, pref_rows, fingerprints

    sampled_ids = {str(row.get("id", "")) for row in default_rows}
    loaded = load_dataset(hub_id, pref_config, revision=revision)
    for split_name in sorted(loaded):
        rows = [dict(row) for row in loaded[split_name]]
        if limit is not None:
            joined = [row for row in rows if str(row.get("id", "")) in sampled_ids]
            # Disjoint id spaces (v2) join to nothing, so the intersection is not
            # a usable smoke sample; fall back to a seeded sample of the config.
            rows = joined if joined else subsample(rows, limit, seed)
        for row in rows:
            row["_source"] = hub_id
        fingerprints[f"{hub_id}/{pref_config}/{split_name}"] = rows_fingerprint(rows)
        pref_rows.extend(rows)
        LOGGER.info(
            "loaded %s %s/%s: %d rows", hub_id, pref_config, split_name, len(rows)
        )

    return default_rows, pref_rows, fingerprints


def load_hub_rows(
    sources: list[dict],
    *,
    limit: int | None = None,
    seed: int = 3407,
) -> tuple[list[dict], list[dict], dict]:
    """Load every source in order and concatenate. Capping happens later.

    ``max_share`` is applied after normalisation, in :func:`apply_source_caps`,
    so the share is a share of the rows that actually survive verification,
    deduplication and rendering rather than of the rows the Hub happened to ship.
    """
    default_rows: list[dict] = []
    pref_rows: list[dict] = []
    fingerprints: dict[str, dict] = {}
    for source in sources:
        rows, prefs, prints = load_source_rows(source, limit=limit, seed=seed)
        default_rows.extend(rows)
        pref_rows.extend(prefs)
        fingerprints.update(prints)
    return default_rows, pref_rows, fingerprints


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def check_outputs(out_dir: Path, force: bool) -> None:
    """Refuse to clobber a previous preparation unless ``--force`` was given."""
    if force:
        return
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing:
        raise SystemExit(
            f"{out_dir} already contains {', '.join(existing)}; "
            "pass --force to overwrite"
        )


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write rows as JSONL, replacing any previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return runlog.append_jsonl(path, rows)


def token_lengths(tokenizer: Any, texts: Iterable[str]) -> list[int]:
    """Token counts of already-rendered strings (no extra special tokens)."""
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]


def percentiles(values: list[int], max_seq_length: int) -> dict:
    """Nearest-rank p50/p95/p99/max plus how many rows exceed the budget."""
    if not values:
        return {"n": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "over_max_seq": 0}
    ordered = sorted(values)

    def at(quantile: float) -> int:
        rank = max(1, math.ceil(quantile * len(ordered)))
        return ordered[rank - 1]

    return {
        "n": len(ordered),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
        "over_max_seq": sum(1 for value in ordered if value > max_seq_length),
    }


def subsample(records: list, limit: int | None, seed: int) -> list:
    """Deterministically keep ``limit`` records, preserving the input order."""
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        return []
    picked = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[index] for index in picked]


def _blank(value: Any) -> bool:
    return not str(value or "").strip()


def is_blank_record(record: data_schema.CosimoRecord) -> bool:
    """True when a record cannot produce a usable supervised target.

    A record with no question, no answer or no reasoning trace still *renders*
    (the system block and the turn markers are always there), so the rendered
    string is never empty — it is just degenerate, e.g. a completion of
    ``"\\n\\nFINAL ANSWER:"``. Those teach the model to emit a bare tag, so they
    are rejected on their source fields, before rendering.

    Which fields are required depends on the record type: only ``exam`` has a
    reasoning trace, ``implementation`` carries its substance in ``code``, and
    an ``agentic`` record's target is the conversation rather than ``answer``.
    Applying the exam rule to all five would reject 78% of the v2 corpus in
    silence.
    """
    if _blank(record.question) and record.record_type != data_schema.AGENTIC:
        return True
    if record.record_type == data_schema.AGENTIC:
        # The supervised span starts at the first assistant turn, so a
        # conversation without one renders a prompt and nothing to learn.
        return not any(
            message.get("role") == "assistant" for message in record.conversation
        )
    if record.record_type == data_schema.IMPLEMENTATION:
        return _blank(record.code) and _blank(record.answer)
    if record.record_type == data_schema.EXAM:
        return _blank(record.answer) or _blank(record.reasoning_trace)
    if record.record_type == data_schema.PREFERENCE:
        # A standalone pair has no gold value; its chosen side is the content,
        # and normalize_standalone_pref_row carries that as the trace.
        return _blank(record.reasoning_trace)
    return _blank(record.answer)


def apply_source_caps(
    records: list[data_schema.CosimoRecord],
    source_by_id: dict[str, str],
    sources: list[dict],
    seed: int,
    holdout_families: set[str],
) -> tuple[list[data_schema.CosimoRecord], dict[str, int]]:
    """Enforce each source's ``max_share`` of the merged **trainable** pool.

    A capped source keeps ``share/(1 - share)`` times the size of the uncapped
    remainder, so its share of the final trainable pool is the configured one.
    The subsample is seeded and order-preserving, and because the pool is sorted
    by id it spreads across generators rather than truncating a contiguous block.

    **Held-out records are exempt from the cap**, and that exemption is the
    point of measuring against the trainable pool rather than all rows. A
    held-out family never enters training — it is the `unseen_stems` measuring
    instrument. Subsampling it would shrink the evaluation slice, and therefore
    widen its confidence interval, as a side effect of retuning the training
    mix. The knob that balances what the model learns must not quietly degrade
    what measures it.
    """
    capped = {
        str(source["hub_id"]): float(source["max_share"])
        for source in sources
        if source.get("max_share") is not None
    }
    if not capped:
        return records, {}
    for hub_id, share in capped.items():
        if not 0.0 < share < 1.0:
            raise ValueError(
                f"dataset.mix max_share for {hub_id!r} must be in (0, 1), "
                f"got {share!r}"
            )
    if sum(capped.values()) >= 1.0:
        raise ValueError(
            f"dataset.mix max_share values sum to {sum(capped.values())}, leaving "
            "no room for the uncapped sources"
        )

    def held_out(record: data_schema.CosimoRecord) -> bool:
        return data_schema.stem_family(record.generator) in holdout_families

    trainable = [r for r in records if not held_out(r)]
    uncapped = [r for r in trainable if source_by_id[r.id] not in capped]
    budget = len(uncapped) / (1.0 - sum(capped.values()))
    surviving = {r.id for r in records if held_out(r)}
    surviving |= {r.id for r in uncapped}
    dropped: dict[str, int] = {}
    for hub_id, share in capped.items():
        pool = [r for r in trainable if source_by_id[r.id] == hub_id]
        keep = subsample(pool, int(round(share * budget)), seed)
        surviving |= {r.id for r in keep}
        if len(keep) < len(pool):
            dropped[hub_id] = len(pool) - len(keep)
            LOGGER.info(
                "max_share=%s for %s: kept %d of %d trainable records "
                "(held-out records are exempt)",
                share,
                hub_id,
                len(keep),
                len(pool),
            )
    return [r for r in records if r.id in surviving], dropped


def _counts_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def measure_identity(rows: list[dict], full_identity: str, short_identity: str) -> dict:
    """Count which identity each written prompt actually carries.

    Measured by substring on the rendered prompt, not re-derived from the id
    hash: a re-derivation would report the intended fraction even if rendering
    ignored it entirely.

    Matched against the identity blocks alone rather than the composed system
    message, because only exam rows carry the exam protocol. The full block is
    tested first: ``prompt.identity_short`` is the opening line of
    ``prompt.identity``, so a short-first test would count every row as short.
    The exam protocol is checked separately, in :func:`validate`.
    """
    full = short = 0
    for row in rows:
        if full_identity in row["prompt"]:
            full += 1
        elif short_identity in row["prompt"]:
            short += 1
    return {
        "n": len(rows),
        "short": short,
        "full": full,
        "neither": len(rows) - short - full,
        "fraction": round(short / len(rows), 4) if rows else 0.0,
    }


# --------------------------------------------------------------------------
# MCQ preference format cue (see the manifest's `mcq_preference` block)
# --------------------------------------------------------------------------


def mcq_options(question: str) -> list[tuple[str, str]]:
    """The ``A. <value>`` option lines embedded in an MCQ question."""
    return OPTION_LINE_RE.findall(question or "")


def _values_match(left: str, right: str) -> bool:
    if str(left).strip() == str(right).strip():
        return True
    a, b = grading.parse_number(left), grading.parse_number(right)
    return a is not None and b is not None and grading.numeric_close(a, b)


def resolve_mcq_letter(
    question: str, chosen_answer: str, rejected_answer: str
) -> tuple[str | None, str]:
    """Find the option letter for an unprefixed rejected MCQ answer.

    In the source data ``chosen.answer`` is letter-prefixed on every MCQ
    preference pair and ``rejected.answer`` on none, so after
    ``build_completion`` the two sides differ by a one-token surface feature.
    DPO can maximise its objective on that cue instead of on the reasoning,
    which directly attacks the pitfall-rate the preference stage exists to move.

    Returns ``(letter, outcome)``; the chosen letter is excluded so a duplicated
    option value (both ``A. 20`` and ``B. 20`` occur) cannot resolve to it.
    """
    if grading.mcq_letter(rejected_answer):
        return None, "already_prefixed"
    options = mcq_options(question)
    if not options:
        return None, "no_options"
    chosen_letter = grading.mcq_letter(chosen_answer)
    matches = [
        letter
        for letter, value in options
        if letter != chosen_letter and _values_match(value, rejected_answer)
    ]
    if not matches:
        return None, "no_match"
    return matches[0], "ambiguous" if len(matches) > 1 else "resolved"


def normalize_mcq_pair(
    record: data_schema.CosimoRecord,
) -> tuple[data_schema.CosimoRecord, str]:
    """Put the option letter on the rejected side so both sides look alike."""
    if record.question_type != "MCQ" or not record.rejected or not record.chosen:
        return record, "not_mcq"
    rejected_answer = str(record.rejected.get("answer", ""))
    letter, outcome = resolve_mcq_letter(
        record.question, str(record.chosen.get("answer", "")), rejected_answer
    )
    if letter is None:
        return record, outcome
    rejected = dict(record.rejected)
    rejected["answer"] = f"{letter}. {rejected_answer.strip()}"
    return replace(record, rejected=rejected), outcome


def written_letter_prefix_counts(rows: list[dict], tag: str) -> dict:
    """Letter-prefix asymmetry of the rows as they were actually written."""
    mcq = [row for row in rows if row.get("question_type") == "MCQ"]
    counts = {"n_mcq": len(mcq)}
    for side in ("chosen", "rejected"):
        counts[side] = sum(
            1
            for row in mcq
            if grading.mcq_letter(grading.extract_final_answer(row[side], tag))
        )
    return counts


# --------------------------------------------------------------------------
# validation gate
# --------------------------------------------------------------------------


def validate(
    files: dict[str, list[dict]],
    *,
    holdout_families: set[str],
    families_present: set[str],
    pool_size: int,
    generator_conflicts: list[dict],
    full_system: str,
    short_system: str,
    exam_protocol: str,
    tag: str,
    eos_token: str = "",
) -> None:
    """Fail loudly. Checks the output against the configuration that produced it.

    Beyond the internal-consistency checks (no id in two splits, no held-out
    family in a training split, ``text`` starts with ``prompt``,
    ``chosen != rejected``), this verifies that the run did what it was asked to
    do: every configured holdout family matched real records, no split came out
    empty, every generator label agrees with its verification template, every
    prompt carries a recognised identity, and the MCQ letter-prefix cue is gone.
    """
    problems: list[str] = []

    # --- the configuration was actually honoured -------------------------
    for family in sorted(holdout_families):
        if any(family.startswith(prefix) for prefix in data_schema.STEM_PREFIXES):
            problems.append(
                f"data.holdout_families entry {family!r} carries a wrapper prefix; "
                f"hold out the family {data_schema.stem_family(family)!r} instead, "
                "or the base stem stays in training"
            )
        elif family not in families_present:
            problems.append(
                f"data.holdout_families entry {family!r} matched no record; "
                "check the spelling against metadata.generator"
            )
    if holdout_families and not files.get(EVAL_FILES[splits.UNSEEN_STEMS]):
        problems.append(
            f"{len(holdout_families)} holdout families are configured but "
            f"{EVAL_FILES[splits.UNSEEN_STEMS]} is empty: the unseen-stem "
            "measurement would not exist and those records would be in training"
        )

    for name in DATA_FILES:
        if not files.get(name):
            problems.append(f"{name} is empty")
    if pool_size:
        for split_name, name in (
            (splits.TEST, EVAL_FILES[splits.TEST]),
            (splits.VAL, SFT_FILES[splits.VAL]),
        ):
            if not files.get(name):
                problems.append(
                    f"the {split_name!r} split is empty for a pool of {pool_size} "
                    f"records: raise data.{split_name}_frac or drop --limit"
                )

    for conflict in generator_conflicts:
        problems.append(
            f"id {conflict['id']!r}: metadata.generator "
            f"{conflict['metadata_generator']!r} disagrees with "
            f"verification.template {conflict['verification_template']!r}, and one "
            "of them is a held-out family — the holdout cannot be trusted"
        )

    # --- no id in two splits --------------------------------------------
    owner: dict[str, str] = {}
    for name in tuple(SFT_FILES.values()) + tuple(EVAL_FILES.values()):
        for row in files.get(name, []):
            previous = owner.get(row["id"])
            if previous is not None:
                problems.append(f"id {row['id']!r} appears in {previous} and {name}")
            else:
                owner[row["id"]] = name

    # A preference row must not share an id with the SFT rows of the same split.
    # It is not a leak in the evaluation sense -- both are training data -- but it
    # is the failure that made the first DPO run a no-op: SFT trains on the pair's
    # `chosen` trace, the implicit reward margin saturates, and the preference
    # stage spends hours at exactly zero gradient. Cheaper to fail here.
    for split_name, pref_name in PREF_FILES.items():
        sft_name = SFT_FILES.get(split_name)
        if not sft_name:
            continue
        shared = {str(row["id"]) for row in files.get(sft_name, [])} & {
            str(row["id"]) for row in files.get(pref_name, [])
        }
        if shared:
            problems.append(
                f"{pref_name}: {len(shared)} id(s) are also in {sft_name} "
                f"(e.g. {sorted(shared)[:3]}). The preference stage would train "
                "against traces SFT already fit, which yields no gradient. Raise "
                "data.preference_holdout_frac."
            )

    # A preference id legitimately repeats its SFT id across *different* splits,
    # so it cannot join the owner map; what must never happen is a preference row
    # for an evaluated id.
    evaluated = {
        row["id"] for name in EVAL_FILES.values() for row in files.get(name, [])
    }
    for name in PREF_FILES.values():
        rows = files.get(name, [])
        for row in rows:
            if row["id"] in evaluated:
                problems.append(
                    f"{name}: id {row['id']!r} is also in an evaluation split"
                )
        duplicates = [
            row_id
            for row_id, count in Counter(row["id"] for row in rows).items()
            if count > 1
        ]
        for row_id in duplicates:
            problems.append(f"{name}: id {row_id!r} appears more than once")

    # --- holdout families stay where they belong -------------------------
    for name in tuple(SFT_FILES.values()) + (EVAL_FILES[splits.TEST],):
        for row in files.get(name, []):
            if row["stem_family"] in holdout_families:
                problems.append(
                    f"{name}: id {row['id']!r} has held-out stem family "
                    f"{row['stem_family']!r}"
                )
    for row in files.get(EVAL_FILES[splits.UNSEEN_STEMS], []):
        if row["stem_family"] not in holdout_families:
            problems.append(
                f"{EVAL_FILES[splits.UNSEEN_STEMS]}: id {row['id']!r} has "
                f"stem family {row['stem_family']!r}, which is not held out"
            )

    # --- the graded suites are exam-only ---------------------------------
    # grading.grade_cosimo reads the value after the last FINAL ANSWER: line and
    # compares it numerically or by option letter. An analysis or abstention
    # record has no such value, so it would be scored wrong on every model and
    # drag both suites' accuracy toward zero while looking like a model result.
    for name in EVAL_FILES.values():
        non_exam = Counter(
            str(row.get("record_type"))
            for row in files.get(name, [])
            if row.get("record_type") != data_schema.EXAM
        )
        if non_exam:
            problems.append(
                f"{name}: {sum(non_exam.values())} non-exam rows ({dict(non_exam)}); "
                "the graded suites cannot score a record with no final-answer value"
            )

    # --- rendering ------------------------------------------------------
    for name in SFT_FILES.values():
        for row in files.get(name, []):
            if not row["text"].startswith(row["prompt"]):
                problems.append(
                    f"{name}: id {row['id']!r} text does not start with prompt"
                )
            if _blank(row["prompt"]) or _blank(row["completion"]):
                problems.append(f"{name}: id {row['id']!r} rendered an empty example")

    # TRL's DPO tokenizer appends EOS unconditionally while ORPO appends it only
    # when absent, so a trailing EOS here would train the two stages on different
    # targets for identical pairs. data_schema.to_pref_row strips it; this proves it.
    eos = eos_token
    for name in PREF_FILES.values():
        for row in files.get(name, []):
            if row["chosen"] == row["rejected"]:
                problems.append(f"{name}: id {row['id']!r} has chosen == rejected")
            if eos and (row["chosen"].endswith(eos) or row["rejected"].endswith(eos)):
                problems.append(
                    f"{name}: id {row['id']!r} preference text still ends with "
                    f"{eos!r}; DPO would train on a doubled EOS"
                )
            if (
                _blank(row["prompt"])
                or _blank(row["chosen"])
                or _blank(row["rejected"])
            ):
                problems.append(f"{name}: id {row['id']!r} rendered an empty example")

    # --- identity actually reached the rendered prompt -------------------
    for name in tuple(SFT_FILES.values()) + tuple(PREF_FILES.values()):
        identity = measure_identity(files.get(name, []), full_system, short_system)
        if identity["neither"]:
            problems.append(
                f"{name}: {identity['neither']} rows carry neither the full nor "
                "the short identity in their rendered prompt"
            )

    # --- the grading contract stays on exam rows -------------------------
    # This is the failure v2 exists to fix, and it is invisible downstream: a
    # model trained to answer an open-ended hedging question with `Step 1.` and
    # a FINAL ANSWER line still scores well on every exam suite. The contract
    # must be in the system block of exam rows and in the system block of
    # nothing else, and it must be the last line of an exam target and appear in
    # no other target.
    for name in tuple(SFT_FILES.values()) + tuple(PREF_FILES.values()):
        wrong_protocol: Counter = Counter()
        wrong_tag: Counter = Counter()
        for row in files.get(name, []):
            is_exam = row.get("record_type") == data_schema.EXAM
            if exam_protocol and (exam_protocol in row["prompt"]) is not is_exam:
                wrong_protocol[row.get("record_type", "?")] += 1
            targets = (
                [row["chosen"], row["rejected"]] if "chosen" in row else [row["completion"]]
            )
            if any((tag in target) is not is_exam for target in targets):
                wrong_tag[row.get("record_type", "?")] += 1
        if wrong_protocol:
            problems.append(
                f"{name}: {sum(wrong_protocol.values())} rows carry the exam "
                f"protocol in the wrong place ({dict(wrong_protocol)}). It belongs "
                "in the system block of exam rows only; on any other record type "
                "it instructs the exam shape into the answer being trained."
            )
        if wrong_tag:
            problems.append(
                f"{name}: {sum(wrong_tag.values())} supervised targets have the "
                f"{tag!r} contract on the wrong record type ({dict(wrong_tag)}). "
                "Exam targets must end with it; no other record type may contain "
                "it."
            )

    # --- the MCQ format cue is gone --------------------------------------
    for name in PREF_FILES.values():
        counts = written_letter_prefix_counts(files.get(name, []), tag)
        if counts["chosen"] != counts["rejected"]:
            problems.append(
                f"{name}: {counts['chosen']}/{counts['n_mcq']} chosen but "
                f"{counts['rejected']}/{counts['n_mcq']} rejected MCQ answers carry "
                "an option letter; the asymmetry is a format cue DPO can exploit"
            )

    if problems:
        shown = "\n  ".join(problems[:MAX_REPORTED_PROBLEMS])
        raise ValueError(
            f"data preparation failed validation ({len(problems)} problems):\n  "
            f"{shown}"
            + (
                f"\n  ... and {len(problems) - MAX_REPORTED_PROBLEMS} more"
                if len(problems) > MAX_REPORTED_PROBLEMS
                else ""
            )
        )


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------


def prepare(
    cfg: dict,
    *,
    sources: list[dict],
    default_rows: list[dict],
    pref_rows: list[dict],
    tokenizer: Any,
    tokenizer_id: str,
    dataset_info: dict,
    out_dir: Path,
    force: bool = False,
) -> dict:
    """Normalise, split, render, write and validate. Returns the manifest."""
    check_outputs(out_dir, force)

    # Before any rendering: the vendor template hardcodes a Microsoft/Phi identity
    # preamble ahead of every system message, which contradicts the identity being
    # trained. Training and evaluation apply the same override.
    template_text = chat.load_chat_template(cfg)
    template_applied = chat.apply_chat_template_override(tokenizer, cfg)
    template_hash = (
        hashlib.sha256(template_text.encode("utf-8")).hexdigest()[:12]
        if template_text
        else None
    )

    seed = int(config_mod.get(cfg, "seed", 3407))
    tag = config_mod.get(cfg, "prompt.final_answer_tag", grading.DEFAULT_TAG)
    variation_rate = float(config_mod.get(cfg, "prompt.variation_rate", 0.0) or 0.0)
    # The identity blocks, not the composed system messages: the exam protocol is
    # on exam rows only, so it cannot be part of what identifies the persona.
    full_system = chat.compose_system(cfg, short=False, exam=False)
    short_system = chat.compose_system(cfg, short=True, exam=False)
    exam_protocol = str(config_mod.get(cfg, "prompt.exam_protocol", "")).strip()
    max_seq_length = int(config_mod.get(cfg, "model.max_seq_length", 2048))
    val_frac = float(config_mod.get(cfg, "data.val_frac", 0.01))
    test_frac = float(config_mod.get(cfg, "data.test_frac", 0.01))
    max_train_records = config_mod.get(cfg, "data.max_train_records")
    preference_holdout_frac = float(
        config_mod.get(cfg, "data.preference_holdout_frac", 0.0) or 0.0
    )
    if not 0.0 <= preference_holdout_frac <= 1.0:
        raise ValueError(
            "data.preference_holdout_frac must be in [0, 1], got "
            f"{preference_holdout_frac!r}"
        )
    drop_unverified = bool(config_mod.get(cfg, "data.drop_unverified", True))
    holdout_families = {
        str(f) for f in (config_mod.get(cfg, "data.holdout_families") or [])
    }

    # 1. normalise, dropping unverified, duplicate and degenerate rows
    records: list[data_schema.CosimoRecord] = []
    seen_ids: set[str] = set()
    source_by_id: dict[str, str] = {}
    dropped: Counter = Counter()
    conflicts: list[dict] = []
    n_conflicts = 0
    # Cross-source question dedupe. 1,840 exam questions are byte-identical
    # across the two corpora under different ids, and the split is keyed by id,
    # so without this the same question can sit in the primary corpus's test
    # slice and a mixed-in corpus's training set. Only the FIRST source to
    # produce a question keeps it; duplicates *within* one corpus are left
    # alone, because they are a pre-existing property of that corpus rather than
    # something the mix introduced.
    questions_by_source: dict[str, set[str]] = defaultdict(set)
    for row in default_rows:
        source = str(row.get("_source") or "")
        if drop_unverified and not row.get("verified", False):
            # `is False` would keep a null/missing flag silently; count the two
            # cases apart so "all verified" cannot be confused with "no flag".
            dropped[
                "unverified_missing_flag"
                if row.get("verified") is None
                else "unverified"
            ] += 1
            continue
        record = data_schema.normalize_record(row)
        if record.id in seen_ids:
            dropped["duplicate_id"] += 1
            continue
        if is_blank_record(record):
            dropped["blank_content"] += 1
            continue
        question = record.question.strip()
        if question and any(
            question in seen
            for other, seen in questions_by_source.items()
            if other != source
        ):
            dropped["duplicate_question_across_sources"] += 1
            continue
        questions_by_source[source].add(question)
        # The generator decides the stem family, and the stem family decides the
        # holdout. A single mislabelled generator moves a held-out stem into
        # training, which is the one failure that cannot be detected downstream.
        metadata = data_schema.decode_mapping(row.get("metadata"))
        verification = data_schema.decode_mapping(row.get("verification"))
        metadata_generator = str(metadata.get("generator") or "")
        verification_template = str(verification.get("template") or "")
        if (
            metadata_generator
            and verification_template
            and metadata_generator != verification_template
        ):
            # Counted, not dropped: the record is still usable, but a conflict
            # that touches a holdout family makes the holdout unverifiable.
            conflict = {
                "id": record.id,
                "metadata_generator": metadata_generator,
                "verification_template": verification_template,
            }
            n_conflicts += 1
            if holdout_families & {
                data_schema.stem_family(metadata_generator),
                data_schema.stem_family(verification_template),
            }:
                conflicts.append(conflict)
        if record.record_type == data_schema.IMPLEMENTATION:
            # Neither counter drops a record: the code block carries the
            # substance either way. `reindented` is the published corpus's
            # dedent defect being undone (see normalize_python_block) and should
            # fall to 0 once v2 is regenerated from the fixed generator;
            # `unparseable` is a block the repair could not recover, which is
            # left out of the target. A rising count means a generator regressed.
            for field, block in (
                ("code", record.code),
                ("test_code", record.test_code),
            ):
                if not block.strip():
                    continue
                normalized = data_schema.normalize_python_block(block)
                if not normalized:
                    dropped[f"implementation_{field}_unparseable"] += 1
                elif normalized != block.strip():
                    dropped[f"implementation_{field}_reindented"] += 1
        seen_ids.add(record.id)
        source_by_id[record.id] = source
        records.append(record)
    records.sort(key=lambda record: record.id)

    # Each mixed-in corpus is capped as a share of the merged pool, after the
    # drops above so the share describes rows that survived rather than rows the
    # Hub shipped.
    records, capped = apply_source_caps(
        records, source_by_id, sources, seed, holdout_families
    )
    for hub_id, n in capped.items():
        dropped[f"over_max_share:{hub_id}"] += n
    LOGGER.info(
        "normalised %d records (%s) (dropped %s)",
        len(records),
        dict(sorted(Counter(r.record_type for r in records).items())),
        dict(sorted(dropped.items())),
    )

    # 2. assign splits. Exam and non-exam records are split separately because
    # only exam records are gradeable: grading.grade_cosimo reads a final-answer
    # value, which an analysis or an abstention response does not have. Non-exam
    # records therefore get test_frac=0 and never reach an evaluation slice. The
    # holdout family set is shared, so a held-out family's analysis and agentic
    # records are kept out of training too -- otherwise the family leaks back in
    # through its other record types.
    eval_rows = {record.id: data_schema.to_eval_row(record) for record in records}
    families_present = {row["stem_family"] for row in eval_rows.values()}
    assignment: dict[str, str] = {}
    for exam_side in (True, False):
        pool = [
            eval_rows[record.id]
            for record in records
            if data_schema.is_exam(record) is exam_side
        ]
        if not pool:
            continue
        assignment.update(
            splits.assign_splits(
                pool,
                val_frac=val_frac,
                test_frac=test_frac if exam_side else 0.0,
                seed=seed,
                holdout_families=holdout_families,
            )
        )
    pool_size = sum(
        1 for split_name in assignment.values() if split_name != splits.UNSEEN_STEMS
    )

    by_split: dict[str, list[data_schema.CosimoRecord]] = defaultdict(list)
    for record in records:
        by_split[assignment[record.id]].append(record)

    train_records = subsample(by_split[splits.TRAIN], max_train_records, seed)
    if len(train_records) != len(by_split[splits.TRAIN]):
        LOGGER.info(
            "max_train_records=%s: kept %d of %d train records",
            max_train_records,
            len(train_records),
            len(by_split[splits.TRAIN]),
        )

    # 3. preference rows FIRST, because which ids they claim decides which ids
    # SFT must not be trained on. Reserving a pair only helps if the policy has
    # never seen its `chosen` trace: a shared-id corpus's reasoning_trace *is*
    # that trace, so an id in both files gives DPO a pre-saturated margin and no
    # gradient. See data.preference_holdout_frac in configs/data.yaml.
    #
    # Two shapes reach this loop:
    #
    #   joined      (v1 `preference_pairs`) ids are shared with supervised rows,
    #               so the stem family comes from the joined `default` record,
    #               the row inherits that record's split, and
    #               preference_holdout_frac decides whether SFT gives it up.
    #   standalone  (v2 `preference`) ids are in the disjoint `cosimopref_`
    #               namespace and the rows carry their own generator, so there is
    #               nothing to join, nothing to reserve -- the overlap that
    #               caused the no-op is impossible -- and they need a split of
    #               their own.
    generator_by_id = {record.id: record.generator for record in records}
    standalone_rows = [
        row
        for row in pref_rows
        if str(row.get("id", "")) not in generator_by_id
        and str(row.get("chosen") or "").strip()
    ]
    standalone_ids = {str(row.get("id", "")) for row in standalone_rows}
    standalone_records = [
        data_schema.normalize_standalone_pref_row(row) for row in standalone_rows
    ]
    # Train/val only: a preference pair is never an evaluation item, and its
    # generators are its own, so a holdout family it shares with the supervised
    # corpus must still be excluded rather than trained on.
    standalone_assignment = (
        splits.assign_splits(
            [data_schema.to_eval_row(record) for record in standalone_records],
            val_frac=val_frac,
            test_frac=0.0,
            seed=seed,
            holdout_families=holdout_families,
        )
        if standalone_records
        else {}
    )
    standalone_by_id = {record.id: record for record in standalone_records}

    pref_files: dict[str, list[dict]] = {
        PREF_FILES[splits.TRAIN]: [],
        PREF_FILES[splits.VAL]: [],
    }
    mcq_outcomes: Counter = Counter()
    pref_modes: Counter = Counter()
    for row in sorted(pref_rows, key=lambda row: str(row.get("id", ""))):
        row_id = str(row.get("id", ""))
        standalone = row_id in standalone_ids
        if not standalone and row_id not in generator_by_id:
            dropped["pref_without_default_row"] += 1
            continue
        split_name = (standalone_assignment if standalone else assignment)[row_id]
        if split_name not in PREF_FILES:
            # Kept apart: the test count is the leak-prevention number, the
            # unseen count is a consequence of the holdout.
            dropped[
                "pref_in_test" if split_name == splits.TEST else "pref_in_unseen_stems"
            ] += 1
            continue
        if standalone:
            record = standalone_by_id[row_id]
        else:
            if (
                chat.id_fraction(row_id, PREFERENCE_HOLDOUT_SALT)
                >= preference_holdout_frac
            ):
                # Not reserved: this row stays in SFT and the preference stage
                # does not get it. At frac 0.0 every row takes this branch, which
                # is the original behaviour and the reason DPO could not learn.
                dropped["pref_kept_for_sft"] += 1
                continue
            record = data_schema.normalize_pref_row(
                {**row, "generator": generator_by_id[row_id]}
            )
        if not data_schema.has_preference(record):
            dropped["pref_unusable_pair"] += 1
            continue
        # A standalone pair has no gold `answer` -- its two sides ARE the
        # responses -- so only the trace field is required of it.
        required = ("reasoning_trace",) if standalone else ("answer", "reasoning_trace")
        if is_blank_record(record) or any(
            _blank((record.chosen or {}).get(field))
            or _blank((record.rejected or {}).get(field))
            for field in required
        ):
            dropped["pref_blank_content"] += 1
            continue
        # The MCQ letter cue is an artefact of exam pairs, where chosen carries
        # an option letter and rejected does not. Standalone pairs are prose.
        record, outcome = normalize_mcq_pair(record)
        mcq_outcomes[outcome] += 1
        if outcome in ("no_match", "no_options"):
            # The letter cannot be recovered, so writing the pair would emit the
            # very format cue this normalisation exists to remove.
            dropped["pref_mcq_cue_unresolved"] += 1
            continue
        pref_modes[record.pref_mode or "n/a"] += 1
        pref_files[PREF_FILES[split_name]].append(
            data_schema.to_pref_row(
                record,
                tokenizer,
                chat.system_for_record(
                    cfg, record.id, exam=data_schema.is_exam(record)
                ),
                tag,
            )
        )

    # Built from the rows actually written, not from the ids considered: a
    # reserved pair that turned out unusable (blank, unresolvable MCQ cue) is not
    # in this set, so it stays in SFT rather than being lost by both stages.
    reserved_ids = {
        str(row["id"]) for rows in pref_files.values() for row in rows
    }
    LOGGER.info(
        "preference rows: %d train, %d val (reserved %d ids from SFT at "
        "preference_holdout_frac=%s); MCQ letter outcomes %s",
        len(pref_files[PREF_FILES[splits.TRAIN]]),
        len(pref_files[PREF_FILES[splits.VAL]]),
        len(reserved_ids),
        preference_holdout_frac,
        dict(sorted(mcq_outcomes.items())),
    )

    # 4. render the supervised rows, minus everything the preference stage
    # claimed. Training rows get the per-id system message, so a deterministic
    # `prompt.variation_rate` fraction carries the short identity; evaluation
    # rows carry no rendered prompt at all (evalrun renders them at eval time
    # with the full identity).
    def sft_row(record: data_schema.CosimoRecord) -> dict:
        return data_schema.to_sft_row(
            record,
            tokenizer,
            chat.system_for_record(cfg, record.id, exam=data_schema.is_exam(record)),
            tag,
        )

    files: dict[str, list[dict]] = {
        SFT_FILES[splits.TRAIN]: [
            sft_row(record)
            for record in train_records
            if record.id not in reserved_ids
        ],
        SFT_FILES[splits.VAL]: [
            sft_row(record)
            for record in by_split[splits.VAL]
            if record.id not in reserved_ids
        ],
        # Exam-only: the graded suites read a final-answer value, which the other
        # four record types do not have. Held-out non-exam records are simply
        # excluded from training and reported in the manifest.
        EVAL_FILES[splits.TEST]: [
            eval_rows[r.id] for r in by_split[splits.TEST] if data_schema.is_exam(r)
        ],
        EVAL_FILES[splits.UNSEEN_STEMS]: [
            eval_rows[r.id]
            for r in by_split[splits.UNSEEN_STEMS]
            if data_schema.is_exam(r)
        ],
        **pref_files,
    }
    LOGGER.info(
        "sft rows: %d train, %d val (%d train rows reserved for the preference "
        "stage)",
        len(files[SFT_FILES[splits.TRAIN]]),
        len(files[SFT_FILES[splits.VAL]]),
        len(train_records) - len(files[SFT_FILES[splits.TRAIN]]),
    )

    # 5. validation gate, before anything reaches disk
    validate(
        files,
        holdout_families=holdout_families,
        families_present=families_present,
        pool_size=pool_size,
        generator_conflicts=conflicts,
        full_system=full_system,
        short_system=short_system,
        exam_protocol=exam_protocol,
        tag=tag,
        eos_token=getattr(tokenizer, "eos_token", None) or "",
    )

    # 6. write
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write_jsonl(out_dir / name, rows)
    config_mod.save_config(cfg, out_dir / CONFIG_FILE)
    runlog.write_json(out_dir / ENV_FILE, runlog.env_info())

    # 7. manifest
    lengths = {
        name: percentiles(
            token_lengths(tokenizer, (row["text"] for row in files[name])),
            max_seq_length,
        )
        for name in SFT_FILES.values()
    }
    for name in PREF_FILES.values():
        lengths[name] = percentiles(
            token_lengths(
                tokenizer, (row["prompt"] + row["chosen"] for row in files[name])
            ),
            max_seq_length,
        )
    # Prompt-only lengths bound DPO/ORPO `max_prompt_length`. With
    # `truncation_mode: keep_end` an over-long prompt is cut from the *start*,
    # which is exactly where the identity block sits.
    prompt_lengths = {
        name: percentiles(
            token_lengths(tokenizer, (row["prompt"] for row in files[name])),
            max_seq_length,
        )
        for name in tuple(SFT_FILES.values()) + tuple(PREF_FILES.values())
    }

    manifest = {
        "created_at": runlog.utc_now(),
        "seed": seed,
        "config_hash": config_mod.config_hash(cfg),
        "dataset": dataset_info,
        "tokenizer_id": tokenizer_id,
        # Which chat template rendered these files: the override strips the
        # vendor "Your name is Phi ... Microsoft" preamble, so a later run can
        # prove it prepared data the same way.
        "chat_template": {
            "path": config_mod.get(cfg, "chat.template_path"),
            "applied": template_applied,
            "sha256": template_hash,
        },
        # Which ids the preference stage claimed. `sft_pref_overlap` must be 0:
        # any id in both files is a DPO pair whose chosen trace SFT already
        # memorised, which contributes no gradient. The validation gate asserts it.
        # `frac` only bites on a shared-id corpus; a standalone pair is disjoint
        # by construction, so `standalone_pairs` rows bypass the reservation.
        "preference_holdout": {
            "frac": preference_holdout_frac,
            "salt": PREFERENCE_HOLDOUT_SALT,
            "reserved_ids": len(reserved_ids),
            "kept_for_sft": dropped.get("pref_kept_for_sft", 0),
            "standalone_pairs": len(standalone_ids),
            "modes": dict(sorted(pref_modes.items())),
            "sft_pref_overlap": len(
                {str(row["id"]) for row in files[SFT_FILES[splits.TRAIN]]}
                & {str(row["id"]) for row in files[PREF_FILES[splits.TRAIN]]}
            ),
        },
        "prompt": {
            "variation_rate": variation_rate,
            "short_identity": {
                name: measure_identity(files[name], full_system, short_system)
                for name in tuple(SFT_FILES.values()) + tuple(PREF_FILES.values())
            },
        },
        "max_seq_length": max_seq_length,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "max_train_records": max_train_records,
        "drop_unverified": drop_unverified,
        "holdout_families": sorted(holdout_families),
        "holdout_records": len(by_split[splits.UNSEEN_STEMS]),
        # Non-exam held-out records are excluded from training but never
        # evaluated -- the graded suites cannot score them. Recorded so the gap
        # between `holdout_records` and the unseen_stems file size is explained
        # rather than looking like rows going missing.
        "holdout_records_not_evaluated": sum(
            1 for r in by_split[splits.UNSEEN_STEMS] if not data_schema.is_exam(r)
        ),
        "generators_present": len(families_present),
        # Where the merged pool came from, after every drop and the max_share
        # caps. The share is what `dataset.mix` was configured to produce.
        "sources": {
            str(source["hub_id"]): {
                "revision": source["revision"],
                "preference_config": source["preference_config"],
                "max_share": source["max_share"],
                "records": sum(
                    1
                    for r in records
                    if source_by_id[r.id] == str(source["hub_id"])
                ),
            }
            for source in sources
        },
        "generator_conflicts": {
            "n": n_conflicts,
            "involving_holdout": len(conflicts),
        },
        "mcq_preference": {
            "outcomes": dict(sorted(mcq_outcomes.items())),
            "written": {
                name: written_letter_prefix_counts(files[name], tag)
                for name in PREF_FILES.values()
            },
        },
        "dropped": dict(sorted(dropped.items())),
        "splits": {name: len(by_split[name]) for name in splits.SPLIT_NAMES},
        "files": {name: len(rows) for name, rows in files.items()},
        "by_program": {
            name: _counts_by(rows, "program") for name, rows in files.items()
        },
        "by_question_type": {
            name: _counts_by(rows, "question_type") for name, rows in files.items()
        },
        # The headline composition number. An SFT corpus that has drifted back to
        # majority-exam is the style-collapse failure returning, and it is
        # visible here before a single GPU-hour is spent.
        "by_record_type": {
            name: _counts_by(rows, "record_type") for name, rows in files.items()
        },
        "token_lengths": lengths,
        "prompt_token_lengths": prompt_lengths,
    }
    runlog.write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


def print_summary(manifest: dict, out_dir: Path) -> None:
    """Compact summary table plus the artifact paths."""
    max_seq_length = manifest["max_seq_length"]
    header = (
        f"{'rows':>9}  {'p50':>6}{'p95':>6}{'p99':>6}{'max':>7}"
        f"{'>' + str(max_seq_length):>8}"
    )
    print(f"\nprepared from {manifest['dataset'].get('hub_id')} -> {out_dir}")
    print(f"{'file (full example)':<32}{header}")
    for name, rows in manifest["files"].items():
        lengths = manifest["token_lengths"].get(name)
        tail = (
            f"  {lengths['p50']:>6}{lengths['p95']:>6}{lengths['p99']:>6}"
            f"{lengths['max']:>7}{lengths['over_max_seq']:>8}"
            if lengths
            else ""
        )
        print(f"{name:<32}{rows:>9}{tail}")

    print(f"\n{'file (prompt only)':<32}{header}")
    for name, lengths in manifest["prompt_token_lengths"].items():
        print(
            f"{name:<32}{lengths['n']:>9}  {lengths['p50']:>6}{lengths['p95']:>6}"
            f"{lengths['p99']:>6}{lengths['max']:>7}{lengths['over_max_seq']:>8}"
        )
    print(
        "prompt lengths bound dpo/orpo max_prompt_length (keep_end truncation "
        "cuts the identity block first)"
    )

    # The persona is ~600-630 tokens on every example, so the headroom under
    # max_seq_length is much smaller than it looks; truncation silently cuts the
    # supervised span, which is the failure this line exists to prevent.
    over = {
        name: lengths["over_max_seq"]
        for name, lengths in manifest["token_lengths"].items()
        if lengths["over_max_seq"]
    }
    if over:
        print(
            f"\nWARNING: rows longer than model.max_seq_length={max_seq_length} "
            f"will be truncated during training: {over}"
        )
    else:
        print(f"\nno row exceeds model.max_seq_length={max_seq_length}")

    print(f"\n{'split':<16}{'rows':>9}")
    for name in splits.SPLIT_NAMES:
        print(f"{name:<16}{manifest['splits'][name]:>9}")

    # Composition is the number to read first: a majority-exam SFT corpus is the
    # style collapse of the first run waiting to happen again.
    sft_types = manifest["by_record_type"][SFT_FILES[splits.TRAIN]]
    total = sum(sft_types.values()) or 1
    print(f"\n{'record type (sft_train)':<24}{'rows':>9}{'share':>9}")
    for name, count in sorted(sft_types.items(), key=lambda kv: -kv[1]):
        print(f"{name or '?':<24}{count:>9}{count / total:>9.1%}")

    print(f"\n{'source':<44}{'records':>9}{'max_share':>11}")
    for hub_id, info in manifest["sources"].items():
        share = "-" if info["max_share"] is None else f"{info['max_share']:.0%}"
        print(f"{hub_id:<44}{info['records']:>9}{share:>11}")

    programs = sorted(
        {
            program
            for counts in manifest["by_program"].values()
            for program in counts
            if program
        }
    )
    columns = list(SFT_FILES.values()) + list(EVAL_FILES.values())
    if programs:
        head = "".join(f"{name.replace('.jsonl', ''):>26}" for name in columns)
        print(f"\n{'program':<20}{head}")
        for program in programs:
            cells = "".join(
                f"{manifest['by_program'][name].get(program, 0):>26}"
                for name in columns
            )
            print(f"{program:<20}{cells}")

    template = manifest["chat_template"]
    dataset = manifest["dataset"]
    mcq = manifest["mcq_preference"]
    print(
        f"\ndataset: {dataset.get('hub_id')}@{dataset.get('revision')} "
        f"sha={dataset.get('resolved_sha')}"
        f"\nchat template: {template['path']} sha256={template['sha256']} "
        f"applied={template['applied']}"
        f"\nshort identity (variation_rate={manifest['prompt']['variation_rate']}): "
        + ", ".join(
            f"{name.replace('.jsonl', '')} {share['short']}/{share['n']}"
            f" ({share['fraction']:.1%})"
            for name, share in manifest["prompt"]["short_identity"].items()
        )
        + f"\nholdout: {len(manifest['holdout_families'])} families, "
        f"{manifest['holdout_records']} records "
        f"({manifest['holdout_records_not_evaluated']} non-exam, held out of "
        "training but not evaluated), "
        f"{manifest['generators_present']} stem families present"
        f"\npreference: {manifest['preference_holdout']['standalone_pairs']} "
        f"standalone pairs, modes {manifest['preference_holdout']['modes']}, "
        f"sft overlap {manifest['preference_holdout']['sft_pref_overlap']}"
        f"\ngenerator conflicts: {manifest['generator_conflicts']}"
        f"\nMCQ preference letter outcomes: {mcq['outcomes']}"
        f"\nMCQ letter prefixes written: {mcq['written']}"
        f"\ndropped: {manifest['dropped']}"
        f"\nmanifest: {out_dir / MANIFEST_FILE}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="seeded sample of N rows per dataset split (fast smoke run)",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="tokenizer to render with (default: model.base_id)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing processed files"
    )
    config_mod.add_config_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    cfg = config_mod.load_config(stage="data", extra=args.config, overrides=args.set)

    out_dir = config_mod.harness_path(
        config_mod.get(cfg, "paths.processed_dir", "data/processed")
    )
    # Checked again inside prepare(); this call is what makes the refusal happen
    # before the download rather than after it.
    check_outputs(out_dir, args.force)

    seed = int(config_mod.get(cfg, "seed", 3407))
    sources = dataset_sources(cfg)
    hub_id = sources[0]["hub_id"]
    revision = sources[0]["revision"]
    tokenizer_id = args.tokenizer_id or config_mod.get(cfg, "model.base_id")
    model_revision = config_mod.get(cfg, "model.revision")

    tokenizer = load_tokenizer(tokenizer_id, model_revision)
    default_rows, pref_rows, fingerprints = load_hub_rows(
        sources, limit=args.limit, seed=seed
    )

    manifest = prepare(
        cfg,
        sources=sources,
        default_rows=default_rows,
        pref_rows=pref_rows,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        dataset_info={
            "hub_id": hub_id,
            "revision": revision,
            "resolved_sha": resolve_dataset_sha(hub_id, revision),
            # `revision: main` is a moving target for every source, not just the
            # primary; a defensible result pins all of them.
            "resolved_shas": {
                str(source["hub_id"]): resolve_dataset_sha(
                    source["hub_id"], source["revision"]
                )
                for source in sources
            },
            "model_revision": model_revision,
            "row_sets": fingerprints,
            "limit": args.limit,
        },
        out_dir=out_dir,
        force=args.force,
    )
    print_summary(manifest, out_dir)


if __name__ == "__main__":
    main()
