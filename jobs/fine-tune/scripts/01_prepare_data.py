#!/usr/bin/env python3
"""Download, normalise, split and render the Cosimo corpus into data/processed/.

Reads `btech-software/cosimo-cfa-frm-71k` (both configs), assigns deterministic
splits, renders every training example through the harness chat template, writes
the JSONL files the training and evaluation scripts consume, and runs a
validation gate that fails loudly on leakage, on a rendering fault, and on a
configuration that did not do what it said it would do.

The gate checks the *output* against the *configuration*, not only against
itself: a holdout family that matched no record, an empty split, a generator
label that disagrees with its verification template, or a preference pair whose
rejected side carries a format cue are all failures, not warnings.

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


def load_hub_rows(
    hub_id: str,
    revision: str | None,
    *,
    limit: int | None = None,
    seed: int = 3407,
) -> tuple[list[dict], list[dict], dict]:
    """Load both dataset configs as plain dicts, plus their row fingerprints.

    ``limit`` takes a *seeded, order-preserving sample* of each split rather than
    a head slice: the Hub stores contiguous ~1000-row blocks per generator, so
    ``[:N]`` would yield a single generator per program and a smoke run that
    exercises almost none of the stratification or the holdout families. The
    preference subset is then derived by joining on the sampled ids, so the two
    configs stay consistent.
    """
    from datasets import load_dataset

    fingerprints: dict[str, dict] = {}

    default_rows: list[dict] = []
    loaded = load_dataset(hub_id, "default", revision=revision)
    for split_name in sorted(loaded):
        dataset = loaded[split_name]
        picked = subsample(list(range(len(dataset))), limit, seed)
        if len(picked) < len(dataset):
            dataset = dataset.select(picked)
        rows = [dict(row) for row in dataset]
        fingerprints[f"default/{split_name}"] = rows_fingerprint(rows)
        default_rows.extend(rows)
        LOGGER.info("loaded default/%s: %d rows", split_name, len(rows))

    sampled_ids = {str(row.get("id", "")) for row in default_rows}
    pref_rows: list[dict] = []
    loaded = load_dataset(hub_id, "preference_pairs", revision=revision)
    for split_name in sorted(loaded):
        rows = [dict(row) for row in loaded[split_name]]
        if limit is not None:
            rows = [row for row in rows if str(row.get("id", "")) in sampled_ids]
        fingerprints[f"preference_pairs/{split_name}"] = rows_fingerprint(rows)
        pref_rows.extend(rows)
        LOGGER.info("loaded preference_pairs/%s: %d rows", split_name, len(rows))

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
    """
    return (
        _blank(record.question)
        or _blank(record.answer)
        or _blank(record.reasoning_trace)
    )


def _counts_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def measure_identity(rows: list[dict], full_system: str, short_system: str) -> dict:
    """Count which identity each written prompt actually carries.

    Measured by substring on the rendered prompt, not re-derived from the id
    hash: a re-derivation would report the intended fraction even if rendering
    ignored it entirely.
    """
    short = sum(1 for row in rows if short_system in row["prompt"])
    full = sum(1 for row in rows if full_system in row["prompt"])
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

    # A preference id legitimately repeats its SFT id, so it cannot join the
    # owner map; what must never happen is a preference row for an evaluated id.
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
    full_system = chat.compose_system(cfg, short=False, exam=True)
    short_system = chat.compose_system(cfg, short=True, exam=True)
    max_seq_length = int(config_mod.get(cfg, "model.max_seq_length", 2048))
    val_frac = float(config_mod.get(cfg, "data.val_frac", 0.01))
    test_frac = float(config_mod.get(cfg, "data.test_frac", 0.01))
    max_train_records = config_mod.get(cfg, "data.max_train_records")
    drop_unverified = bool(config_mod.get(cfg, "data.drop_unverified", True))
    holdout_families = {
        str(f) for f in (config_mod.get(cfg, "data.holdout_families") or [])
    }

    # 1. normalise, dropping unverified, duplicate and degenerate rows
    records: list[data_schema.CosimoRecord] = []
    seen_ids: set[str] = set()
    dropped: Counter = Counter()
    conflicts: list[dict] = []
    n_conflicts = 0
    for row in default_rows:
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
        # The generator decides the stem family, and the stem family decides the
        # holdout. A single mislabelled generator moves a held-out stem into
        # training, which is the one failure that cannot be detected downstream.
        metadata_generator = str((row.get("metadata") or {}).get("generator") or "")
        verification_template = str(
            (row.get("verification") or {}).get("template") or ""
        )
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
        seen_ids.add(record.id)
        records.append(record)
    records.sort(key=lambda record: record.id)
    LOGGER.info(
        "normalised %d records (dropped %s)",
        len(records),
        dict(sorted(dropped.items())),
    )

    # 2. assign splits from the `default` config; the preference rows join on id
    eval_rows = {record.id: data_schema.to_eval_row(record) for record in records}
    families_present = {row["stem_family"] for row in eval_rows.values()}
    assignment = splits.assign_splits(
        list(eval_rows.values()),
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        holdout_families=holdout_families,
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

    # 3. render. Training rows get the per-id system message, so a deterministic
    # `prompt.variation_rate` fraction carries the short identity; evaluation
    # rows carry no rendered prompt at all (evalrun renders them at eval time
    # with the full identity).
    files: dict[str, list[dict]] = {
        SFT_FILES[splits.TRAIN]: [
            data_schema.to_sft_row(
                record, tokenizer, chat.system_for_record(cfg, record.id), tag
            )
            for record in train_records
        ],
        SFT_FILES[splits.VAL]: [
            data_schema.to_sft_row(
                record, tokenizer, chat.system_for_record(cfg, record.id), tag
            )
            for record in by_split[splits.VAL]
        ],
        EVAL_FILES[splits.TEST]: [eval_rows[r.id] for r in by_split[splits.TEST]],
        EVAL_FILES[splits.UNSEEN_STEMS]: [
            eval_rows[r.id] for r in by_split[splits.UNSEEN_STEMS]
        ],
        PREF_FILES[splits.TRAIN]: [],
        PREF_FILES[splits.VAL]: [],
    }

    # The `preference_pairs` config carries no generator column, so the stem
    # family can only come from the joined `default` record.
    generator_by_id = {record.id: record.generator for record in records}
    mcq_outcomes: Counter = Counter()
    for row in sorted(pref_rows, key=lambda row: str(row.get("id", ""))):
        row_id = str(row.get("id", ""))
        if row_id not in generator_by_id:
            dropped["pref_without_default_row"] += 1
            continue
        split_name = assignment[row_id]
        if split_name not in PREF_FILES:
            # Kept apart: the test count is the leak-prevention number, the
            # unseen count is a consequence of the holdout.
            dropped[
                "pref_in_test" if split_name == splits.TEST else "pref_in_unseen_stems"
            ] += 1
            continue
        record = data_schema.normalize_pref_row(
            {**row, "generator": generator_by_id[row_id]}
        )
        if not data_schema.has_preference(record):
            dropped["pref_unusable_pair"] += 1
            continue
        if is_blank_record(record) or any(
            _blank((record.chosen or {}).get(field))
            or _blank((record.rejected or {}).get(field))
            for field in ("answer", "reasoning_trace")
        ):
            dropped["pref_blank_content"] += 1
            continue
        record, outcome = normalize_mcq_pair(record)
        mcq_outcomes[outcome] += 1
        if outcome in ("no_match", "no_options"):
            # The letter cannot be recovered, so writing the pair would emit the
            # very format cue this normalisation exists to remove.
            dropped["pref_mcq_cue_unresolved"] += 1
            continue
        files[PREF_FILES[split_name]].append(
            data_schema.to_pref_row(
                record, tokenizer, chat.system_for_record(cfg, record.id), tag
            )
        )
    LOGGER.info(
        "preference rows: %d train, %d val; MCQ letter outcomes %s",
        len(files[PREF_FILES[splits.TRAIN]]),
        len(files[PREF_FILES[splits.VAL]]),
        dict(sorted(mcq_outcomes.items())),
    )

    # 4. validation gate, before anything reaches disk
    validate(
        files,
        holdout_families=holdout_families,
        families_present=families_present,
        pool_size=pool_size,
        generator_conflicts=conflicts,
        full_system=full_system,
        short_system=short_system,
        tag=tag,
        eos_token=getattr(tokenizer, "eos_token", None) or "",
    )

    # 5. write
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write_jsonl(out_dir / name, rows)
    config_mod.save_config(cfg, out_dir / CONFIG_FILE)
    runlog.write_json(out_dir / ENV_FILE, runlog.env_info())

    # 6. manifest
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
        "generators_present": len(families_present),
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
        f"{manifest['holdout_records']} records, "
        f"{manifest['generators_present']} stem families present"
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
    hub_id = config_mod.get(cfg, "dataset.hub_id")
    revision = config_mod.get(cfg, "dataset.revision")
    tokenizer_id = args.tokenizer_id or config_mod.get(cfg, "model.base_id")
    model_revision = config_mod.get(cfg, "model.revision")

    tokenizer = load_tokenizer(tokenizer_id, model_revision)
    default_rows, pref_rows, fingerprints = load_hub_rows(
        hub_id, revision, limit=args.limit, seed=seed
    )

    manifest = prepare(
        cfg,
        default_rows=default_rows,
        pref_rows=pref_rows,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        dataset_info={
            "hub_id": hub_id,
            "revision": revision,
            "resolved_sha": resolve_dataset_sha(hub_id, revision),
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
